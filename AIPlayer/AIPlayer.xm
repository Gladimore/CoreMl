// =============================================================================
// AIPlayer.xm — standalone AI-only tweak.
//
// Shows a small floating "AI: OFF / AI: ON" button. Tapping it starts/stops
// a screen-capture + Core ML inference loop. Every high-confidence detection
// is logged AND injected into the game as a synthesized swipe via
// HybridTouchSynthesizer (see the "Touch synthesis" MARK block below)
// — the button flashes gold for ~150ms each time a swipe is actually sent,
// so you can visually confirm firing rate.
//
// HybridTouchSynthesizer now comes from the standalone TouchSynthesis
// module (#import "TouchSynthesis.h" below; link against
// TouchSynthesis.dylib) instead of being copy-pasted inline. It builds a
// synthetic UITouch + GSEventProxy-backed UIEvent by hand (Layer B, via
// the private _initWithEvent:touches: initializer) and sends it, then
// unconditionally also dispatches touchesBegan/Moved/Ended/Cancelled:withEvent:
// directly on the hit-tested view (Layer A) — this replaces two earlier
// approaches that didn't work on this device/iOS build: TouchInjector
// (system-HID via IOHIDEventSystemClient, requires jailbreak-level
// SpringBoard access this Sideloadly-injected process doesn't have) and
// InProcessTouchInjector (a KIF-style sendEvent:-only approach confirmed via
// on-device logging to leave event.allTouches.count==0 on every dispatch).
// These are still undocumented private APIs with no stability guarantee
// across iOS versions — validate against on-screen gameplay with AI: ON
// before trusting it in a real run, and watch Console.app for the
// UnityView [Diag] hook (further down this file, NOT in the module —
// it's Unity-specific) confirming touches actually reach UnityView.
//
// IMPORTANT: TouchSynthesis.dylib itself contains no Logos %hook/%group
// directives and therefore no MobileSubstrate dependency, so it's safe
// to link into this non-jailbroken/Sideloadly build. Every hook in THIS
// file (UIWindow(GGMakeKeyAndVisibleHook), the UnityView diagnostic
// swizzle) was already written as plain
// method_exchangeImplementations/method_setImplementation for that same
// reason — see the comments at each for why %hook isn't used here.
//
// PREPROCESSING CONTRACT — must exactly match build_dataset_v3_gpu.py:
//   1. Resize to (kImgSize, kImgSize) via SQUARE STRETCH (not
//      aspect-preserving) — torchvision's TF.resize(..., antialias=True)
//   2. RGB -> grayscale via ITU-R 601-2 luma:
//        gray = 0.2989*R + 0.5870*G + 0.1140*B
//   3. diff[t] = clamp( floorDiv((frame[t]-frame[t-1])*127, 255) + 127, 0, 255 )
//      diff[0] = 127 (neutral). floorDiv is FLOOR division, not truncating —
//      see floorDiv() below, do not replace with a plain `/`.
//   4. fast_frame / slow_frame values passed to the model are RAW 0-255
//      float32 — do NOT divide by 255 in packFrame:diff:. The exported
//      CoreML graph already does `x * (1.0/255.0)` internally (traced from
//      CausalSwipeAnnotator._cnn_features in convert_to_coreml.py's
//      StepWrapper). Normalizing here too silently shrinks the real signal
//      by ~255x relative to the CNN's bias terms and effectively blinds the
//      model to the frame — see packFrame:diff: for the full explanation.
//   5. h_fast_in / h_slow_in are carried indefinitely for the whole session
//      (only zeroed once, in resetSession) — this is the model's intended
//      streaming design (see model_causal.py's step()/init_state() docs),
//      not a bug to "fix" by periodic resets.
//
// If any of this drifts from the training pipeline, inference silently
// produces wrong (but plausible-looking) predictions — no crash, no error.
// =============================================================================

#import <UIKit/UIKit.h>
#import <ReplayKit/ReplayKit.h>
#import <CoreImage/CoreImage.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#include <os/log.h>
#include <math.h>
#include <mach/mach_time.h>
#include <mach-o/dyld.h>
#import "TouchSynthesis.h"   // <- HybridTouchSynthesizer + TouchSynthesisKeyWindow()

// Shared log handle for the tweak. Using a custom subsystem/category makes
// it easy to filter in Console.app / iMazing's console viewer.
static os_log_t AIPlayerLog(void) {
    static os_log_t log;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        log = os_log_create("com.aiplayer.tweak", "AIPlayer");
    });
    return log;
}

// ── Model config — MUST match your checkpoint's arch dict + dataset meta ───
static const NSInteger kImgSize     = 128;   // meta['img_size'] — CONFIRM against your dataset
static const NSInteger kFastLayers  = 2;     // arch['fast_layers']
static const NSInteger kSlowLayers  = 1;     // arch['slow_layers']
static const NSInteger kHidden      = 192;   // arch['hidden']
static const NSInteger kSlowHidden  = kHidden / 2;

static const NSInteger kSlowBranchEveryNTicks = 6;  // VALIDATE against SLOW_OFFSETS spacing (19,13 -> stride 6)
static const float     kDetectionThreshold    = 0.80f;
static const int       kTargetFPS             = 24;

// ── Slow-branch delay line ──────────────────────────────────────────────────
// build_dataset_v3_gpu.py builds slow_x from SLOW_OFFSETS=(19,13): two frames
// strictly older than the fast window, fed to slow_gru in order [-19, -13]
// from a *zero* hidden state, and forward() takes the hidden state after the
// LAST step, i.e. after consuming the -13 frame. Deployment instead carries
// h_slow forward across the whole session (per model_causal.py's documented
// step()/forward() approximation) and only gets ONE new sample per cadence
// tick, so there's no single call that can replicate the two-step
// [-19 -> -13] transition exactly. The closest faithful reproduction is to
// feed a sample that is ALWAYS exactly kSlowDelayTicks behind "now": that
// keeps the step size between consecutive slow samples fixed at exactly
// kSlowBranchEveryNTicks (matching 19->13's spacing) every single time, and
// kSlowDelayTicks=13 additionally means each fed sample sits at the same
// offset as the LAST frame slow_gru saw during training, i.e. the same
// offset whose post-update hidden state forward() actually returns to the
// heads. This is still the approximation the model docstring flags as
// needing empirical validation (compare against forward() on a real
// captured session) — it is not claimed to be exact.
static const NSInteger kSlowDelayTicks      = 13;
static const NSInteger kGrayHistoryCapacity = 32;  // must be > kSlowDelayTicks + 1; 32 gives comfortable margin

// ── Touch injection tuning ──────────────────────────────────────────────────
// kInjectCooldown: minimum gap between two injected swipes. A single real
// swipe can cross kDetectionThreshold on several consecutive ticks (the
// model wasn't trained to emit a single spike), so without a cooldown one
// physical swipe could fire multiple synthesized swipes in a row.
//
// 0.15s, not the original 0.35s guess — checked against a real swipes.csv
// session (5778 labeled swipes): at 0.35s cooldown, 34% of genuine
// consecutive swipe pairs in that session were closer together than the
// cooldown window (median gap 0.50s, but a meaningful tail down to 0.125s
// between two *different*-direction swipes back to back). 0.35s would have
// silently dropped over a third of legitimate rapid inputs, not just
// duplicate detections. 0.15s only sacrifices ~0.2% of that session's real
// swipes and still clears kSwipeDuration below, so it can't cut off a swipe
// still mid-flight. If your game's swipe cadence is denser than this
// session's, re-derive this the same way against your own swipes.csv.
static const NSTimeInterval kInjectCooldown  = 0.15;  // seconds
static const NSTimeInterval kSwipeDuration   = 0.12;  // seconds — one synthesized swipe's down->up span
static const CGFloat        kSwipeMagnitude  = 0.35;  // fraction of min(screen.width, screen.height)

typedef NS_ENUM(NSInteger, SwipeDirection) {
    SwipeDirUp = 0, SwipeDirDown = 1, SwipeDirLeft = 2, SwipeDirRight = 3,
};

// =============================================================================
// MARK: - InferenceEngine
// =============================================================================

@interface SwipePrediction : NSObject
@property (nonatomic) float detProbability;
@property (nonatomic) SwipeDirection direction;
@property (nonatomic) float dirConfidence;
@end
@implementation SwipePrediction @end

@interface InferenceEngine : NSObject
@property (nonatomic, strong, nullable) MLModel *model;
@property (nonatomic, strong, nullable) MLMultiArray *hFast;
@property (nonatomic, strong, nullable) MLMultiArray *hSlow;
@property (nonatomic, strong, nullable) NSData *prevGrayBuffer;
@property (nonatomic, strong) CIContext *ciContext;
// Ring buffer of this session's (gray, diff) history, so the slow branch can
// be fed a genuinely-delayed frame instead of "now". Fixed-capacity,
// pre-filled with NSNull; historyTickCount is the monotonic absolute tick
// count (0-based) of the most recently pushed frame + 1.
@property (nonatomic, strong, nullable) NSMutableArray<id> *grayHistory;
@property (nonatomic, strong, nullable) NSMutableArray<id> *diffHistory;
@property (nonatomic, assign) NSInteger historyTickCount;
+ (instancetype)sharedEngine;
- (BOOL)loadModelAtURL:(NSURL *)url error:(NSError **)error;
- (void)resetSession;
- (nullable SwipePrediction *)predictWithPixelBuffer:(CVPixelBufferRef)pb
                                     hasNewSlowFrame:(BOOL)hasNewSlow
                                                error:(NSError **)error;
@end

@implementation InferenceEngine

+ (instancetype)sharedEngine {
    static InferenceEngine *inst = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ inst = [InferenceEngine new]; });
    return inst;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _ciContext = [CIContext contextWithOptions:@{
            kCIContextUseSoftwareRenderer: @NO,
            kCIContextWorkingColorSpace: (__bridge_transfer id)CGColorSpaceCreateDeviceRGB(),
        }];
    }
    return self;
}

- (BOOL)loadModelAtURL:(NSURL *)url error:(NSError **)error {
    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsAll;
    self.model = [MLModel modelWithContentsOfURL:url configuration:config error:error];
    if (!self.model) return NO;
    [self resetSession];
    return YES;
}

- (void)resetSession {
    self.hFast = [self zerosShape:@[@(kFastLayers), @1, @(kHidden)]];
    self.hSlow = [self zerosShape:@[@(kSlowLayers), @1, @(kSlowHidden)]];
    self.prevGrayBuffer = nil;

    // New session -> no valid history yet. Re-fill with NSNull rather than
    // leaving stale frames from a previous session sitting in old slots;
    // historyTickCount=0 means predictWithPixelBuffer: won't attempt to read
    // a delayed slow sample until kSlowDelayTicks real frames have been
    // pushed (see the haveEnoughHistory gate there).
    self.grayHistory = [NSMutableArray arrayWithCapacity:kGrayHistoryCapacity];
    self.diffHistory  = [NSMutableArray arrayWithCapacity:kGrayHistoryCapacity];
    for (NSInteger i = 0; i < kGrayHistoryCapacity; i++) {
        [self.grayHistory addObject:[NSNull null]];
        [self.diffHistory  addObject:[NSNull null]];
    }
    self.historyTickCount = 0;
}

- (MLMultiArray *)zerosShape:(NSArray<NSNumber *> *)shape {
    NSError *err = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&err];
    memset((float *)arr.dataPointer, 0, arr.count * sizeof(float));
    return arr;
}

- (nullable NSData *)grayscaleFromPixelBuffer:(CVPixelBufferRef)pb {
    CIImage *input = [CIImage imageWithCVPixelBuffer:pb];
    CGRect extent = input.extent;
    if (CGRectIsEmpty(extent)) return nil;

    // SQUARE STRETCH — independent x/y scale, matches TF.resize's
    // non-aspect-preserving square resize. Do not swap for a uniform scale.
    CGFloat sx = (CGFloat)kImgSize / extent.size.width;
    CGFloat sy = (CGFloat)kImgSize / extent.size.height;
    CIImage *stretched = [input imageByApplyingTransform:CGAffineTransformMakeScale(sx, sy)];

    size_t bytesPerRow = kImgSize * 4;
    NSMutableData *rgba = [NSMutableData dataWithLength:bytesPerRow * kImgSize];
    [self.ciContext render:stretched
                   toBitmap:rgba.mutableBytes
                 rowBytes:bytesPerRow
                   bounds:CGRectMake(0, 0, kImgSize, kImgSize)
                   format:kCIFormatRGBA8
               colorSpace:CGColorSpaceCreateDeviceRGB()];

    NSMutableData *gray = [NSMutableData dataWithLength:kImgSize * kImgSize];
    const uint8_t *src = (const uint8_t *)rgba.bytes;
    uint8_t *dst = (uint8_t *)gray.mutableBytes;
    for (NSInteger i = 0; i < kImgSize * kImgSize; i++) {
        float l = 0.2989f * src[i*4] + 0.5870f * src[i*4+1] + 0.1140f * src[i*4+2];
        dst[i] = (uint8_t)fminf(255.0f, fmaxf(0.0f, roundf(l)));
    }
    return gray;
}

static inline int32_t floorDiv(int32_t a, int32_t b) {
    int32_t q = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q--;
    return q;
}

- (NSData *)diffFromCurrent:(NSData *)cur previous:(nullable NSData *)prevOrNil {
    NSMutableData *diff = [NSMutableData dataWithLength:kImgSize * kImgSize];
    const uint8_t *c = (const uint8_t *)cur.bytes;
    const uint8_t *p = prevOrNil ? (const uint8_t *)prevOrNil.bytes : c; // start-of-session: neutral
    uint8_t *dst = (uint8_t *)diff.mutableBytes;
    for (NSInteger i = 0; i < kImgSize * kImgSize; i++) {
        int32_t d = (int32_t)c[i] - (int32_t)p[i];
        int32_t s = floorDiv(d * 127, 255) + 127;
        dst[i] = (uint8_t)MAX(0, MIN(255, s));
    }
    return diff;
}

- (MLMultiArray *)packFrame:(NSData *)gray diff:(NSData *)diff {
    // DO NOT divide by 255 here. The exported CoreML model's forward pass
    // (StepWrapper -> CausalSwipeAnnotator._cnn_features) does
    // `x = x.to(float32) * (1.0/255.0)` INTERNALLY — that normalization is
    // traced straight into the .mlmodel graph. This function must hand the
    // model raw 0-255 values (as float32), exactly like fast_frame's input
    // description says: "normalized 0-255 uint8 range". Dividing by 255
    // here on top of that produces inputs in ~[0, 0.0039] instead of
    // [0, 255] — a ~255x scale-down that drowns the real per-pixel signal
    // under the CNN's (scale-invariant-to-this-bug) bias terms, leaving the
    // GRU fed an almost frame-invariant feature vector every tick. That's
    // the double-normalization bug that was causing det/dirConf to freeze
    // near-flat within the first second and stay there regardless of
    // on-screen content — see AIPlayer.xm diagnosis notes.
    NSError *err = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:@[@1, @2, @(kImgSize), @(kImgSize)]
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&err];
    float *dst = (float *)arr.dataPointer;
    const uint8_t *g = (const uint8_t *)gray.bytes, *d = (const uint8_t *)diff.bytes;
    NSInteger n = kImgSize * kImgSize;
    for (NSInteger i = 0; i < n; i++) {
        dst[i]     = (float)g[i];
        dst[n + i] = (float)d[i];
    }
    return arr;
}

- (nullable SwipePrediction *)predictWithPixelBuffer:(CVPixelBufferRef)pb
                                     hasNewSlowFrame:(BOOL)hasNewSlow
                                                error:(NSError **)error {
    if (!self.model) {
        if (error) *error = [NSError errorWithDomain:@"AIPlayer" code:1
                                             userInfo:@{NSLocalizedDescriptionKey: @"Model not loaded"}];
        return nil;
    }

    NSData *gray = [self grayscaleFromPixelBuffer:pb];
    if (!gray) return nil;
    NSData *diff = [self diffFromCurrent:gray previous:self.prevGrayBuffer];
    self.prevGrayBuffer = gray;

    // Push this tick's (gray, diff) into the ring buffer BEFORE computing the
    // delayed slow sample below, so nowTick already reflects "this frame" —
    // matches diffs.bin's semantics exactly, since `diff` here is already
    // the ordinary frame[t]-frame[t-1] causal diff (same formula as
    // build_dataset_v3_gpu.py), just cached per-tick instead of recomputed.
    NSInteger slot = self.historyTickCount % kGrayHistoryCapacity;
    self.grayHistory[slot] = gray;
    self.diffHistory[slot] = diff;
    NSInteger nowTick = self.historyTickCount;   // 0-based absolute tick of THIS frame
    self.historyTickCount++;

    MLMultiArray *fastFrame = [self packFrame:gray diff:diff];

    // Slow branch: feed a genuinely kSlowDelayTicks-old frame, never "now".
    // BUG FIX: this used to be `hasNewSlow ? fastFrame : zeros` — i.e. on
    // every cadence tick it fed the model's own current frame as the "long
    // range context" sample, which is exactly what the slow branch was
    // never trained to see (see kSlowDelayTicks comment above). Gate on
    // real history depth too: early in a session there aren't yet
    // kSlowDelayTicks frames to look back on, so treat those cadence ticks
    // as "no slow sample" (has_slow=0) rather than reading garbage/stale
    // NSNull slots — h_slow simply stays at zero a little longer, which is
    // harmless and self-corrects within ~0.5s.
    BOOL haveEnoughHistory = nowTick >= kSlowDelayTicks;
    BOOL shouldFeedSlow = hasNewSlow && haveEnoughHistory;

    MLMultiArray *slowFrame;
    if (shouldFeedSlow) {
        NSInteger delayedTick = nowTick - kSlowDelayTicks;
        NSData *delayedGray = self.grayHistory[delayedTick % kGrayHistoryCapacity];
        NSData *delayedDiff = self.diffHistory[delayedTick % kGrayHistoryCapacity];
        slowFrame = [self packFrame:delayedGray diff:delayedDiff];
    } else {
        slowFrame = [self zerosShape:@[@1, @2, @(kImgSize), @(kImgSize)]];
    }
    MLMultiArray *hasSlowArr = [self zerosShape:@[@1]];
    hasSlowArr[0] = shouldFeedSlow ? @1.0f : @0.0f;

    NSDictionary *inputs = @{
        @"fast_frame": [MLFeatureValue featureValueWithMultiArray:fastFrame],
        @"slow_frame": [MLFeatureValue featureValueWithMultiArray:slowFrame],
        @"has_slow":   [MLFeatureValue featureValueWithMultiArray:hasSlowArr],
        @"h_fast_in":  [MLFeatureValue featureValueWithMultiArray:self.hFast],
        @"h_slow_in":  [MLFeatureValue featureValueWithMultiArray:self.hSlow],
    };
    MLDictionaryFeatureProvider *provider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs error:error];
    if (!provider) return nil;

    id<MLFeatureProvider> out = [self.model predictionFromFeatures:provider error:error];
    if (!out) return nil;

    MLMultiArray *detLogits = [out featureValueForName:@"det_logits"].multiArrayValue;
    MLMultiArray *dirLogits = [out featureValueForName:@"dir_logits"].multiArrayValue;
    self.hFast = [out featureValueForName:@"h_fast_out"].multiArrayValue;
    self.hSlow = [out featureValueForName:@"h_slow_out"].multiArrayValue;

    float detProb = 1.0f / (1.0f + expf(-detLogits[0].floatValue));

    float dl[4]; for (int i = 0; i < 4; i++) dl[i] = dirLogits[i].floatValue;
    float maxL = fmaxf(fmaxf(dl[0], dl[1]), fmaxf(dl[2], dl[3]));
    float ex[4], sum = 0; for (int i = 0; i < 4; i++) { ex[i] = expf(dl[i]-maxL); sum += ex[i]; }
    int best = 0; float bestP = 0;
    for (int i = 0; i < 4; i++) { float p = ex[i]/sum; if (p > bestP) { bestP = p; best = i; } }

    SwipePrediction *pred = [SwipePrediction new];
    pred.detProbability = detProb;
    pred.direction = (SwipeDirection)best;
    pred.dirConfidence = bestP;
    return pred;
}

@end

// =============================================================================
// MARK: - Touch synthesis
//
// The synthesis engine itself (HybridTouchSynthesizer, GSEventProxy, the
// UITouch window/view fallback swizzle, TouchSynthesisKeyWindow()) now
// lives in the TouchSynthesis module (see #import "TouchSynthesis.h"
// near the top of this file) instead of being duplicated inline here.
// That module contains no MobileSubstrate/Logos %hook dependency, so it
// is safe to link against a non-jailbroken/Sideloadly deployment like
// this one -- see TouchSynthesis.xm's own header comment for why that
// matters.
//
// What's left below is unchanged from before, just re-pointed at the
// module instead of a local copy: FloatingSwipeButtonManager (the demo
// "Swipe" button + the -executeSingleShotSwipeFrom:to:duration: helper
// that -injectSwipeForDirection:confidence:detection: further down this
// file calls to actually deliver the AI's predicted swipes), the
// UIWindow(GGMakeKeyAndVisibleHook) swizzle that shows that button, and
// the UnityView touch-diagnostic swizzle + its install-retry loop. None
// of those three use Logos %hook either -- they were already written as
// plain method_exchangeImplementations/method_setImplementation swizzles
// for the same non-jailbroken reason, so they're unaffected by this
// change other than one mechanical rename throughout: the old local
// key-window helper now calls the module's TouchSynthesisKeyWindow(),
// and logging now goes through this file's own pre-existing
// AIPlayerLog() instead of a second, separate log handle (no need for
// two logging subsystems now that the engine has moved into the
// module, which keeps its own private one internally).
// =============================================================================
@interface FloatingSwipeButtonManager : NSObject
+ (instancetype)sharedInstance;
- (void)showOverlayButton;
// Declaration added (method body below is unedited) so
// -injectSwipeForDirection:confidence:detection: elsewhere in this file can
// call it without a -Werror "may not respond to selector" build failure.
- (void)executeSingleShotSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration;
@end

@implementation FloatingSwipeButtonManager {
    UIButton *_actionButton;
    UITouch *_currentPersistentTouch;
}

+ (instancetype)sharedInstance {
    static FloatingSwipeButtonManager *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[FloatingSwipeButtonManager alloc] init];
    });
    return instance;
}

- (void)dealloc {
    // Demonstrates -finalizeAnyActiveTouch: usage: if the button (and thus
    // this manager's lifecycle) goes away mid-swipe, make sure the app
    // isn't left thinking a finger is still down, matching the binary's
    // own session-teardown behavior rather than silently dropping the touch.
    if (_currentPersistentTouch) {
        // ARC won't let you pass &ivar directly to an __autoreleasing
        // out-param ("passing address of non-local object to
        // __autoreleasing parameter for write-back") — route through a
        // local temp instead.
        UITouch *temp = _currentPersistentTouch;
        [[HybridTouchSynthesizer sharedInstance] finalizeAnyActiveTouch:&temp];
        _currentPersistentTouch = temp;
    }
}

- (void)showOverlayButton {
    UIWindow *keyWindow = TouchSynthesisKeyWindow();
    if (!keyWindow || _actionButton) {
        os_log(AIPlayerLog(), "[Touch]: showOverlayButton aborted — keyWindow=%{public}@ existingButton=%{public}@", keyWindow, _actionButton);
        return;
    }

    _actionButton = [UIButton buttonWithType:UIButtonTypeCustom];
    _actionButton.frame = CGRectMake(20, 100, 70, 70);
    _actionButton.backgroundColor = [UIColor colorWithRed:0.9 green:0.2 blue:0.2 alpha:0.9];
    _actionButton.layer.cornerRadius = 35;
    [_actionButton setTitle:@"Swipe" forState:UIControlStateNormal];

    // Register Exclusion View
    [HybridTouchSynthesizer sharedInstance].excludedTouchView = _actionButton;

    UIPanGestureRecognizer *pan = [[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(handlePan:)];
    [_actionButton addGestureRecognizer:pan];
    [_actionButton addTarget:self action:@selector(triggerSwipe) forControlEvents:UIControlEventTouchUpInside];

    [keyWindow addSubview:_actionButton];
    os_log(AIPlayerLog(), "[Touch]: overlay button added to %{public}@", keyWindow);
}

- (void)handlePan:(UIPanGestureRecognizer *)pan {
    CGPoint translation = [pan translationInView:pan.view.superview];
    pan.view.center = CGPointMake(pan.view.center.x + translation.x, pan.view.center.y + translation.y);
    [pan setTranslation:CGPointZero inView:pan.view.superview];
}

- (void)triggerSwipe {
    UIWindow *keyWindow = TouchSynthesisKeyWindow();
    if (!keyWindow) return;

    // If a previous swipe never reached Ended/Cancelled for some reason,
    // finalize it before starting a new one instead of silently leaking it.
    if (_currentPersistentTouch) {
        UITouch *temp = _currentPersistentTouch;
        [[HybridTouchSynthesizer sharedInstance] finalizeAnyActiveTouch:&temp];
        _currentPersistentTouch = temp;
    }

    NSInteger direction = arc4random_uniform(4);
    CGRect bounds = keyWindow.bounds;
    CGFloat margin = 100.0;
    CGFloat midX = CGRectGetMidX(bounds);
    CGFloat midY = CGRectGetMidY(bounds);

    CGPoint start, end;
    if (direction == 0)      { start = CGPointMake(midX, bounds.size.height - margin); end = CGPointMake(midX, margin); } // Up
    else if (direction == 1) { start = CGPointMake(midX, margin); end = CGPointMake(midX, bounds.size.height - margin); } // Down
    else if (direction == 2) { start = CGPointMake(bounds.size.width - margin, midY); end = CGPointMake(margin, midY);  } // Left
    else                     { start = CGPointMake(margin, midY); end = CGPointMake(bounds.size.width - margin, midY);  } // Right

    os_log(AIPlayerLog(), "[Touch]: triggerSwipe — direction=%ld start=(%.1f,%.1f) end=(%.1f,%.1f)",
          (long)direction, start.x, start.y, end.x, end.y);
    [self executeSingleShotSwipeFrom:start to:end duration:0.3];
}

- (void)executeSingleShotSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration {
    int steps = 15;
    NSTimeInterval stepDelay = duration / steps;
    HybridTouchSynthesizer *synth = [HybridTouchSynthesizer sharedInstance];

    UITouch *beganTouch = _currentPersistentTouch;
    [synth dispatchHybridTouchAtPoint:start phase:UITouchPhaseBegan inOutTouch:&beganTouch];
    _currentPersistentTouch = beganTouch;

    for (int i = 1; i <= steps; i++) {
        CGFloat progress = (CGFloat)i / (CGFloat)steps;
        CGPoint currentPoint = CGPointMake(
            start.x + (end.x - start.x) * progress,
            start.y + (end.y - start.y) * progress
        );

        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(i * stepDelay * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
            UITouch *movedTouch = self->_currentPersistentTouch;
            [synth dispatchHybridTouchAtPoint:currentPoint phase:UITouchPhaseMoved inOutTouch:&movedTouch];
            self->_currentPersistentTouch = movedTouch;

            if (i == steps) {
                dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.01 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
                    UITouch *endedTouch = self->_currentPersistentTouch;
                    [synth dispatchHybridTouchAtPoint:currentPoint phase:UITouchPhaseEnded inOutTouch:&endedTouch];
                    self->_currentPersistentTouch = endedTouch;
                });
            }
        });
    }
}
@end

// ---------------------------------------------------------
// NOT a Logos %hook — see the note above GG_SwizzleClassMethod further
// down for why (short version: MSHookMessageEx needs MobileSubstrate on
// the device, which this Sideloadly/non-jailbroken deployment doesn't
// have). Same method_exchangeImplementations idiom as UITouch
// (TouchSynthesisFallback) above, just applied to UIWindow instead.
// ---------------------------------------------------------
@interface UIWindow (GGMakeKeyAndVisibleHook)
- (void)gg_makeKeyAndVisible;
@end

@implementation UIWindow (GGMakeKeyAndVisibleHook)

+ (void)load {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        Method original = class_getInstanceMethod(self, @selector(makeKeyAndVisible));
        Method swizzled = class_getInstanceMethod(self, @selector(gg_makeKeyAndVisible));
        if (original && swizzled) {
            method_exchangeImplementations(original, swizzled);
        }
        os_log(AIPlayerLog(), "[Touch]: -[UIWindow makeKeyAndVisible] swizzle %{public}@",
              (original && swizzled) ? @"installed" : @"FAILED — check selector names");
    });
}

// Post-swap, sending -gg_makeKeyAndVisible to self actually runs the
// ORIGINAL -makeKeyAndVisible implementation (classic swizzle idiom, same
// as -gg_touchSynthesis_window/-view above) — this is not infinite recursion.
- (void)gg_makeKeyAndVisible {
    [self gg_makeKeyAndVisible];
    os_log(AIPlayerLog(), "[Touch]: -[UIWindow makeKeyAndVisible] hook fired for %{public}@ — scheduling overlay button in 1s", self);
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(1.0 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        [[FloatingSwipeButtonManager sharedInstance] showOverlayButton];
    });
}

@end

// ---------------------------------------------------------
// DIAGNOSTIC ONLY — not part of the synthesis engine itself.
// Confirms whether the synthetic touch actually reaches Unity's own
// touch handlers on UnityView, or gets intercepted/dropped somewhere
// between -[UIWindow sendEvent:] and UnityView's override.
//
// IMPORTANT (found via the previous test — zero [Diag] lines fired even
// though hitTest: later found a genuine UnityView instance): %hook
// blocks install at library-LOAD time by default. UnityView is a class
// defined inside UnityFramework, which very plausibly hasn't registered
// its Objective-C classes yet at the moment this dylib is injected —
// NSClassFromString(@"UnityView") returns nil at that point, so Logos
// silently skips installing the hook (no crash, no log). UIWindow is a
// system class that's always present, which is why that hook installed
// fine while this one didn't. Put in a named group and initialized
// lazily/retried below instead of relying on the automatic ctor.
// ---------------------------------------------------------
// ---------------------------------------------------------
// GG_SwizzleClassMethod: swaps in a replacement IMP for a selector on a
// class that's only resolvable at RUNTIME (UnityView lives inside
// UnityFramework — there's no compile-time header for it, so the
// category+method_exchangeImplementations idiom used for UIWindow/UITouch
// above doesn't apply directly; there's no selector to declare a category
// method under). method_setImplementation is the standard equivalent for
// this case: it swaps the method's IMP in place and hands back the
// original IMP directly, which the replacement below calls to chain
// through — the swizzle-on-an-unknown-class version of Logos's %orig.
//
// NOT a Logos %hook, same reason as the UIWindow swizzle above: %hook
// compiles to MSHookMessageEx, which needs MobileSubstrate installed on
// the device (and which is also what emitted the
// `.linker_option "-framework CydiaSubstrate"` directive that broke the CI
// link step even under a plain `library` Makefile target — that directive
// comes from Logos's generated code itself, not anything in this Makefile,
// so switching Makefile target types alone can't fix it). This project's
// deployment target is non-jailbroken/Sideloadly-injected, so
// MobileSubstrate isn't just unnecessary here, it's actually absent on
// device — %hook's generated code would fail to resolve at load time even
// if the build itself succeeded. method_setImplementation/
// method_exchangeImplementations need nothing beyond the Objective-C
// runtime itself, which is always present.
// ---------------------------------------------------------
typedef void (*GGTouchesEventIMP)(id, SEL, NSSet *, UIEvent *);

static GGTouchesEventIMP GG_OrigTouchesBegan;
static GGTouchesEventIMP GG_OrigTouchesMoved;
static GGTouchesEventIMP GG_OrigTouchesEnded;
static GGTouchesEventIMP GG_OrigTouchesCancelled;

static void GG_Diag_TouchesBegan(id self, SEL _cmd, NSSet *touches, UIEvent *event) {
    os_log(AIPlayerLog(), "[Diag]: UnityView touchesBegan fired — count=%lu touches=%{public}@",
           (unsigned long)touches.count, touches);
    if (GG_OrigTouchesBegan) GG_OrigTouchesBegan(self, _cmd, touches, event);
}
static void GG_Diag_TouchesMoved(id self, SEL _cmd, NSSet *touches, UIEvent *event) {
    os_log(AIPlayerLog(), "[Diag]: UnityView touchesMoved fired — count=%lu", (unsigned long)touches.count);
    if (GG_OrigTouchesMoved) GG_OrigTouchesMoved(self, _cmd, touches, event);
}
static void GG_Diag_TouchesEnded(id self, SEL _cmd, NSSet *touches, UIEvent *event) {
    os_log(AIPlayerLog(), "[Diag]: UnityView touchesEnded fired — count=%lu", (unsigned long)touches.count);
    if (GG_OrigTouchesEnded) GG_OrigTouchesEnded(self, _cmd, touches, event);
}
static void GG_Diag_TouchesCancelled(id self, SEL _cmd, NSSet *touches, UIEvent *event) {
    os_log(AIPlayerLog(), "[Diag]: UnityView touchesCancelled fired — count=%lu", (unsigned long)touches.count);
    if (GG_OrigTouchesCancelled) GG_OrigTouchesCancelled(self, _cmd, touches, event);
}

// Returns NO (leaving *origOut untouched) if `cls` doesn't implement
// `selector` at all, so a caller can tell a genuine miss apart from success.
static BOOL GG_SwizzleClassMethod(Class cls, SEL selector, IMP replacementIMP, IMP *origOut) {
    Method m = class_getInstanceMethod(cls, selector);
    if (!m) return NO;
    *origOut = method_setImplementation(m, replacementIMP);
    return YES;
}

// Watch for these three outcomes in the log during a triggered swipe,
// once "[Diag]: UnityView touch diagnostics install succeeded" confirms
// the swizzle actually went in:
//   1. No touchesBegan/Moved/Ended [Diag] lines despite install
//      succeeding -> the touch really isn't reaching UnityView's
//      handlers even though the hook is live. Worth testing routing
//      through [[UIApplication sharedApplication] sendEvent:] instead
//      of the window directly, in case Unity's actual capture point
//      sits at the UIApplication level rather than window/view delivery.
//   2. [Diag] lines appear with the expected phase sequence and touch
//      count=1 -> the touch is genuinely reaching Unity's native input
//      code, and whatever is blocking the swipe is downstream of
//      Objective-C entirely (uninitialized UITouch fields like `type`/
//      `majorRadius`/`force`, or app-side touch validation).
//   3. [Diag] lines appear but with an unexpected count/phase pattern
//      -> the touch is arriving malformed in some way not yet
//      identified; compare against a REAL manual swipe's [Diag] output
//      as a baseline to see what differs.

// Recursively retries every 0.5s (up to ~15s) until UnityFramework has
// registered UnityView as a runtime class, then installs the swizzles.
// Safe to call repeatedly — GG_SwizzleClassMethod is idempotent enough
// for this purpose (each call just re-swaps the current IMP), and this
// only ever reaches the install branch once in practice, on the attempt
// where the class first exists.
//
// Logs unconditionally on EVERY attempt (not just the two terminal
// outcomes) — an earlier version only logged "found" or "gave up", which
// couldn't distinguish "never called at all" from "retry chain silently
// stalled partway through" from "class found but the install itself
// failed silently." This version removes that blind spot.
static void GG_TryInstallUnityViewDiagnosticHook(int attemptsRemaining) {
    Class unityViewClass = NSClassFromString(@"UnityView");
    os_log(AIPlayerLog(), "[Diag]: retry check — attemptsRemaining=%d classFound=%{public}@",
           attemptsRemaining, unityViewClass ? @"YES" : @"NO");

    if (unityViewClass) {
        os_log(AIPlayerLog(), "[Diag]: class found — installing UnityView touch diagnostics now");
        BOOL ok = YES;
        ok &= GG_SwizzleClassMethod(unityViewClass, @selector(touchesBegan:withEvent:),
                                     (IMP)GG_Diag_TouchesBegan, (IMP *)&GG_OrigTouchesBegan);
        ok &= GG_SwizzleClassMethod(unityViewClass, @selector(touchesMoved:withEvent:),
                                     (IMP)GG_Diag_TouchesMoved, (IMP *)&GG_OrigTouchesMoved);
        ok &= GG_SwizzleClassMethod(unityViewClass, @selector(touchesEnded:withEvent:),
                                     (IMP)GG_Diag_TouchesEnded, (IMP *)&GG_OrigTouchesEnded);
        ok &= GG_SwizzleClassMethod(unityViewClass, @selector(touchesCancelled:withEvent:),
                                     (IMP)GG_Diag_TouchesCancelled, (IMP *)&GG_OrigTouchesCancelled);
        os_log(AIPlayerLog(), "[Diag]: UnityView touch diagnostics install %{public}@ (attemptsRemaining=%d)",
               ok ? @"succeeded" : @"PARTIALLY FAILED — check selector availability", attemptsRemaining);
        return;
    }
    if (attemptsRemaining <= 0) {
        os_log(AIPlayerLog(), "[Diag]: UnityView class never appeared after retries — giving up on diagnostic hook");
        return;
    }
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.5 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        GG_TryInstallUnityViewDiagnosticHook(attemptsRemaining - 1);
    });
}

// UIWindow's swizzle installs itself via +load automatically (see
// UIWindow(GGMakeKeyAndVisibleHook) above) — this constructor only needs
// to log the build marker and kick off the UnityView retry loop.
__attribute__((constructor))
static void GG_Init(void) {
    // BUILD MARKER: logs unconditionally, first thing, with the actual
    // compile-time date/time baked in via __DATE__/__TIME__. If this
    // exact line (or a fresher timestamp than expected) never shows up
    // in the device log after a rebuild+reinstall+relaunch, the running
    // binary is stale — none of the logic below it ran at all. Compare
    // the printed timestamp against when you actually rebuilt.
    os_log(AIPlayerLog(), "[Touch]: ===== BUILD MARKER: compiled %{public}s %{public}s =====", __DATE__, __TIME__);
    GG_TryInstallUnityViewDiagnosticHook(30); // ~15s of retries for UnityDiagnostics
}


// =============================================================================
// MARK: - Passthrough window
//
// FIX: the previous version put the hitTest: override on a plain UIView
// (AIPassthroughView) that was merely the root view controller's view
// *inside* a stock UIWindow. That only affects view-level hit-testing.
// UIWindow itself is what UIApplication asks first when routing a touch,
// and a stock UIWindow made visible (hidden = NO) can end up key and
// swallow the whole event before your view's hitTest: is ever consulted
// for anything beyond "does this window want the point".
//
// AIOverlayWindow overrides hitTest: at the WINDOW level, so a touch
// outside the button falls through to the window underneath (the game's
// own key window) exactly the same way the proven-working GrayHueCapture
// tweak's CaptureOverlayWindow does. It also explicitly refuses to become
// the key window, so the game's window keeps first-responder/key status
// at all times — this window only ever exists to host the toggle button.
// =============================================================================

@interface AIOverlayWindow : UIWindow
@end

@implementation AIOverlayWindow

- (UIView *)hitTest:(CGPoint)p withEvent:(UIEvent *)e {
    UIView *hit = [super hitTest:p withEvent:e];
    return (hit == self) ? nil : hit;
}

// Never let this overlay window take key status — the game's own window
// must stay key so its touch/gesture handling keeps working normally.
- (BOOL)canBecomeKeyWindow { return NO; }

@end

// =============================================================================
// MARK: - Passthrough view (lets touches fall through to the game)
// =============================================================================

@interface AIPassthroughView : UIView @end
@implementation AIPassthroughView
- (UIView *)hitTest:(CGPoint)p withEvent:(UIEvent *)e {
    UIView *hit = [super hitTest:p withEvent:e];
    return (hit == self) ? nil : hit;
}
@end

// =============================================================================
// MARK: - Overlay: single start/stop button
// =============================================================================

// =============================================================================
// MARK: - Resource bundle lookup
//
// DEPLOYMENT MODEL: non-jailbroken, sideloaded via Sideloadly's "inject
// dylib/framework/bundle" option -- NOT a jailbreak dpkg .deb install.
// There is no MobileSubstrate, no /var/jb, no root-owned /Library/... tree
// on this device; none of that exists to look inside. AIPlayer.dylib and
// SwipeAnnotator.mlmodelc are instead injected straight into the app's own
// .app bundle (most commonly under "<App>.app/Frameworks/", alongside each
// other) before Sideloadly re-signs and installs the IPA.
//
// ROOT CAUSE (of the earlier "model not found" failures): this file used
// to derive a jailbreak-style root from the dylib's load path and look for
// "<jb root>/Library/Application Support/AIPlayer/AIPlayer.bundle/...".
// Under Sideloadly injection there is no such root and no such tree --
// every one of those checks (including the recursive scan of
// "/Library/Application Support") was guaranteed to fail on this install
// method, which is exactly what the device logs showed: every fallback
// exhausted, model never found.
//
// Fix: never assume a jailbreak-style root. Resolve everything relative to
// two anchors that are valid regardless of install method:
//   (a) the directory AIPlayer.dylib itself physically loaded from (works
//       under injection, TrollStore, or a real jailbreak alike), and
//   (b) [NSBundle mainBundle], i.e. the app's own .app directory.
// Try the realistic candidate layouts under those anchors, then fall back
// to a small bounded scan of the .app bundle itself (the only place
// something Sideloadly-injected could plausibly be) before giving up.
// =============================================================================

static NSString *AIPlayerDylibDirectory(void) {
    uint32_t count = _dyld_image_count();
    NSString *lastResortMatch = nil;
    for (uint32_t i = 0; i < count; i++) {
        const char *name = _dyld_get_image_name(i);
        if (!name) continue;
        NSString *path = [NSString stringWithUTF8String:name];
        if ([path.lastPathComponent isEqualToString:@"AIPlayer.dylib"])
            return path.stringByDeletingLastPathComponent;
        if ([path.lastPathComponent.lowercaseString containsString:@"aiplayer"] &&
            [path.pathExtension.lowercaseString isEqualToString:@"dylib"]) {
            lastResortMatch = path.stringByDeletingLastPathComponent;
        }
    }
    return lastResortMatch;
}

// Recursively searches `root` for a directory literally named `filename`,
// depth-limited so a search rooted somewhere unexpectedly large can't hang
// startup. Fallback only -- see comment block above.
static NSString *AIPlayerFindDirectoryNamed(NSString *filename, NSString *root, NSInteger maxDepth) {
    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL isDir = NO;
    if (![fm fileExistsAtPath:root isDirectory:&isDir] || !isDir) return nil;

    NSDirectoryEnumerator<NSURL *> *e = [fm
        enumeratorAtURL:[NSURL fileURLWithPath:root isDirectory:YES]
        includingPropertiesForKeys:@[NSURLIsDirectoryKey]
        options:NSDirectoryEnumerationSkipsHiddenFiles
        errorHandler:^BOOL(NSURL *url, NSError *error) { return YES; }];
    if (!e) return nil;

    NSInteger rootDepth = root.pathComponents.count;
    for (NSURL *url in e) {
        NSInteger depth = url.pathComponents.count - rootDepth;
        if (depth > maxDepth) { [e skipDescendants]; continue; }

        NSNumber *isDirNum = nil;
        [url getResourceValue:&isDirNum forKey:NSURLIsDirectoryKey error:nil];
        if ([isDirNum boolValue] && [url.lastPathComponent isEqualToString:filename]) {
            return url.path;
        }
    }
    return nil;
}

// Returns the URL to the compiled model, or nil with diagnostic logging if
// it can't be found anywhere.
//
// NOTE ON LOGGING: unlike the old jailbreak-path version, these paths are
// NOT run through NSLog's %@ private-data redaction concerns in the same
// way -- app-container paths are still subject to OS log privacy redaction
// (you'll see <private> in Console.app/log unless you view with --info
// or --private), but the *last path component* of each candidate is also
// logged separately below so you can tell candidates apart even when the
// full path is redacted in a captured log.
static NSURL *AIPlayerModelURL(void) {
    NSString *dylibDir  = AIPlayerDylibDirectory();
    NSString *bundleDir = [NSBundle mainBundle].bundlePath;
    os_log(AIPlayerLog(), "Derived dylib directory = %@", dylibDir ?: @"(nil)");
    os_log(AIPlayerLog(), "Main bundle directory = %@", bundleDir ?: @"(nil)");

    static NSString *const kModelName = @"SwipeAnnotator.mlmodelc";

    // Candidate layouts, most to least likely for a Sideloadly
    // dylib+bundle injection (which typically drops both into
    // "<App>.app/Frameworks/" side by side):
    NSMutableArray<NSString *> *candidates = [NSMutableArray array];
    if (dylibDir) {
        // 1. Model bundle sitting directly next to the dylib.
        [candidates addObject:[dylibDir stringByAppendingPathComponent:kModelName]];
        // 2. Model bundle nested inside an AIPlayer.bundle next to the dylib
        //    (in case Sideloadly preserves the bundle's own directory name
        //    as a wrapper rather than flattening its contents).
        [candidates addObject:[[dylibDir stringByAppendingPathComponent:@"AIPlayer.bundle"]
                                stringByAppendingPathComponent:kModelName]];
    }
    if (bundleDir) {
        // 3. Directly inside the .app root.
        [candidates addObject:[bundleDir stringByAppendingPathComponent:kModelName]];
        // 4. Inside the .app's own Frameworks/ dir (covers the case where
        //    dyld reports a resolved/symlinked path for (1)/(2) that
        //    doesn't textually match, e.g. via a private/var symlink).
        [candidates addObject:[[bundleDir stringByAppendingPathComponent:@"Frameworks"]
                                stringByAppendingPathComponent:kModelName]];
        [candidates addObject:[[[bundleDir stringByAppendingPathComponent:@"Frameworks"]
                                 stringByAppendingPathComponent:@"AIPlayer.bundle"]
                                stringByAppendingPathComponent:kModelName]];
    }

    for (NSString *candidate in candidates) {
        os_log(AIPlayerLog(), "Trying candidate (leaf=%{public}@): %@", candidate.lastPathComponent, candidate);
        if ([[NSFileManager defaultManager] fileExistsAtPath:candidate]) {
            os_log(AIPlayerLog(), "Found SwipeAnnotator.mlmodelc at: %@", candidate);
            return [NSURL fileURLWithPath:candidate isDirectory:YES];
        }
    }

    // Last resort: bounded scan of the .app bundle itself -- under
    // Sideloadly injection this is the only place an injected bundle could
    // plausibly be, so we search here instead of any system/jailbreak path.
    if (bundleDir) {
        os_log(AIPlayerLog(), "Scanning under app bundle: %@", bundleDir);
        NSString *found = AIPlayerFindDirectoryNamed(kModelName, bundleDir, /*maxDepth=*/4);
        if (found) {
            os_log(AIPlayerLog(), "Found SwipeAnnotator.mlmodelc via fallback scan at: %@", found);
            return [NSURL fileURLWithPath:found isDirectory:YES];
        }
        os_log(AIPlayerLog(), "Scan under app bundle found nothing");
    }

    os_log(AIPlayerLog(), "Exhausted all lookups -- model not found. "
          "This build expects a non-jailbroken, Sideloadly-injected "
          "deployment: SwipeAnnotator.mlmodelc must be injected as a "
          "bundle/framework alongside AIPlayer.dylib (Sideloadly's "
          "'inject dylib/framework/bundle' option), NOT packaged as a "
          ".deb -- there is no jailbreak root on this device for a .deb "
          "install to land in. If this persists, use Sideloadly's log "
          "viewer or `log stream --private` on-device to see exactly "
          "where AIPlayer.dylib and its bundle were actually placed "
          "inside the .app, and compare that against the candidates "
          "this function tries.");
    return nil;
}

@interface AIOverlayVC : UIViewController
@property (nonatomic, strong) UIButton *toggleButton;
@property (nonatomic, assign) BOOL isPlaying;
@property (nonatomic, assign) NSInteger frameTick;
@property (nonatomic, assign) CFTimeInterval lastInjectTime;
// TEMP DIAGNOSTIC — total frames the ReplayKit handler has seen vs. how many
// made it past the FPS throttle into actual processing this session.
@property (nonatomic, assign) NSInteger diagFramesReceived;
@property (nonatomic, assign) NSInteger diagFramesProcessed;
@end

@implementation AIOverlayVC

- (void)loadView {
    AIPassthroughView *v = [[AIPassthroughView alloc] initWithFrame:UIScreen.mainScreen.bounds];
    v.backgroundColor = [UIColor clearColor];
    self.view = v;
}

- (void)viewDidLoad {
    [super viewDidLoad];

    self.toggleButton = [UIButton buttonWithType:UIButtonTypeCustom];
    self.toggleButton.frame = CGRectMake(0, 0, 130, 44);
    self.toggleButton.center = CGPointMake(self.view.bounds.size.width - 80, 100);
    self.toggleButton.layer.cornerRadius = 12;
    self.toggleButton.clipsToBounds = YES;
    self.toggleButton.backgroundColor = [UIColor colorWithWhite:0.15 alpha:0.92];
    self.toggleButton.titleLabel.font = [UIFont monospacedSystemFontOfSize:12 weight:UIFontWeightBold];
    [self.toggleButton setTitle:@"▶ AI: OFF" forState:UIControlStateNormal];
    [self.toggleButton setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    [self.toggleButton addTarget:self action:@selector(toggleTapped)
                forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:self.toggleButton];

    UIPanGestureRecognizer *drag = [[UIPanGestureRecognizer alloc]
        initWithTarget:self action:@selector(handleDrag:)];
    [self.toggleButton addGestureRecognizer:drag];

    // Load the compiled model, staged by Theos alongside AIPlayer.dylib
    // itself (see the MARK: Resource bundle lookup comment above for why).
    NSURL *modelURL = AIPlayerModelURL();
    if (modelURL) {
        NSError *err = nil;
        if (![[InferenceEngine sharedEngine] loadModelAtURL:modelURL error:&err]) {
            os_log(AIPlayerLog(), "Failed to load model: %{public}@", err);
        } else {
            os_log(AIPlayerLog(), "Model loaded from %@", modelURL);
        }
    } else {
        os_log(AIPlayerLog(), "Model not loaded -- SwipeAnnotator.mlmodelc was not found (see preceding log lines for paths checked)");
    }
}

- (void)handleDrag:(UIPanGestureRecognizer *)pan {
    CGPoint d = [pan translationInView:self.view];
    pan.view.center = CGPointMake(pan.view.center.x + d.x, pan.view.center.y + d.y);
    [pan setTranslation:CGPointZero inView:self.view];
}

- (void)toggleTapped {
    // TEMP DIAGNOSTIC — catches double-fire (e.g. the drag gesture recognizer
    // on the button interacting with UIControlEventTouchUpInside and causing
    // two toggles per tap, which would flip isPlaying back off silently).
    os_log(AIPlayerLog(), "[DIAG] toggleTapped called. isPlaying before flip = %d", self.isPlaying);

    self.isPlaying = !self.isPlaying;
    if (self.isPlaying) {
        [self startCapture];
        [self.toggleButton setTitle:@"■ AI: ON" forState:UIControlStateNormal];
        self.toggleButton.backgroundColor = [UIColor colorWithRed:0.20 green:0.70 blue:0.35 alpha:0.95];
    } else {
        [self stopCapture];
        [self.toggleButton setTitle:@"▶ AI: OFF" forState:UIControlStateNormal];
        self.toggleButton.backgroundColor = [UIColor colorWithWhite:0.15 alpha:0.92];
    }
}

- (void)startCapture {
    [[InferenceEngine sharedEngine] resetSession];
    self.frameTick = 0;

    // TEMP DIAGNOSTIC — reset per-session frame counters (see processCapturedBuffer:).
    self.diagFramesReceived = 0;
    self.diagFramesProcessed = 0;

    // TEMP DIAGNOSTIC — snapshot RPScreenRecorder's own state right before
    // starting. If isRecording is already YES here from a prior session that
    // never cleanly stopped, startCaptureWithHandler: can behave oddly.
    RPScreenRecorder *rec = [RPScreenRecorder sharedRecorder];
    os_log(AIPlayerLog(), "[DIAG] pre-start recorder state: isRecording=%d isMicrophoneEnabled=%d isCameraEnabled=%d isAvailable=%d",
          rec.isRecording, rec.isMicrophoneEnabled, rec.isCameraEnabled, rec.isAvailable);

    __weak typeof(self) weakSelf = self;
    [[RPScreenRecorder sharedRecorder]
        startCaptureWithHandler:^(CMSampleBufferRef buf, RPSampleBufferType type, NSError *err) {
            // TEMP DIAGNOSTIC — logs once per second regardless of the guards
            // below, so we can see raw frame arrival + which guard (if any)
            // is dropping every frame. Remove once frames are confirmed
            // reaching processCapturedBuffer:.
            __strong typeof(weakSelf) diagSelf = weakSelf;
            if (diagSelf) diagSelf.diagFramesReceived++;

            static CFTimeInterval lastDiagLog = 0;
            CFTimeInterval diagNow = CACurrentMediaTime();
            if (diagNow - lastDiagLog > 1.0) {
                lastDiagLog = diagNow;
                os_log(AIPlayerLog(), "[DIAG] handler fired: type=%ld err=%{public}@ selfAlive=%d isPlaying=%d "
                      "totalReceived=%ld totalProcessed=%ld",
                      (long)type, err, diagSelf != nil, diagSelf.isPlaying,
                      (long)diagSelf.diagFramesReceived, (long)diagSelf.diagFramesProcessed);
            }

            if (type != RPSampleBufferTypeVideo || err) return;
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf || !strongSelf.isPlaying) return;
            [strongSelf processCapturedBuffer:buf];
        }
        completionHandler:^(NSError *err) {
            // TEMP DIAGNOSTIC — fully expanded error info instead of just %@,
            // since NSError's default description can hide the domain/code
            // that actually tells us what failed.
            if (err) {
                os_log(AIPlayerLog(), "startCapture error: domain=%{public}@ code=%ld description=%{public}@ userInfo=%{public}@",
                      err.domain, (long)err.code, err.localizedDescription, err.userInfo);
            } else {
                os_log(AIPlayerLog(), "Capture started.");
            }
        }];
}

- (void)stopCapture {
    // TEMP DIAGNOSTIC — expanded error info, same rationale as startCapture's completion handler.
    [[RPScreenRecorder sharedRecorder] stopCaptureWithHandler:^(NSError *err) {
        if (err) {
            os_log(AIPlayerLog(), "Capture stopped (error: domain=%{public}@ code=%ld description=%{public}@)",
                  err.domain, (long)err.code, err.localizedDescription);
        } else {
            os_log(AIPlayerLog(), "Capture stopped.");
        }
    }];
}

// Runs on RPScreenRecorder's capture queue, not main thread.
- (void)processCapturedBuffer:(CMSampleBufferRef)buf {
    static CFTimeInterval lastTime = 0;
    CFTimeInterval now = CACurrentMediaTime();
    if (now - lastTime < (1.0 / kTargetFPS)) return;
    lastTime = now;

    self.diagFramesProcessed++;

    CVImageBufferRef px = CMSampleBufferGetImageBuffer(buf);
    if (!px) {
        os_log(AIPlayerLog(), "[DIAG] CMSampleBufferGetImageBuffer returned NULL on frame %ld", (long)self.diagFramesProcessed);
        return;
    }

    // TEMP DIAGNOSTIC — dump pixel buffer format/dimensions once, on the
    // first frame that reaches this point. Confirms the capture buffer
    // actually looks like a normal screen frame (right size, known pixel
    // format) rather than something degenerate.
    static BOOL diagLoggedFormat = NO;
    if (!diagLoggedFormat) {
        diagLoggedFormat = YES;
        size_t w = CVPixelBufferGetWidth(px);
        size_t h = CVPixelBufferGetHeight(px);
        OSType fmt = CVPixelBufferGetPixelFormatType(px);
        char fmtChars[5] = {
            (char)((fmt >> 24) & 0xFF), (char)((fmt >> 16) & 0xFF),
            (char)((fmt >> 8) & 0xFF),  (char)(fmt & 0xFF), 0
        };
        os_log(AIPlayerLog(), "[DIAG] first pixel buffer: %zux%zu format='%{public}s' (0x%08X)",
              w, h, fmtChars, fmt);
    }

    self.frameTick++;
    BOOL hasNewSlow = (self.frameTick % kSlowBranchEveryNTicks) == 0;

    NSError *err = nil;
    SwipePrediction *pred = [[InferenceEngine sharedEngine] predictWithPixelBuffer:px
                                                                   hasNewSlowFrame:hasNewSlow
                                                                              error:&err];
    if (!pred) {
        // TEMP DIAGNOSTIC — fully expanded error, same rationale as above.
        if (err) {
            os_log(AIPlayerLog(), "predict error: domain=%{public}@ code=%ld description=%{public}@ userInfo=%{public}@",
                  err.domain, (long)err.code, err.localizedDescription, err.userInfo);
        } else {
            os_log(AIPlayerLog(), "[DIAG] predictWithPixelBuffer returned nil with no error set (frame %ld)",
                  (long)self.diagFramesProcessed);
        }
        return;
    }

    // TEMP DIAGNOSTIC — log the raw model output once per second regardless
    // of threshold, so we can see whether the model is producing sane,
    // near-threshold, or completely flat/degenerate values even when it
    // never actually crosses kDetectionThreshold.
    static CFTimeInterval lastPredDiagLog = 0;
    CFTimeInterval predDiagNow = CACurrentMediaTime();
    if (predDiagNow - lastPredDiagLog > 1.0) {
        lastPredDiagLog = predDiagNow;
        os_log(AIPlayerLog(), "[DIAG] pred: det=%.3f dir=%ld dirConf=%.3f threshold=%.2f frame=%ld",
              pred.detProbability, (long)pred.direction, pred.dirConfidence,
              kDetectionThreshold, (long)self.diagFramesProcessed);
    }

    if (pred.detProbability >= kDetectionThreshold) {
        CFTimeInterval nowTime = CACurrentMediaTime();
        if (nowTime - self.lastInjectTime < kInjectCooldown) {
            // Still inside the previous swipe's cooldown window — almost
            // certainly the same physical swipe crossing threshold on a
            // second consecutive tick. Log it for visibility but don't
            // double-fire.
            os_log(AIPlayerLog(), "swipe dir=%ld conf=%.2f det=%.2f — suppressed (cooldown)",
                  (long)pred.direction, pred.dirConfidence, pred.detProbability);
            return;
        }
        self.lastInjectTime = nowTime;

        os_log(AIPlayerLog(), "swipe dir=%ld conf=%.2f det=%.2f — injecting",
              (long)pred.direction, pred.dirConfidence, pred.detProbability);
        [self injectSwipeForDirection:pred.direction confidence:pred.dirConfidence detection:pred.detProbability];
    }
}

- (void)injectSwipeForDirection:(SwipeDirection)dir confidence:(float)conf detection:(float)det {
    CGSize screen = [UIScreen mainScreen].bounds.size;
    CGPoint center = CGPointMake(screen.width / 2.0, screen.height / 2.0);
    CGFloat mag = MIN(screen.width, screen.height) * kSwipeMagnitude;

    CGPoint end = center;
    switch (dir) {
        case SwipeDirUp:    end = CGPointMake(center.x, center.y - mag); break;
        case SwipeDirDown:  end = CGPointMake(center.x, center.y + mag); break;
        case SwipeDirLeft:  end = CGPointMake(center.x - mag, center.y); break;
        case SwipeDirRight: end = CGPointMake(center.x + mag, center.y); break;
    }

    // center/end are computed from the screen bounds; TouchSynthesisKeyWindow() (used
    // throughout HybridTouchSynthesizer) resolves to the same window whose
    // bounds these points were computed from, so no conversion is needed.
    //
    // Delivered through FloatingSwipeButtonManager's own, unedited
    // -executeSingleShotSwipeFrom:to:duration: (see the ported engine MARK
    // block above) -- the exact same method -triggerSwipe calls for its own
    // random-direction demo swipe, just given the model's predicted
    // start/end points instead of a random direction. No dispatch logic is
    // reimplemented here.
    [[FloatingSwipeButtonManager sharedInstance] executeSingleShotSwipeFrom:center to:end duration:kSwipeDuration];

    __weak typeof(self) weakSelf = self;
    dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf flashInjectionFeedback]; });
}

// Briefly flashes the toggle button gold so you can visually confirm firing
// rate/timing against on-screen gameplay without reading the console.
- (void)flashInjectionFeedback {
    if (!self.isPlaying) return;
    UIColor *playingColor = [UIColor colorWithRed:0.20 green:0.70 blue:0.35 alpha:0.95];
    self.toggleButton.backgroundColor = [UIColor colorWithRed:0.95 green:0.75 blue:0.15 alpha:0.95];
    __weak typeof(self) weakSelf = self;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.15 * NSEC_PER_SEC)),
                   dispatch_get_main_queue(), ^{
        __strong typeof(weakSelf) strongSelf = weakSelf;
        if (strongSelf && strongSelf.isPlaying)
            strongSelf.toggleButton.backgroundColor = playingColor;
    });
}

@end

// =============================================================================
// MARK: - Runtime introspection dump (diagnostic-only)
//
// InProcessTouchInjector's +isSupported logged that -[UIEvent _clearTouches]
// and -[UIEvent _addTouch:forDelayedDelivery:] are gone on this iOS build.
// Rather than guess at replacement names, dump every instance method UIEvent,
// UITouch, and UIApplication actually respond to on THIS device/iOS version
// right now, so the real current selector names can be read straight out of
// the Console log instead of inferred from a KIF version that may predate
// this iOS release.
//
// Runs once, ~200ms after launch (after the swizzle/class-loading dust
// settles), and only logs -- doesn't touch any injection behavior. Safe to
// leave in during this investigation phase; strip it once the replacement
// selectors are found and wired in.
// =============================================================================

static void AIPlayerDumpMethods(Class cls, NSString *label) {
    unsigned int count = 0;
    Method *methods = class_copyMethodList(cls, &count);
    if (!methods) {
        os_log(AIPlayerLog(), "[DIAG] %{public}@: class_copyMethodList returned NULL", label);
        return;
    }
    NSMutableArray<NSString *> *names = [NSMutableArray arrayWithCapacity:count];
    for (unsigned int i = 0; i < count; i++) {
        [names addObject:NSStringFromSelector(method_getName(methods[i]))];
    }
    free(methods);
    [names sortUsingSelector:@selector(compare:)];
    os_log(AIPlayerLog(), "[DIAG] %{public}@ (%u methods):", label, count);
    for (NSString *n in names) {
        os_log(AIPlayerLog(), "[DIAG]   %{public}@ %{public}@", label, n);
    }
}

static void AIPlayerDumpTouchAPISurface(void) {
    // Instance methods only (class_copyMethodList on the class object itself
    // would give class/+ methods -- +load, +new, etc. -- which aren't what
    // we're after here; the private touch-delivery API is all -instance).
    AIPlayerDumpMethods(object_getClass([UIEvent class]) ? [UIEvent class] : Nil, @"UIEvent");
    AIPlayerDumpMethods([UITouch class], @"UITouch");
    AIPlayerDumpMethods([UIApplication class], @"UIApplication");

    // Also walk UIEvent's superclass chain -- if touch-set mutation moved to
    // a new internal base class (e.g. some private _UIInternalEvent) rather
    // than staying on UIEvent itself, a plain class dump above would miss it
    // entirely, so log the chain to know whether that's worth checking too.
    Class c = [UIEvent class];
    NSMutableArray<NSString *> *chain = [NSMutableArray array];
    while (c) {
        [chain addObject:NSStringFromClass(c)];
        c = class_getSuperclass(c);
    }
    os_log(AIPlayerLog(), "[DIAG] UIEvent superclass chain: %{public}@",
          [chain componentsJoinedByString:@" -> "]);
}



__attribute__((constructor))
static void AIPlayerInit(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        static UIWindow *win = nil;
        // AIOverlayWindow (not stock UIWindow): overrides hitTest: at the
        // window level and refuses key-window status, so it can never
        // intercept touches meant for the game. See the MARK: Passthrough
        // window comment block above for why the previous view-level-only
        // override wasn't enough.
        win = [[AIOverlayWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
        win.windowLevel = UIWindowLevelAlert + 1;
        win.backgroundColor = [UIColor clearColor];
        win.rootViewController = [AIOverlayVC new];
        win.hidden = NO;
    });

    // TEMP DIAGNOSTIC — one-shot dump of the real UIEvent/UITouch/
    // UIApplication method surface on this device's actual iOS build.
    // 200ms delay: give UIKit's own class loading/swizzling time to settle
    // before enumerating, so the dump reflects final runtime state.
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.2 * NSEC_PER_SEC)),
                   dispatch_get_main_queue(), ^{
        AIPlayerDumpTouchAPISurface();
    });
}
