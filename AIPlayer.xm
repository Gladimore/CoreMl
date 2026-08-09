// =============================================================================
// AIPlayer.xm — standalone AI-only tweak.
//
// Shows a small floating "AI: OFF / AI: ON" button. Tapping it starts/stops
// a screen-capture + Core ML inference loop. Every high-confidence detection
// is logged AND injected into the game as a synthesized swipe via
// HybridTouchSynthesizer (see the "Touch synthesis engine" MARK block below)
// — the button flashes gold for ~150ms each time a swipe is actually sent,
// so you can visually confirm firing rate.
//
// HybridTouchSynthesizer is ported verbatim from RandomSwipeTweak.xm: it
// builds a synthetic UITouch + GSEventProxy-backed UIEvent by hand (Layer B,
// via the private _initWithEvent:touches: initializer) and sends it, then
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
// UnityDiagnostics [Diag] hook confirming touches actually reach UnityView.
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
// MARK: - Touch synthesis engine — ported VERBATIM from RandomSwipeTweak.xm
//
// Replaces the old TouchInjector (system-HID, requires jailbreak, never
// worked here) and InProcessTouchInjector (KIF-style sendEvent:, confirmed
// via on-device Console.app logging to leave event.allTouches.count==0 on
// this device/iOS build -- see conversation history). Every method below is
// copied unedited from RandomSwipeTweak.xm, including its own changelog
// comment (kept intact since it documents exactly how confident each
// reconstructed private-API behavior is). The only changes anywhere in this
// block are: (1) this MARK header itself, and (2) the four #import lines at
// the top of the original file were dropped since AIPlayer.xm's own header
// already imports UIKit/UIKit.h, objc/runtime.h, os/log.h, and math.h.
//
// -injectSwipeForDirection:confidence:detection: (further down this file)
// now drives a swipe by calling FloatingSwipeButtonManager's own, unedited
// -executeSingleShotSwipeFrom:to:duration: with the model's predicted
// start/end points instead of -triggerSwipe's random direction -- it does
// not reimplement any dispatch logic itself.
//
// NOTE: this ports RandomSwipeTweak.xm's OWN overlay button too
// (FloatingSwipeButtonManager -showOverlayButton, wired to fire on
// -[UIWindow makeKeyAndVisible] below) -- a second, separate red "Swipe"
// button will appear on screen alongside AIPlayer's existing AI ON/OFF
// button, and tapping it fires an independent random-direction demo swipe
// through this same engine. Left in because it's a real method in the
// source file and the instruction was every method, no edits -- remove the
// UIWindow(GGMakeKeyAndVisibleHook) category below if you don't want that
// second button.
// =============================================================================

// ===========================================================
// CHANGELOG — see Tweak_xm_factcheck.md for the first-pass evidence.
//
// SEVENTH PASS (this revision, Tweak_5.xm) — EMPIRICAL, not ground-truth-
// derived. On-device Console.app logging (see the UnityDiagnostics [Diag]
// hook further down) proved sendEvent:-based delivery (Layer B) is not
// reaching -[UnityView touches...:withEvent:] on this device/iOS build at
// all, even though event construction always succeeds (syntheticEvent
// built OK every time, no nil, no exception — sendEvent: just silently
// drops it). Real finger touches fire [Diag] lines immediately; the exact
// same synthetic phase sequence, sent via sendSyntheticEvent:toWindow:,
// produced none. Fix #18: -performEventCreationFailureFallbackForTouch:
// (Layer A — direct touchesBegan/Moved/Ended:withEvent: calls on
// touch.view) is now called UNCONDITIONALLY in
// -dispatchHybridTouchAtPoint:phase:inOutTouch:, not just when event
// creation returns nil as originally gated. See the inline comment at the
// call site for the known double-delivery risk on other devices/builds.
//
// SIXTH PASS (Tweak_4.xm): cross-checked directly against the
// actual ground-truth SOURCE (iGameGod.c's decompiled
// __UIEvent_Synthesize__initWithTouch__), not just prior disassembly/notes.
// Two real findings:
// 16. Fix #13 (flags mapping) was WRONG as of the fifth pass. Ground truth,
//     read directly from __UIEvent_Synthesize__initWithTouch__:
//         uVar9 = 0x1010180;                  // default
//         if (phase != Ended(3)) {
//             if (phase != Cancelled(4)) {
//                 uVar9 = 0x3010180;           // override
//             }
//         }
//     i.e. Began(0)/Moved(1)/Stationary(2) get overridden to 0x3010180, and
//     Ended(3)/Cancelled(4) BOTH keep the default 0x1010180. The fifth pass
//     had this backwards (default-for-everything-except-Cancelled), which
//     silently re-broke a bug this project had already found and fixed once
//     empirically on the first working attempt (see project notes: "Began/
//     Moved/Stationary should get flags 0x3010180, Ended/Cancelled should
//     get 0x1010180"). Reverted to match the actual binary.
// 17. buildSyntheticEventForTouch: never set the constructed EVENT object's
//     own _timestamp ivar — only the touch's. Ground truth explicitly reads
//     back into the just-created event's _timestamp field immediately after
//     _initWithEvent:touches: returns non-nil, before returning it to the
//     caller. Added below.
//
// THIRD VERIFICATION PASS (earlier revision) independently re-derived every
// claim below directly from iGameGod.c rather than trusting the first-pass
// document, and found the document's own headline claims on #1/#2/#8 were
// wrong or overconfident. Corrected here; see inline comments at each call
// site for the exact line-level evidence.
//
// FIFTH PASS (Tweak_3.xm): raw disassembly of
// -[UIEvent(Synthesize) _initWithTouch:] and FUN_00584b84/FUN_0058eb90/
// FUN_00587624/FUN_0057efe4 confirmed everything in items #1/#8 above was
// actually correct as stated (the earlier "raw disassembly" citations were
// unverifiable at the time they were written, but turned out accurate).
// NOTE: this pass's own item #13 (flags) turned out to be wrong — see
// SIXTH PASS item #16 above, which supersedes it.
// Also confirmed and left unchanged: FUN_00587624's action-code table
// (actionCodeForTouch:, further down) matches the binary's
// FUN_0057efe4 exactly, bit for bit, across all six input booleans and
// all five return values — no changes needed there.
//
// BUILD FIXES (Xcode 26.6 / iOS 26.5 SDK, -Werror): these are toolchain/
// ARC issues, not ground-truth discrepancies — nothing here changes
// runtime behavior versus the binary.
// 14. [UIApplication sharedApplication].keyWindow is deprecated (iOS 13+,
//     ignores multi-scene apps). Replaced all three call sites with
//     GG_KeyWindow(), which walks connectedScenes for the foreground-
//     active UIWindowScene's key window and only falls back to the
//     deprecated accessor if that comes up empty.
// 15. Passing &_currentPersistentTouch (an ivar) directly to a
//     UITouch ** / __autoreleasing out-param doesn't compile under ARC
//     ("passing address of non-local object to __autoreleasing parameter
//     for write-back"). Fixed at all five call sites by routing through a
//     local temp variable and assigning back to the ivar afterward.
// ===========================================================
// 1. RESOLVED (fourth pass, via raw disassembly of FUN_00584b84 and
//    FUN_0058eb90 directly): Layer A via FUN_0058eb90 is not confined to
//    session-boundary cleanup (the first-pass doc's "only caller is
//    FUN_0058442c" was false), and the trigger is no longer unknown either
//    (the third pass's "opaque bit of param_4, unrecoverable" was also
//    wrong — that was a decompiler-C-level limitation, not a real
//    ambiguity). Disassembly shows `cbz x0, LAB_00584e64` sitting
//    immediately after _objc_retainAutoreleasedReturnValue, immediately
//    after the call to _TouchSynthesisCreateEvent — it's a direct null
//    check on the created event, nothing else. FUN_0058eb90 fires exactly
//    when event creation fails, and its own body confirms an exact
//    phase->selector map (0/1/3/4 -> touchesBegan/Moved/Ended/Cancelled,
//    2 -> no-op), passing the possibly-nil event straight through. Now
//    wired into the normal per-event path (see
//    -performEventCreationFailureFallbackForTouch:targetView:event:phase:
//    and its call site in -dispatchHybridTouchAtPoint:phase:inOutTouch:)
//    instead of left as a dangling manual-only method.
//    -finalizeAnyActiveTouch: (FUN_0058442c) remains correct as the
//    separate, confirmed session-cleanup path — the two are genuinely
//    distinct call sites in the binary, not duplicates.
// 2. CORRECTED similarly: Layer C is not scrollview-only. The first-pass
//    doc's claim that _TouchSynthesisPerformGestureRecognizerFallback
//    "has no caller at all" is false — it's called from FUN_0058fb74,
//    itself called from FUN_00587624, which IS called directly from the
//    normal per-tick replay function FUN_00584b84 (on a specific recorded
//    phase-byte value, not just from the scrollview compensator
//    FUN_0058dda0). The exact gating condition and the internal
//    phase-byte's mapping to Began/Moved/Ended/Cancelled could not be
//    confirmed from this decompilation (may not match the real
//    UITouchPhase enum used elsewhere in the binary).
//    -performGestureRecognizerFallbackOnView: remains an explicit,
//    manually-callable method (fixed internally, see #4/#5) rather than
//    auto-firing, since the real trigger condition is unconfirmed either
//    way.
// 3. sendEvent: now goes to the touch's window directly, falling back to
//    [UIApplication sharedApplication] only if there's no window — matches
//    _TouchSynthesisDispatchEvent's own nil check. Previously always went
//    through UIApplication.
// 4. Gesture-recognizer state is now set ONCE, all actions for a recognizer
//    fire, then state is restored ONCE — not set/restore per pair.
// 5. _targets is now read via KVC (valueForKey:), matching
//    _TouchSynthesisGestureTargetEntries, not raw Ivar/object_getIvar.
// 6. The window/view ivar-setter chain's final KVC fallback now uses the
//    PUBLIC key ("window"/"view"), not "_window"/"_view".
// 7. Added the associated-object shadow-storage + swizzled -window/-view
//    fallback that the binary has and this file previously lacked entirely.
// 8. GSEventProxy offsets: RESOLVED (fourth pass) via raw disassembly of
//    -[UIEvent(Synthesize) _initWithTouch:], not just the decompiled C.
//    flags(0x08)/type(0x0C), sizeX/sizeY(0x68, both 1.0), and x3/y3(0x70/0x74,
//    duplicating x1/y1) all confirmed as before. x1/y1(0x14/0x18) trace to a
//    genuine [touch locationInView:window] call (d8/d9 -> fcvt -> stp), and
//    x2/y2(0x1c/0x20) trace to a genuine [touch previousLocationInView:window]
//    call (d10/d11 -> fcvt -> stp) — not a duplicate of the current point,
//    and not a "param_2 bit-cast" as the third pass concluded from the
//    Ghidra C-level view alone. That reading was a decompiler
//    misattribution of which SSA value fed the store, not a real ambiguity
//    in the binary. x2/y2 = previousLocationInView:, restored below.
//    Directly re-confirmed sixth pass: __UIEvent_Synthesize__initWithTouch__
//    calls -locationInView: and -previousLocationInView: on the touch
//    itself for x1/y1 and x2/y2, respectively — the low-level struct is
//    always built FROM the touch's own already-populated state, never from
//    anything else.
// 9. The gesture-fallback per-entry respondsToSelector: guard is kept but
//    now explicitly labeled as an unverified addition, not confirmed
//    binary behavior (ground truth only checks target != nil && the
//    action string is non-empty).
// 12. NEW, third pass: the class-name safety check
//    (_TouchSynthesisClassNameLooksUnsafeForGestureFallback) rejects more
//    than WK/_WK prefixes — it also rejects any class name with prefix
//    "Web" or containing the substring "WebKit". Both
//    -isSafeGestureFallbackTarget: and -isSafeGestureFallbackTargetEntry:
//    only checked WK/_WK; fixed below to match all four conditions.
//
// SECOND VERIFICATION PASS — traced the actual touch-creation entry point
// (initInView:at:withCount:) rather than just the setter helpers this file
// already mirrored. This independently re-confirmed fixes #6/#7 (found
// via a second, separate function that does the same associated-object
// store + ivar/private-setter/public-setter/KVC chain) and turned up two
// real gaps that existed even before this revision:
// 10. _tapCount was never initialized on a new touch. Confirmed the binary
//     always writes the touch count (1, for a single-point touch) into it.
// 11. _touchFlags was never touched at all. Confirmed the binary sets bits
//     0x1/0x2 on every new touch, and clears bit 0x2 whenever a location
//     update moves the touch more than 2.0pt in x or y.
// Two things noted but NOT changed, flagged with lower confidence/out of
// scope: (a) the binary's init function calls [super init] in a way that
// implies it may be defined on an actual UITouch subclass rather than a
// category on UITouch itself — this file still uses a plain UITouch
// instance; (b) the binary's location setter accepts a point in an
// arbitrary view's coordinate space and converts it to window coordinates
// via convertPoint:fromView: — this file still assumes the caller always
// passes window coordinates directly, which is fine for this file's own
// demo usage but is narrower than the real API surface.
// ===========================================================


// ---------------------------------------------------------
// LOGGING: switched from NSLog to real unified-logging (os_log) calls.
// As of iOS 26, NSLog redacts dynamic-string format arguments to
// `<private>` essentially unconditionally — appending %{public}@ to an
// NSLog call no longer reliably un-redacts it the way it used to on
// earlier iOS versions. Calling os_log() directly (with an explicit
// os_log_t handle) is the supported way to get %{public}@ honored again.
// Initialized via a constructor so gg_log is valid before any +load
// method in this file (including UITouch's own +load below) can fire.
// View this with Console.app streaming from the device — the `log`
// CLI tool and idevicesyslog-style tools have been unreliable for
// reading back %{public} os_log output on iOS 26 at the time of writing.
// ---------------------------------------------------------
static os_log_t gg_log;

__attribute__((constructor))
static void GG_InitLog(void) {
    gg_log = os_log_create("com.gg.touchsynthesis", "touch");
}

// ---------------------------------------------------------
// -[UIApplication keyWindow] is deprecated (iOS 13+): it doesn't account
// for multi-scene apps, returning a key window across all connected
// scenes rather than the actually-active one. This walks connectedScenes
// for the foreground-active UIWindowScene's key window instead, falling
// back to the deprecated accessor only pre-iOS 13 or if no active scene
// can be found (e.g. very early app launch).
// ---------------------------------------------------------
static UIWindow *GG_KeyWindow(void) {
    if (@available(iOS 13.0, *)) {
        for (UIScene *scene in [UIApplication sharedApplication].connectedScenes) {
            if (scene.activationState != UISceneActivationStateForegroundActive) continue;
            if (![scene isKindOfClass:[UIWindowScene class]]) continue;
            UIWindowScene *windowScene = (UIWindowScene *)scene;
            for (UIWindow *window in windowScene.windows) {
                if (window.isKeyWindow) return window;
            }
        }
    }
    // Fallback: pre-iOS 13, or no active scene found yet (e.g. very early
    // app launch before a scene has been marked foreground-active).
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return [UIApplication sharedApplication].keyWindow;
    #pragma clang diagnostic pop
}

// ---------------------------------------------------------
// PRIVATE iOS INTERFACES REQUIRED FOR TOUCH SYNTHESIS
// ---------------------------------------------------------
@interface UITouch (Private)
- (void)setPhase:(UITouchPhase)phase;
- (void)setTimestamp:(NSTimeInterval)timestamp;
@end

// Memory layout mimicking Apple's internal GSEvent struct.
// Required to prevent _initWithEvent:touches: from crashing.
// FIELD OFFSETS confirmed against __UIEvent_Synthesize__initWithTouch__ by
// computing byte offsets from the decompiled class_t layout: flags(0x08),
// type(0x0C), x1/y1/x2/y2(0x14-0x23), sizeX/sizeY(0x68/0x6C), x3/y3(0x70/0x74)
// all land where this struct implies.
@interface GSEventProxy : NSObject {
@public
    unsigned int flags;
    unsigned int type;
    unsigned int ignored1;
    float x1, y1, x2, y2;
    unsigned int ignored2[10];
    unsigned int ignored3[7];
    float sizeX, sizeY;
    float x3, y3;
    unsigned int ignored4[3];
}
@end
@implementation GSEventProxy
@end


// ---------------------------------------------------------
// FIX #7: Associated-object shadow storage + swizzled -window/-view.
// CONFIRMED: the binary unconditionally calls objc_setAssociatedObject
// with the intended window/view BEFORE attempting any ivar write
// (_TouchSynthesisStoreFallbackTarget), and swizzles UITouch's -window/
// -view at +load so that if the real accessor ever returns nil, it falls
// back to the associated object. This file previously had no equivalent,
// meaning a failed ivar chain on some iOS version would silently break
// hit-testing/dispatch with no safety net.
// ---------------------------------------------------------
static void *kTouchSynthesisFallbackWindowKey = &kTouchSynthesisFallbackWindowKey;
static void *kTouchSynthesisFallbackViewKey = &kTouchSynthesisFallbackViewKey;

@interface UITouch (TouchSynthesisFallback)
- (id)gg_touchSynthesis_window;
- (id)gg_touchSynthesis_view;
@end

@implementation UITouch (TouchSynthesisFallback)

+ (void)load {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        Method originalWindow = class_getInstanceMethod(self, @selector(window));
        Method swizzledWindow = class_getInstanceMethod(self, @selector(gg_touchSynthesis_window));
        if (originalWindow && swizzledWindow) {
            method_exchangeImplementations(originalWindow, swizzledWindow);
        }

        Method originalView = class_getInstanceMethod(self, @selector(view));
        Method swizzledView = class_getInstanceMethod(self, @selector(gg_touchSynthesis_view));
        if (originalView && swizzledView) {
            method_exchangeImplementations(originalView, swizzledView);
        }

        os_log(gg_log, "[Touch]: tweak loaded, -window/-view swizzle %{public}@ (window: %{public}@, view: %{public}@)",
              (originalWindow && swizzledWindow && originalView && swizzledView) ? @"installed" : @"FAILED — check selector names",
              originalWindow ? @"ok" : @"MISSING",
              originalView ? @"ok" : @"MISSING");
    });
}

// NOTE: after the +load swap, sending -gg_touchSynthesis_window to self
// actually runs the ORIGINAL -window implementation (classic swizzle
// idiom) — this is not infinite recursion.
- (id)gg_touchSynthesis_window {
    id result = [self gg_touchSynthesis_window];
    if (!result) {
        result = objc_getAssociatedObject(self, kTouchSynthesisFallbackWindowKey);
    }
    return result;
}

- (id)gg_touchSynthesis_view {
    id result = [self gg_touchSynthesis_view];
    if (!result) {
        result = objc_getAssociatedObject(self, kTouchSynthesisFallbackViewKey);
    }
    return result;
}

@end


// ---------------------------------------------------------
// SYNTHESIS ENGINE (Ground Truth Architecture)
// ---------------------------------------------------------
@interface HybridTouchSynthesizer : NSObject
@property (nonatomic, strong) UIView *excludedTouchView;
// Simplified single-instance analog of the binary's DAT_0128c0f0 global weak
// cache — "the control/cell we currently believe the active touch is over."
// FUN_00587624's FUN_0057efe4 gate compares this against expectations before
// allowing any UIControl/cell action to fire (see -performControlAndCellFallback...).
@property (nonatomic, weak) UIControl *currentControl;
@end

@implementation HybridTouchSynthesizer

+ (instancetype)sharedInstance {
    static HybridTouchSynthesizer *instance = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        instance = [[HybridTouchSynthesizer alloc] init];
    });
    return instance;
}

// 1. Raw memory write for SCALARS ONLY (coordinates, phase, timestamps)
- (void)writeScalarIvarOnObject:(id)object name:(const char *)name type:(const char *)type valuePtr:(void *)valuePtr {
    Ivar ivar = class_getInstanceVariable([object class], name);
    if (!ivar) return;

    ptrdiff_t offset = ivar_getOffset(ivar);
    void *ivarMemory = (uint8_t *)(__bridge void *)object + offset;

    if (strcmp(type, @encode(CGPoint)) == 0) {
        *(CGPoint *)ivarMemory = *(CGPoint *)valuePtr;
    } else if (strcmp(type, @encode(UITouchPhase)) == 0) {
        *(NSInteger *)ivarMemory = *(NSInteger *)valuePtr;
    } else if (strcmp(type, @encode(NSTimeInterval)) == 0) {
        *(NSTimeInterval *)ivarMemory = *(NSTimeInterval *)valuePtr;
    } else if (strcmp(type, @encode(NSInteger)) == 0) {
        // FIX #10: explicit NSInteger case, used for _tapCount below.
        // (Previously only reachable via the UITouchPhase branch, which
        // happens to be the same underlying width but was misleading to
        // reuse for a non-phase field.)
        *(NSInteger *)ivarMemory = *(NSInteger *)valuePtr;
    }
}

// Companion reader, needed for fix #8 (capturing the touch's location
// BEFORE overwriting it, to use as the true "previous" point).
- (CGPoint)readCGPointIvarOnObject:(id)object name:(const char *)name fallback:(CGPoint)fallback {
    Ivar ivar = class_getInstanceVariable([object class], name);
    if (!ivar) return fallback;
    ptrdiff_t offset = ivar_getOffset(ivar);
    void *ivarMemory = (uint8_t *)(__bridge void *)object + offset;
    return *(CGPoint *)ivarMemory;
}

// FIX #11: raw pointer helper for _touchFlags (a 16-bit bitfield that needs
// OR/AND bit manipulation, not a full-width overwrite like the fields
// above). Confirmed present in __UITouch_Synthesize__initInView_at_withCount__
// (sets bits 0x3 on every new touch) and __UITouch_Synthesize__setLocationInWindow__
// (clears bit 0x2 when movement exceeds 2.0pt in x or y) — this file
// previously never touched _touchFlags at all.
- (void *)rawIvarPointerOnObject:(id)object name:(const char *)name {
    Ivar ivar = class_getInstanceVariable([object class], name);
    if (!ivar) return NULL;
    ptrdiff_t offset = ivar_getOffset(ivar);
    return (uint8_t *)(__bridge void *)object + offset;
}

// 2. Defensive Fallback Chain for OBJECTS ONLY (_window, _view)
- (void)defensiveSetObject:(id)value forProperty:(NSString *)propName onObject:(id)target {
    // FIX #7: store the associated-object shadow copy first, unconditionally
    // — matches _TouchSynthesisStoreFallbackTarget's placement before any
    // ivar attempt below.
    if ([propName isEqualToString:@"window"]) {
        objc_setAssociatedObject(target, kTouchSynthesisFallbackWindowKey, value, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    } else if ([propName isEqualToString:@"view"]) {
        objc_setAssociatedObject(target, kTouchSynthesisFallbackViewKey, value, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }

    NSString *ivarName = [NSString stringWithFormat:@"_%@", propName];
    Ivar ivar = class_getInstanceVariable([target class], [ivarName UTF8String]);

    // Attempt A: Safe Runtime Ivar assignment
    if (ivar) {
        object_setIvar(target, ivar, value);
    } else {
        os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ — no ivar named %{public}@ found on %{public}@", propName, ivarName, [target class]);
    }

    // Verify Attempt A
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Warc-performSelector-leaks"
    id readBack = [target performSelector:NSSelectorFromString(propName)];
    if (readBack == value) {
        os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ = %{public}@ succeeded via raw ivar", propName, value);
        return;
    }

    // Attempt B: Private Setter (_setWindow: / _setView:)
    NSString *privateSelectorString = [NSString stringWithFormat:@"_set%@:", [propName capitalizedString]];
    SEL privateSelector = NSSelectorFromString(privateSelectorString);
    if ([target respondsToSelector:privateSelector]) {
        [target performSelector:privateSelector withObject:value];
        if ([target performSelector:NSSelectorFromString(propName)] == value) {
            os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ = %{public}@ succeeded via private setter", propName, value);
            return;
        }
    }

    // Attempt C: Public Setter (setWindow: / setView:)
    NSString *publicSelectorString = [NSString stringWithFormat:@"set%@:", [propName capitalizedString]];
    SEL publicSelector = NSSelectorFromString(publicSelectorString);
    if ([target respondsToSelector:publicSelector]) {
        [target performSelector:publicSelector withObject:value];
        if ([target performSelector:NSSelectorFromString(propName)] == value) {
            os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ = %{public}@ succeeded via public setter", propName, value);
            return;
        }
    }
    #pragma clang diagnostic pop

    // Attempt D: KVC Fallback of last resort.
    // FIX #6: use the PUBLIC key ("window"/"view"), matching
    // _TouchSynthesisTrySetValue(self, &cf_window/&cf_view, ...) —
    // previously this used "_window"/"_view", which isn't what the
    // binary's own last-resort call does.
    [target setValue:value forKey:propName];
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Warc-performSelector-leaks"
    os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ = %{public}@ — raw ivar/private/public setters all failed, used KVC fallback (readback now: %{public}@)",
          propName, value, [target performSelector:NSSelectorFromString(propName)]);
    #pragma clang diagnostic pop
}

// Shared event builder, factored out so both the normal per-event
// dispatch and -finalizeAnyActiveTouch: (fix #1) can use identical
// Layer-B event construction.
- (UIEvent *)buildSyntheticEventForTouch:(UITouch *)touch atPoint:(CGPoint)point previousPoint:(CGPoint)previousPoint phase:(UITouchPhase)phase {
    GSEventProxy *gsProxy = [[GSEventProxy alloc] init];
    gsProxy->x1 = point.x;         gsProxy->y1 = point.y;
    // FIX #8 (RESOLVED via raw disassembly of -[UIEvent(Synthesize) _initWithTouch:],
    // and directly re-confirmed sixth pass by reading __UIEvent_Synthesize__initWithTouch__
    // in iGameGod.c itself): offset 0x14/0x18 (x1/y1) trace directly to a real
    // [touch locationInView:window] call; offset 0x1c/0x20 (x2/y2) trace to a
    // real [touch previousLocationInView:window] call. x2/y2 = previousLocationInView:,
    // NOT a duplicate of the current point.
    gsProxy->x2 = previousPoint.x; gsProxy->y2 = previousPoint.y;
    gsProxy->x3 = point.x;         gsProxy->y3 = point.y;
    gsProxy->sizeX = 1.0;          gsProxy->sizeY = 1.0;

    // FIX #13 (SIXTH PASS — REVERTED back to correct; the fifth pass had this
    // backwards). Ground truth, read directly from
    // __UIEvent_Synthesize__initWithTouch__ in iGameGod.c:
    //     uVar9 = 0x1010180;                    // default
    //     if (phase != Ended(3)) {
    //         if (phase != Cancelled(4)) {
    //             uVar9 = 0x3010180;             // override
    //         }
    //     }
    // So Began(0)/Moved(1)/Stationary(2) get the OVERRIDE (0x3010180), and
    // Ended(3)/Cancelled(4) BOTH keep the DEFAULT (0x1010180). type = 0xbb9
    // (3001), confirmed constant across all phases.
    gsProxy->flags = (phase == UITouchPhaseEnded || phase == UITouchPhaseCancelled)
        ? 0x1010180
        : 0x3010180;
    gsProxy->type = 3001;

    Class touchesEventClass = NSClassFromString(@"UITouchesEvent");
    if (!touchesEventClass) {
        os_log(gg_log, "[Touch]: buildSyntheticEventForTouch — UITouchesEvent class not found via NSClassFromString, falling back to plain UIEvent");
    }
    UIEvent *syntheticEvent = [touchesEventClass alloc];
    NSSet *touchesSet = [NSSet setWithObject:touch];

    if ([syntheticEvent respondsToSelector:@selector(_initWithEvent:touches:)]) {
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Warc-performSelector-leaks"
        syntheticEvent = [syntheticEvent performSelector:@selector(_initWithEvent:touches:) withObject:gsProxy withObject:touchesSet];
        #pragma clang diagnostic pop
    } else {
        os_log(gg_log, "[Touch]: buildSyntheticEventForTouch — _initWithEvent:touches: not available on %{public}@, falling back to plain UIEvent", touchesEventClass);
        syntheticEvent = [[UIEvent alloc] init];
    }
    if (!syntheticEvent) {
        os_log(gg_log, "[Touch]: buildSyntheticEventForTouch — _initWithEvent:touches: itself returned nil (this triggers the Layer-A fallback path)");
    } else {
        // FIX #17 (SIXTH PASS, new): ground truth explicitly writes _timestamp
        // onto the constructed EVENT object itself immediately after
        // _initWithEvent:touches: returns non-nil — this file previously only
        // ever timestamped the touch, never the event.
        NSTimeInterval eventTimestamp = [[NSProcessInfo processInfo] systemUptime];
        [self writeScalarIvarOnObject:syntheticEvent name:"_timestamp" type:@encode(NSTimeInterval) valuePtr:&eventTimestamp];
    }
    return syntheticEvent;
}

// FIX #3: send to the touch's window directly, UIApplication only as a
// nil-window fallback — matches _TouchSynthesisDispatchEvent(event, window):
// `if (window == nil) [[UIApplication sharedApplication] sendEvent:event];
//  else [window sendEvent:event];`
// Also applies the confirmed exclusion-toggle behavior: never skip the
// send, just briefly disable userInteractionEnabled on the excluded view
// when it currently owns the dispatch window and is interactive.
- (void)sendSyntheticEvent:(UIEvent *)syntheticEvent toWindow:(UIWindow *)dispatchWindow {
    UIView *excluded = self.excludedTouchView;
    BOOL isExcludable = (excluded != nil &&
                         excluded.window != nil &&
                         excluded.window == dispatchWindow &&
                         excluded.isUserInteractionEnabled);

    os_log(gg_log, "[Touch]: sendSyntheticEvent -> %{public}@ (excludable: %{public}@)",
          dispatchWindow ? [NSString stringWithFormat:@"%@", dispatchWindow] : @"nil, will use [UIApplication sharedApplication]",
          isExcludable ? @"YES" : @"NO");

    void (^send)(void) = ^{
        if (dispatchWindow) {
            [dispatchWindow sendEvent:syntheticEvent];
        } else {
            [[UIApplication sharedApplication] sendEvent:syntheticEvent];
        }
    };

    if (isExcludable) {
        BOOL previousState = excluded.isUserInteractionEnabled;
        excluded.userInteractionEnabled = NO;
        send();
        excluded.userInteractionEnabled = previousState;
    } else {
        send();
    }
}

// 3. The Core Per-Event Dispatcher (FUN_00584b84's actual per-tick path)
// FIX #1/#2: this now does Layer B ONLY. Direct responder-chain calls and
// unconditional gesture-recognizer fallback have been removed from here —
// see the changelog at the top of this file and -finalizeAnyActiveTouch:
// below for where Layer A actually belongs.
- (void)dispatchHybridTouchAtPoint:(CGPoint)point phase:(UITouchPhase)phase inOutTouch:(UITouch **)activeTouch {
    os_log(gg_log, "[Touch]: dispatchHybridTouchAtPoint (%.1f, %.1f) phase=%ld", point.x, point.y, (long)phase);
    UIWindow *keyWindow = GG_KeyWindow();
    if (!keyWindow) {
        os_log(gg_log, "[Touch]: dispatchHybridTouchAtPoint aborted — GG_KeyWindow() returned nil");
        return;
    }

    UITouch *touch = *activeTouch;
    BOOL isNewTouch = (!touch || phase == UITouchPhaseBegan);

    // Setup UITouch
    // NOTE (unresolved, not changed here): the binary's touch-creation entry
    // point calls [super init] with UITouch explicitly as the superclass,
    // which only compiles that way if the implementing class is itself a
    // UITouch *subclass* rather than a plain category on UITouch. This file
    // still allocates a bare UITouch. That's a long-established technique
    // and should keep working, but it's a structural difference from what
    // the binary appears to do, worth re-checking if you see odd behavior
    // tied to +alloc/-init on UITouch specifically.
    if (isNewTouch) {
        touch = [[UITouch alloc] init];
        *activeTouch = touch;

        UIView *targetView = [keyWindow hitTest:point withEvent:nil];
        if (!targetView) targetView = keyWindow;
        os_log(gg_log, "[Touch]: new touch — hitTest at (%.1f, %.1f) -> %{public}@", point.x, point.y, targetView);
        os_log(gg_log, "[Touch]: hitTest view class=%{public}@ gestureRecognizers=%{public}@",
              NSStringFromClass([targetView class]),
              targetView.gestureRecognizers);

        [self defensiveSetObject:keyWindow forProperty:@"window" onObject:touch];
        [self defensiveSetObject:targetView forProperty:@"view" onObject:touch];

        // FIX #10: _tapCount was never initialized before. Confirmed the
        // binary always writes the passed-in count (1, for a single-point
        // touch) into _tapCount when a new touch is created.
        NSInteger tapCount = 1;
        [self writeScalarIvarOnObject:touch name:"_tapCount" type:@encode(NSInteger) valuePtr:&tapCount];

        // FIX #11: confirmed a brand-new touch always has _touchFlags bits
        // 0x1 and 0x2 set (`*touchFlags |= 3`) — previously never set here.
        uint16_t *touchFlagsPtr = (uint16_t *)[self rawIvarPointerOnObject:touch name:"_touchFlags"];
        if (touchFlagsPtr) *touchFlagsPtr |= 0x3;
    }

    // Capture the touch's location BEFORE we overwrite it. Used below by
    // the FIX #11 movement-threshold check, and by GSEventProxy's x2/y2
    // fields (FIX #8, now resolved via disassembly — see buildSyntheticEventForTouch:).
    CGPoint previousPoint = isNewTouch
        ? point
        : [self readCGPointIvarOnObject:touch name:"_locationInWindow" fallback:point];

    // FIX #11 (continued): confirmed setLocationInWindow: clears _touchFlags
    // bit 0x2 whenever the move exceeds 2.0pt in x or y. Only applies to
    // updates on an existing touch — a brand-new touch's flags were just
    // set above and shouldn't be immediately cleared again.
    if (!isNewTouch) {
        CGFloat dx = point.x - previousPoint.x;
        CGFloat dy = point.y - previousPoint.y;
        if (fabs(dx) > 2.0 || fabs(dy) > 2.0) {
            uint16_t *touchFlagsPtr = (uint16_t *)[self rawIvarPointerOnObject:touch name:"_touchFlags"];
            if (touchFlagsPtr) *touchFlagsPtr &= 0xFFFD;
        }
    }

    NSTimeInterval timestamp = [[NSProcessInfo processInfo] systemUptime];
    [self writeScalarIvarOnObject:touch name:"_locationInWindow" type:@encode(CGPoint) valuePtr:&point];
    [self writeScalarIvarOnObject:touch name:"_previousLocationInWindow" type:@encode(CGPoint) valuePtr:&previousPoint];
    [self writeScalarIvarOnObject:touch name:"_phase" type:@encode(UITouchPhase) valuePtr:&phase];
    [self writeScalarIvarOnObject:touch name:"_timestamp" type:@encode(NSTimeInterval) valuePtr:&timestamp];

    UIEvent *syntheticEvent = [self buildSyntheticEventForTouch:touch atPoint:point previousPoint:previousPoint phase:phase];

    UIWindow *dispatchWindow = touch.window ?: keyWindow;
    os_log(gg_log, "[Touch]: touch.window=%{public}@ touch.view=%{public}@ syntheticEvent=%{public}@ dispatchWindow=%{public}@",
          touch.window, touch.view, syntheticEvent ? @"built OK" : @"NIL", dispatchWindow);

    // FIX #1 (RESOLVED): confirmed via disassembly that event creation can
    // genuinely return nil (a real, reachable path in -[UIEvent(Synthesize)
    // _initWithTouch:] when the private _initWithEvent:touches: initializer
    // itself returns nil). When that happens, the binary does NOT call
    // sendEvent: at all — it routes to FUN_0058eb90 instead. Mirror that here.
    if (syntheticEvent) {
        [self sendSyntheticEvent:syntheticEvent toWindow:dispatchWindow];
    } else {
        os_log(gg_log, "[Touch]: syntheticEvent was nil — routing to performEventCreationFailureFallbackForTouch (Layer A)");
    }

    // FIX #18 (SEVENTH PASS — empirical, NOT ground-truth-derived. Flagging
    // that explicitly since everything else in this file traces to a
    // decompiled/disassembled source; this one doesn't.):
    // On-device logging (UnityDiagnostics [Diag] hook) showed real finger
    // touches firing -[UnityView touchesBegan/Moved/Ended:withEvent:]
    // immediately every time. The exact same phase sequence sent via
    // sendSyntheticEvent:toWindow: (Layer B, event always built OK, sendEvent:
    // always called, no nil, no exception) produced ZERO [Diag] lines for the
    // entire swipe — sendEvent: is silently swallowing a well-formed event.
    // That's not the failure mode FUN_0058eb90/Layer A was gated on in the
    // binary (a nil event) — it's a different, silent delivery failure this
    // build/device combo exhibits. Given that, Layer A is now called
    // UNCONDITIONALLY here, not just when event creation fails, since it's
    // the one path confirmed (by definition — it directly invokes the same
    // methods real touches land on) to actually reach UnityView.
    // KNOWN RISK: if sendEvent: DOES work correctly on some other
    // device/iOS build, this would double-fire touchesBegan/Moved/Ended for
    // the same touch (once via real delivery, once via this direct call).
    // Watch for duplicate [Diag] lines per dispatch if you retarget this at
    // a different device — if that happens, make this an else-branch again
    // (only on nil OR on a delivery-confirmation timeout) instead of
    // unconditional.
    [self performEventCreationFailureFallbackForTouch:touch targetView:touch.view event:syntheticEvent phase:phase];

    if (phase == UITouchPhaseEnded || phase == UITouchPhaseCancelled) {
        *activeTouch = nil;
    }
}

// 3b. FUN_0058eb90 — event-creation-failure fallback (RESOLVED via disassembly
// of FUN_00584b84 and FUN_0058eb90 directly, not inferred). Confirmed trigger:
// fires exactly when _TouchSynthesisCreateEvent returns nil (the branch is a
// literal `cbz x0, ...` on the create-call's return register, immediately
// after _objc_retainAutoreleasedReturnValue — no intervening instructions,
// so this is not the "opaque bit of param_4" some earlier analysis claimed).
// Confirmed exact phase->selector mapping from FUN_0058eb90's own body:
//   phase 0 (Began)     -> touchesBegan:withEvent:
//   phase 1 (Moved)     -> touchesMoved:withEvent:
//   phase 2              -> NO-OP, returns immediately, calls nothing
//   phase 3 (Ended)     -> touchesEnded:withEvent:
//   phase 4 (Cancelled) -> touchesCancelled:withEvent:
// Confirmed: the (possibly-nil) event object is passed straight through as
// the withEvent: argument in every case — this fallback does not require a
// valid UIEvent to fire.
- (void)performEventCreationFailureFallbackForTouch:(UITouch *)touch targetView:(UIView *)targetView event:(UIEvent *)syntheticEvent phase:(UITouchPhase)phase {
    if (!targetView) return;
    NSSet *touchesSet = [NSSet setWithObject:touch];
    switch (phase) {
        case UITouchPhaseBegan:
            if ([targetView respondsToSelector:@selector(touchesBegan:withEvent:)])
                [targetView touchesBegan:touchesSet withEvent:syntheticEvent];
            break;
        case UITouchPhaseMoved:
            if ([targetView respondsToSelector:@selector(touchesMoved:withEvent:)])
                [targetView touchesMoved:touchesSet withEvent:syntheticEvent];
            break;
        case UITouchPhaseStationary:
            // Confirmed: no-op in the binary. Calls nothing for this phase.
            break;
        case UITouchPhaseEnded:
            if ([targetView respondsToSelector:@selector(touchesEnded:withEvent:)])
                [targetView touchesEnded:touchesSet withEvent:syntheticEvent];
            break;
        case UITouchPhaseCancelled:
        default:
            if ([targetView respondsToSelector:@selector(touchesCancelled:withEvent:)])
                [targetView touchesCancelled:touchesSet withEvent:syntheticEvent];
            break;
    }
}


// CONFIRMED: this — not the per-event dispatcher above — is where direct
// touchesBegan:/Moved:/Ended:/Cancelled:withEvent: calls (Layer A) actually
// live in the binary. Its only callers there are session teardown
// (recording stop/clear) and replay-finished/looped. Call this instead of
// just discarding *activeTouch whenever a swipe/replay sequence is
// interrupted or torn down mid-flight, so the app doesn't end up thinking
// a finger is still down.
- (void)finalizeAnyActiveTouch:(UITouch **)activeTouch {
    UITouch *touch = *activeTouch;
    if (!touch) return;

    UIWindow *window = touch.window;
    if (!window) {
        os_log(gg_log, "[Touch]: finalizeAnyActiveTouch — touch.window is nil, dropping without dispatch");
        *activeTouch = nil;
        return;
    }
    os_log(gg_log, "[Touch]: finalizeAnyActiveTouch — phase=%ld window=%{public}@", (long)touch.phase, window);

    UIView *targetView = touch.view;
    UITouchPhase phase = touch.phase;
    NSSet *touchesSet = [NSSet setWithObject:touch];
    CGPoint point = [self readCGPointIvarOnObject:touch name:"_locationInWindow" fallback:CGPointZero];
    CGPoint previousPoint = [self readCGPointIvarOnObject:touch name:"_previousLocationInWindow" fallback:point];
    UIEvent *syntheticEvent = [self buildSyntheticEventForTouch:touch atPoint:point previousPoint:previousPoint phase:phase];

    // Layer B first (matches call order in FUN_0058442c).
    [self sendSyntheticEvent:syntheticEvent toWindow:window];

    // Layer A: dispatch directly based on the touch's actual current phase.
    if (targetView) {
        switch (phase) {
            case UITouchPhaseBegan:
                if ([targetView respondsToSelector:@selector(touchesBegan:withEvent:)])
                    [targetView touchesBegan:touchesSet withEvent:syntheticEvent];
                break;
            case UITouchPhaseMoved:
                if ([targetView respondsToSelector:@selector(touchesMoved:withEvent:)])
                    [targetView touchesMoved:touchesSet withEvent:syntheticEvent];
                break;
            case UITouchPhaseStationary:
                // Confirmed: the binary's cleanup pass silently drops this phase.
                break;
            case UITouchPhaseEnded:
                if ([targetView respondsToSelector:@selector(touchesEnded:withEvent:)])
                    [targetView touchesEnded:touchesSet withEvent:syntheticEvent];
                break;
            case UITouchPhaseCancelled:
            default:
                if ([targetView respondsToSelector:@selector(touchesCancelled:withEvent:)])
                    [targetView touchesCancelled:touchesSet withEvent:syntheticEvent];
                break;
        }
    }

    *activeTouch = nil;
}

// 4b. FUN_00587624's UIControl/cell action dispatcher, gated by FUN_0057efe4
// (fifth pass: fully resolved via decompiled C + disassembly cross-check of
// FUN_0057efe4 itself, obtained via a Ghidra call-graph dump). Both gates
// and the full 0-4 action-code decision chain are now confirmed, not
// approximated.
//
// CORRECTION to the fourth-pass version: Gate 1 was previously described as
// "block if the cached control doesn't match" — that was backwards. The
// real gate value (`local_b8` in the binary) is populated ONLY when the
// weakly-cached control DOES match the candidate, and even then holds a
// separate stored field from that cache entry — not a match/mismatch
// signal. The gate blocks when that field is non-zero, which in practice
// mostly means "this control was already consumed/fired, don't re-fire
// it" — a re-entrancy guard, not a match-requirement gate. Renamed
// accordingly below.
- (NSInteger)actionCodeForTouch:(UITouch *)touch
                candidateControl:(UIView *)candidate
                 boundsContainsPoint:(BOOL)boundsContainsPoint
                 alreadyConsumedFlag:(BOOL)alreadyConsumedFlag
                   allTargetsPresent:(BOOL)allTargetsPresent
                hasSecondaryTargets:(BOOL)hasSecondaryTargets
                   cellAncestorFound:(BOOL)cellAncestorFound
                        isKeyWindow:(BOOL)isKeyWindow
                       touchFlagsBit1:(BOOL)touchFlagsBit1 {
    // Gate 1 (corrected): re-entrancy guard, not a match requirement.
    if (alreadyConsumedFlag) return 0;
    // Gate 2 (confirmed unchanged): touch point must be inside target bounds.
    if (!boundsContainsPoint) return 0;

    BOOL candidatePresent = (candidate != nil);

    if (allTargetsPresent && candidatePresent) {
        return 1;
    }
    if (touchFlagsBit1) {
        if (candidatePresent) {
            return hasSecondaryTargets ? 2 : 3;
        } else if (cellAncestorFound) {
            return isKeyWindow ? 0 : 4;
        } else {
            return hasSecondaryTargets ? 2 : 3;
        }
    }
    return 0;
}

// Bounds/cache-gate helper matching the earlier (corrected) semantics, for
// callers that just need the pass/fail without the full 0-4 action code.
- (BOOL)shouldAllowControlDispatchForTouch:(UITouch *)touch onView:(UIView *)targetView {
    if (!targetView) return NO;
    CGPoint pointInTarget = [touch locationInView:targetView];
    if (!CGRectContainsPoint(targetView.bounds, pointInTarget)) {
        return NO;
    }
    // alreadyConsumedFlag intentionally not modeled here without a real
    // cache-entry equivalent wired up; treat as NO (not consumed) by default.
    return YES;
}

// NOTE: candidatePresent/allTargetsPresent/hasSecondaryTargets/
// cellAncestorFound/isKeyWindow/touchFlagsBit1 all need real inputs wired
// from your call site to use -actionCodeForTouch:... meaningfully — this
// implements the confirmed DECISION LOGIC precisely, but stops short of
// wiring live UIKit queries for every input (e.g. what "candidate" vs.
// touch's own gestureRecognizers/allTargets should be at your call site is
// still your call, not something uniquely determined by the binary without
// also resolving param_5 vs param_6's distinct roles, which remains
// unconfirmed — see param_8 note below).
// param_8 (a bitflag argument to FUN_00587624 itself, gating an entirely
// separate FUN_0058f2a0 pre-check before this logic even runs) is also not
// modeled here — its role wasn't traced.

// 5. Gesture-recognizer fallback (Layer C) — fires target/action directly,
// bypassing sendEvent:, gated by a bundle-ownership safety check. Not
// called automatically anywhere in this file anymore (see changelog #2);
// kept as a standalone, correctly-gated method you can wire up explicitly.

// FIX #12 (third pass): the binary's shared helper
// (_TouchSynthesisClassNameLooksUnsafeForGestureFallback) checks FOUR
// conditions, not two — WK: prefix, _WK: prefix, "Web" prefix, and a
// "WebKit" substring anywhere in the name. An earlier revision only
// checked the first two at both call sites. Factored into one helper here
// so the two call sites can't drift out of sync with each other again.
- (BOOL)classNameLooksUnsafeForGestureFallback:(NSString *)className {
    if ([className hasPrefix:@"WK"]) return YES;
    if ([className hasPrefix:@"_WK"]) return YES;
    if ([className hasPrefix:@"Web"]) return YES;
    if ([className containsString:@"WebKit"]) return YES;
    return NO;
}

- (BOOL)isSafeGestureFallbackTargetEntry:(id)target {
    // Per-entry check (nested inside the recognizer-level gate below).
    // Confirmed: if target's bundle is app-owned, entry passes immediately.
    if (!target) return NO;
    NSBundle *targetBundle = [NSBundle bundleForClass:[target class]];
    if (targetBundle == [NSBundle mainBundle]) return YES;
    NSString *mainPath = [[NSBundle mainBundle] bundlePath];
    NSString *targetPath = targetBundle.bundlePath;
    if (targetPath.length > 0 && mainPath.length > 0 && [targetPath hasPrefix:mainPath]) {
        return YES; // app-owned via path containment
    }

    // Not app-owned: reject on unsafe class name.
    NSString *className = NSStringFromClass([target class]);
    if ([self classNameLooksUnsafeForGestureFallback:className]) return NO;

    // Confirmed polarity: paths under /System/Library or containing
    // PrivateFrameworks are REJECTED here, not allowed. Everything else
    // with a non-empty path passes.
    if (targetPath.length == 0) return NO;
    if ([targetPath hasPrefix:@"/System/Library"]) return NO;
    if ([targetPath containsString:@"PrivateFrameworks"]) return NO;
    return YES;
}

- (BOOL)isSafeGestureFallbackTarget:(id)target {
    // Recognizer-level gate: reject only if not app-owned AND class name unsafe.
    if (!target) return NO;
    NSBundle *targetBundle = [NSBundle bundleForClass:[target class]];
    if (targetBundle == [NSBundle mainBundle]) return YES;
    NSString *mainPath = [[NSBundle mainBundle] bundlePath];
    NSString *targetPath = targetBundle.bundlePath;
    BOOL appOwned = (targetPath.length > 0 && mainPath.length > 0 && [targetPath hasPrefix:mainPath]);
    if (appOwned) return YES;
    NSString *className = NSStringFromClass([target class]);
    if ([self classNameLooksUnsafeForGestureFallback:className]) return NO;
    return YES;
}

- (id)extractGestureTarget:(id)entry {
    // Confirmed: binary tries "_target" first, falls back to "target".
    id target = [entry valueForKey:@"_target"];
    if (!target) target = [entry valueForKey:@"target"];
    return target;
}

- (SEL)extractGestureAction:(id)entry {
    // Confirmed: binary tries "_action" first, falls back to "action", then
    // handles three representations: NSString directly; NSValue wrapping a
    // boxed SEL pointer; otherwise falls back to -description. (Ground
    // truth actually round-trips the NSValue case through
    // NSStringFromSelector/NSSelectorFromString rather than casting the
    // pointer straight to SEL — functionally equivalent for a genuine
    // boxed SEL, kept as the simpler direct cast here.)
    id action = [entry valueForKey:@"_action"];
    if (!action) action = [entry valueForKey:@"action"];
    if (!action) return NULL;

    if ([action isKindOfClass:[NSString class]]) {
        return NSSelectorFromString(action);
    } else if ([action isKindOfClass:[NSValue class]]) {
        SEL boxed = (SEL)[action pointerValue];
        return boxed;
    } else if ([action respondsToSelector:@selector(description)]) {
        return NSSelectorFromString([action description]);
    }
    return NULL;
}

- (void)performGestureRecognizerFallbackOnView:(UIView *)view {
    for (UIGestureRecognizer *recognizer in view.gestureRecognizers) {
        // Recognizer-level gate first.
        if (![self isSafeGestureFallbackTarget:recognizer]) continue;

        // FIX #5: read _targets via KVC (_TouchSynthesisGestureTargetEntries
        // is literally [obj valueForKey:@"_targets"]), not raw Ivar access.
        NSArray *targets = nil;
        @try {
            id value = [recognizer valueForKey:@"_targets"];
            if ([value isKindOfClass:[NSArray class]]) targets = value;
        } @catch (__unused NSException *exception) {
            targets = nil;
        }
        if (!targets) targets = @[];

        // Confirmed: all-or-nothing per recognizer. If ANY valid
        // target/action entry fails the per-entry safety check, the binary
        // aborts the whole recognizer (fires nothing) rather than skipping
        // just that entry. Validate every entry first, then only fire if
        // all passed.
        NSMutableArray *validEntries = [NSMutableArray array];
        BOOL allSafe = YES;
        for (id targetEntry in targets) {
            id targetObj = [self extractGestureTarget:targetEntry];
            SEL action = [self extractGestureAction:targetEntry];
            // FIX #9: ground truth's per-entry gate is just
            // `target != nil && actionString.length != 0` — no
            // respondsToSelector: check. The check below is kept as an
            // explicit, UNVERIFIED extra safety net (prevents a crash if a
            // target genuinely doesn't implement the action) rather than a
            // confirmed behavior match. Remove it if you want exact parity.
            if (!targetObj || !action) continue;
            if (![targetObj respondsToSelector:action]) continue; // unverified addition, see FIX #9
            if (![self isSafeGestureFallbackTargetEntry:targetObj]) {
                allSafe = NO;
                break;
            }
            [validEntries addObject:@[targetObj, [NSValue valueWithPointer:action]]];
        }
        if (!allSafe) continue;
        if (validEntries.count == 0) continue;

        // FIX #4: set state ONCE, fire ALL entries, restore ONCE — matches
        // _TouchSynthesisPerformGestureRecognizerFallbackWithState, which
        // sets state before the enumeration loop and restores it after,
        // not per pair.
        UIGestureRecognizerState previousState = recognizer.state;
        [recognizer setValue:@(UIGestureRecognizerStateEnded) forKey:@"state"];

        for (NSArray *pair in validEntries) {
            id targetObj = pair[0];
            SEL action = (SEL)[pair[1] pointerValue];

            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Warc-performSelector-leaks"
            [[UIApplication sharedApplication] sendAction:action to:targetObj from:recognizer forEvent:nil];
            #pragma clang diagnostic pop
        }

        [recognizer setValue:@(previousState) forKey:@"state"];
    }
}
@end
// ---------------------------------------------------------
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
    UIWindow *keyWindow = GG_KeyWindow();
    if (!keyWindow || _actionButton) {
        os_log(gg_log, "[Touch]: showOverlayButton aborted — keyWindow=%{public}@ existingButton=%{public}@", keyWindow, _actionButton);
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
    os_log(gg_log, "[Touch]: overlay button added to %{public}@", keyWindow);
}

- (void)handlePan:(UIPanGestureRecognizer *)pan {
    CGPoint translation = [pan translationInView:pan.view.superview];
    pan.view.center = CGPointMake(pan.view.center.x + translation.x, pan.view.center.y + translation.y);
    [pan setTranslation:CGPointZero inView:pan.view.superview];
}

- (void)triggerSwipe {
    UIWindow *keyWindow = GG_KeyWindow();
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

    os_log(gg_log, "[Touch]: triggerSwipe — direction=%ld start=(%.1f,%.1f) end=(%.1f,%.1f)",
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
        os_log(gg_log, "[Touch]: -[UIWindow makeKeyAndVisible] swizzle %{public}@",
              (original && swizzled) ? @"installed" : @"FAILED — check selector names");
    });
}

// Post-swap, sending -gg_makeKeyAndVisible to self actually runs the
// ORIGINAL -makeKeyAndVisible implementation (classic swizzle idiom, same
// as -gg_touchSynthesis_window/-view above) — this is not infinite recursion.
- (void)gg_makeKeyAndVisible {
    [self gg_makeKeyAndVisible];
    os_log(gg_log, "[Touch]: -[UIWindow makeKeyAndVisible] hook fired for %{public}@ — scheduling overlay button in 1s", self);
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
    os_log(gg_log, "[Diag]: UnityView touchesBegan fired — count=%lu touches=%{public}@",
           (unsigned long)touches.count, touches);
    if (GG_OrigTouchesBegan) GG_OrigTouchesBegan(self, _cmd, touches, event);
}
static void GG_Diag_TouchesMoved(id self, SEL _cmd, NSSet *touches, UIEvent *event) {
    os_log(gg_log, "[Diag]: UnityView touchesMoved fired — count=%lu", (unsigned long)touches.count);
    if (GG_OrigTouchesMoved) GG_OrigTouchesMoved(self, _cmd, touches, event);
}
static void GG_Diag_TouchesEnded(id self, SEL _cmd, NSSet *touches, UIEvent *event) {
    os_log(gg_log, "[Diag]: UnityView touchesEnded fired — count=%lu", (unsigned long)touches.count);
    if (GG_OrigTouchesEnded) GG_OrigTouchesEnded(self, _cmd, touches, event);
}
static void GG_Diag_TouchesCancelled(id self, SEL _cmd, NSSet *touches, UIEvent *event) {
    os_log(gg_log, "[Diag]: UnityView touchesCancelled fired — count=%lu", (unsigned long)touches.count);
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
    os_log(gg_log, "[Diag]: retry check — attemptsRemaining=%d classFound=%{public}@",
           attemptsRemaining, unityViewClass ? @"YES" : @"NO");

    if (unityViewClass) {
        os_log(gg_log, "[Diag]: class found — installing UnityView touch diagnostics now");
        BOOL ok = YES;
        ok &= GG_SwizzleClassMethod(unityViewClass, @selector(touchesBegan:withEvent:),
                                     (IMP)GG_Diag_TouchesBegan, (IMP *)&GG_OrigTouchesBegan);
        ok &= GG_SwizzleClassMethod(unityViewClass, @selector(touchesMoved:withEvent:),
                                     (IMP)GG_Diag_TouchesMoved, (IMP *)&GG_OrigTouchesMoved);
        ok &= GG_SwizzleClassMethod(unityViewClass, @selector(touchesEnded:withEvent:),
                                     (IMP)GG_Diag_TouchesEnded, (IMP *)&GG_OrigTouchesEnded);
        ok &= GG_SwizzleClassMethod(unityViewClass, @selector(touchesCancelled:withEvent:),
                                     (IMP)GG_Diag_TouchesCancelled, (IMP *)&GG_OrigTouchesCancelled);
        os_log(gg_log, "[Diag]: UnityView touch diagnostics install %{public}@ (attemptsRemaining=%d)",
               ok ? @"succeeded" : @"PARTIALLY FAILED — check selector availability", attemptsRemaining);
        return;
    }
    if (attemptsRemaining <= 0) {
        os_log(gg_log, "[Diag]: UnityView class never appeared after retries — giving up on diagnostic hook");
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
    os_log(gg_log, "[Touch]: ===== BUILD MARKER: compiled %{public}s %{public}s =====", __DATE__, __TIME__);
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

    // center/end are computed from the screen bounds; GG_KeyWindow() (used
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
