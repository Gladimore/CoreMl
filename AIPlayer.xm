// =============================================================================
// AIPlayer.xm — standalone AI-only tweak.
//
// Shows a small floating "AI: OFF / AI: ON" button. Tapping it starts/stops
// a screen-capture + Core ML inference loop. Every high-confidence detection
// is logged AND injected into the game as a synthesized swipe (see
// TouchInjector below) — the button flashes gold for ~150ms each time a
// swipe is actually sent, so you can visually confirm firing rate.
//
// TouchInjector uses undocumented IOHIDEvent digitizer APIs (the standard
// jailbreak-tweak technique for synthetic touch input — see the class
// comment for details and caveats). It only works on a jailbroken device
// with SpringBoard-level HID access; it will silently no-op on stock iOS.
// Validate against on-screen gameplay with AI: ON before trusting it in a
// real run — these are private APIs with no stability guarantee across
// iOS versions.
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

static const NSInteger kSlowBranchEveryNTicks = 6;  // VALIDATE against SLOW_OFFSETS spacing
static const float     kDetectionThreshold    = 0.80f;
static const int       kTargetFPS             = 24;

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

    MLMultiArray *fastFrame = [self packFrame:gray diff:diff];
    MLMultiArray *slowFrame = hasNewSlow ? fastFrame : [self zerosShape:@[@1, @2, @(kImgSize), @(kImgSize)]];
    MLMultiArray *hasSlowArr = [self zerosShape:@[@1]];
    hasSlowArr[0] = hasNewSlow ? @1.0f : @0.0f;

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
// MARK: - TouchInjector — synthesizes swipes via IOHIDEvent digitizer events
//
// This is the standard technique used across jailbreak-tweak touch-simulation
// tools (e.g. STHIDEventGenerator-style utilities): build IOHIDEvent
// "digitizer finger" events by hand and dispatch them straight into the HID
// event system, the same path a real finger's events travel. There is no
// public API for this — the declarations below are reconstructed from
// widely-circulated reverse-engineering references, not an Apple header.
//
// Caveats (read before relying on this):
//   • Undocumented/private. Field names, bit values, and behavior can change
//     between iOS versions with no notice and no deprecation warning.
//   • Requires SpringBoard-level HID access, which a system-injected dylib
//     on a jailbroken device has — this will silently do nothing on stock
//     iOS or in a sandboxed app.
//   • kIOHIDDigitizerEventSenderID below is a commonly-reused placeholder
//     sender ID, not something read from the real touchscreen driver on
//     your specific device. It has worked broadly in practice, but if
//     injected touches don't land, try clearing it (senderID 0) first.
//
// A synthetic swipe is dispatched as three phases, a few ms apart:
//   1. finger down at the start point   (touch=YES, range=YES)
//   2. a handful of interpolated moves  (touch=YES, range=YES)
//   3. finger up at the end point       (touch=NO,  range=NO)
// =============================================================================

typedef double IOHIDFloat;
typedef uint32_t IOOptionBits;   // real def is `typedef UInt32 IOOptionBits` in IOKit/IOTypes.h —
                                 // declared by hand here because that header (via IOReturn.h /
                                 // device_types.h) trips a Clang-modules-in-extern-C error on
                                 // some SDK/toolchain combos. Same width, no header dependency.
typedef struct __IOHIDEvent *IOHIDEventRef;
typedef struct __IOHIDEventSystemClient *IOHIDEventSystemClientRef;

extern "C" {
extern IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef allocator);
extern void IOHIDEventSystemClientDispatchEvent(IOHIDEventSystemClientRef client, IOHIDEventRef event);
extern IOHIDEventRef IOHIDEventCreateDigitizerFingerEvent(
    CFAllocatorRef allocator, uint64_t timeStamp, uint32_t index, uint32_t identity,
    uint32_t eventMask, IOHIDFloat x, IOHIDFloat y, IOHIDFloat z,
    IOHIDFloat tipPressure, IOHIDFloat twist, Boolean range, Boolean touch, IOOptionBits options);
extern void IOHIDEventSetSenderID(IOHIDEventRef event, uint64_t senderID);
}

static const uint32_t kIOHIDDigitizerEventRange    = 1 << 0;
static const uint32_t kIOHIDDigitizerEventTouch    = 1 << 1;
static const uint32_t kIOHIDDigitizerEventPosition = 1 << 2;
static const uint64_t kIOHIDDigitizerEventSenderID = 0x8000000817319375ULL;

@interface TouchInjector : NSObject
+ (instancetype)sharedInjector;
// YES iff IOHIDEventSystemClientCreate succeeded, i.e. injectSwipeFrom:to:duration:
// can actually reach the HID system. NO on a non-jailbroken/sideloaded install
// (see init below) -- callers should not report/flash success when this is NO.
@property (nonatomic, readonly) BOOL isAvailable;
// start/end are in points, in the same coordinate space as UIScreen.mainScreen.bounds.
- (void)injectSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration;
@end

@implementation TouchInjector {
    IOHIDEventSystemClientRef _client;
}

- (BOOL)isAvailable { return _client != nil; }

+ (instancetype)sharedInjector {
    static TouchInjector *inst = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ inst = [TouchInjector new]; });
    return inst;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
        if (!_client) os_log(AIPlayerLog(), "TouchInjector: IOHIDEventSystemClientCreate returned NULL "
                            "(expected on a non-jailbroken device — injection will no-op)");
    }
    return self;
}

- (void)sendFingerEventAtPoint:(CGPoint)p touch:(BOOL)touch range:(BOOL)range identity:(uint32_t)identity {
    if (!_client) return;
    uint32_t mask = kIOHIDDigitizerEventRange | kIOHIDDigitizerEventTouch | kIOHIDDigitizerEventPosition;
    IOHIDEventRef event = IOHIDEventCreateDigitizerFingerEvent(
        kCFAllocatorDefault, mach_absolute_time(), 0, identity, mask,
        p.x, p.y, 0, touch ? 1.0 : 0.0, 0, range, touch, 0);
    if (!event) return;
    IOHIDEventSetSenderID(event, kIOHIDDigitizerEventSenderID);
    IOHIDEventSystemClientDispatchEvent(_client, event);
    CFRelease(event);
}

// Runs the down->move->up sequence on a background queue via usleep, so this
// call returns immediately and never blocks the capture callback that
// triggered it.
- (void)injectSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration {
    static uint32_t identityCounter = 1000;   // arbitrary range, kept away from real-finger identities
    uint32_t identity = identityCounter++;
    const NSInteger steps = 8;
    const NSTimeInterval stepInterval = duration / steps;

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
        [self sendFingerEventAtPoint:start touch:YES range:YES identity:identity];
        for (NSInteger i = 1; i <= steps; i++) {
            usleep((useconds_t)(stepInterval * 1e6));
            CGFloat t = (CGFloat)i / steps;
            CGPoint p = CGPointMake(start.x + (end.x - start.x) * t,
                                     start.y + (end.y - start.y) * t);
            [self sendFingerEventAtPoint:p touch:YES range:YES identity:identity];
        }
        usleep((useconds_t)(stepInterval * 1e6));
        [self sendFingerEventAtPoint:end touch:NO range:NO identity:identity];
    });
}

@end

// =============================================================================
// MARK: - In-process touch injection (KIF technique, works without jailbreak)
//
// TouchInjector above goes through IOHIDEventSystemClient, which requires a
// SpringBoard-level HID entitlement a Sideloadly-injected, non-jailbroken
// process cannot obtain (see TouchInjector's -init) -- that's an OS-enforced
// entitlement/AMFI boundary, not something fixable by writing different code
// while non-jailbroken.
//
// InProcessTouchInjector takes a different path entirely: it never asks the
// OS to deliver anything system-wide. It reuses UIApplication's own
// UITouchesEvent object (the same one UIKit's real touch pipeline uses),
// attaches a hand-crafted IOHIDEventRef to it and to a synthetic UITouch via
// private setters, and calls -[UIApplication sendEvent:] itself -- entirely
// in-process, no privileged dispatch involved. This is the technique the KIF
// iOS UI-testing framework (Apache-2.0, Copyright 2011-2016 Square, Inc.,
// https://github.com/kif-framework/KIF) has shipped in production for over a
// decade, including an explicit "iOS 26 compatibility fix" in v3.12.2
// (https://github.com/kif-framework/KIF/releases) confirming the injection
// mechanism itself -- as distinct from hit-testing coordinate resolution,
// which is what that particular fix addressed -- is unchanged on iOS 26.
// Adapted here from KIF's Sources/KIF/Additions/{UIView,UITouch,UIEvent,
// UIApplication}-KIFAdditions.{h,m} and Sources/KIF/Classes/IOHIDEvent+KIF.
// {h,m}, trimmed to just the down/move/up swipe path this needs (no
// UIWebView/WKWebView special-casing, no multi-finger gestures, no XCTest
// run-loop pumping).
//
// DEFENSIVE BY DESIGN: +isSupported probes every private selector via
// respondsToSelector:/instancesRespondToSelector: before any of this is
// ever invoked, and the actual injection sequence is wrapped in @try/@catch
// as a second line of defense -- if some future iOS renames or removes one
// of these, this logs clearly once and disables itself for the rest of the
// session instead of crashing the host game process.
// =============================================================================

@interface AIOverlayWindow : UIWindow
@end

// ---- private selectors this relies on (all confirmed present as of KIF
//      v3.12.3's source, the release with the explicit iOS 26 fix) ----

@interface UIApplication (AIPlayerPrivateTouch)
- (UIEvent *)_touchesEvent;
// SPECULATIVE — selector name only confirmed present via method-list dump;
// argument types below are a guess from the selector shape (touches set,
// the event, and "touchable" -- presumably the target view/window), not a
// verified signature. May be for Pencil-style estimated-touch updates
// specifically rather than general touch registration -- semantics unknown.
// Wrapped in a respondsToSelector: + NSInvocation-free direct call, and the
// whole attempt is easy to strip if it doesn't pan out.
- (void)_registerEstimatedTouches:(NSSet<UITouch *> *)touches event:(UIEvent *)event forTouchable:(id)touchable;
@end

@interface UIEvent (AIPlayerPrivateTouch)
// -_clearTouches and -_addTouch:forDelayedDelivery: removed here -- confirmed
// gone on iOS 26.5 via live method-list dump, not called anywhere anymore.
- (void)_setHIDEvent:(IOHIDEventRef)event;
@end

@interface UITouch (AIPlayerPrivateTouch)
- (void)setWindow:(UIWindow *)window;
- (void)setView:(UIView *)view;
- (void)setTapCount:(NSUInteger)tapCount;
- (void)setPhase:(UITouchPhase)phase;
- (void)setTimestamp:(NSTimeInterval)timestamp;
- (void)setGestureView:(UIView *)view;
- (void)_setLocationInWindow:(CGPoint)location resetPrevious:(BOOL)resetPrevious;
- (void)_setIsFirstTouchForView:(BOOL)firstTouchForView;
// NOTE: lowercase "id", different casing than UIEvent's _setHIDEvent: above.
// Confirmed against KIF's actual current source -- not a typo.
- (void)_setHidEvent:(IOHIDEventRef)event;
@end

// ---- low-level hand+finger IOHIDEvent construction, ported from KIF's
//      IOHIDEvent+KIF.m (same file the "iOS 26 compatibility" release
//      still ships unchanged) ----

typedef struct { uint32_t hi; uint32_t lo; } AIPlayerAbsoluteTime;

extern "C" {
extern IOHIDEventRef IOHIDEventCreateDigitizerEvent(
    CFAllocatorRef allocator, AIPlayerAbsoluteTime timeStamp, uint32_t transducerType,
    uint32_t index, uint32_t identity, uint32_t eventMask, uint32_t buttonMask,
    IOHIDFloat x, IOHIDFloat y, IOHIDFloat z, IOHIDFloat tipPressure, IOHIDFloat barrelPressure,
    Boolean range, Boolean touch, IOOptionBits options);
extern IOHIDEventRef IOHIDEventCreateDigitizerFingerEventWithQuality(
    CFAllocatorRef allocator, AIPlayerAbsoluteTime timeStamp, uint32_t index, uint32_t identity, uint32_t eventMask,
    IOHIDFloat x, IOHIDFloat y, IOHIDFloat z, IOHIDFloat tipPressure, IOHIDFloat twist,
    IOHIDFloat minorRadius, IOHIDFloat majorRadius, IOHIDFloat quality, IOHIDFloat density, IOHIDFloat irregularity,
    Boolean range, Boolean touch, IOOptionBits options);
extern void IOHIDEventAppendEvent(IOHIDEventRef event, IOHIDEventRef childEvent);
extern void IOHIDEventSetIntegerValue(IOHIDEventRef event, uint32_t field, int value);
}

static const uint32_t kAIPlayerIOHIDDigitizerTransducerTypeHand = 3;   // kIOHIDDigitizerTransducerTypeHand
static const uint32_t kAIPlayerIOHIDEventTypeDigitizer          = 11;  // kIOHIDEventTypeDigitizer
static const uint32_t kAIPlayerIOHIDDigitizerEventRangeFlag     = 0x00000001;
static const uint32_t kAIPlayerIOHIDDigitizerEventTouchFlag     = 0x00000002;
static const uint32_t kAIPlayerIOHIDDigitizerEventPositionFlag  = 0x00000004;
// kIOHIDEventFieldDigitizerIsDisplayIntegrated is the 26th (index 25) entry
// of the digitizer field enum in KIF's IOHIDEvent+KIF.m -- vendored as a
// named constant rather than re-deriving the index by hand to avoid an
// off-by-one.
static const uint32_t kAIPlayerIOHIDEventFieldDigitizerIsDisplayIntegrated =
    (kAIPlayerIOHIDEventTypeDigitizer << 16) + 25;

static IOHIDEventRef AIPlayerBuildTouchHIDEvent(NSArray<UITouch *> *touches) {
    uint64_t abTime = mach_absolute_time();
    AIPlayerAbsoluteTime timeStamp = { (uint32_t)(abTime >> 32), (uint32_t)abTime };

    IOHIDEventRef handEvent = IOHIDEventCreateDigitizerEvent(
        kCFAllocatorDefault, timeStamp, kAIPlayerIOHIDDigitizerTransducerTypeHand,
        0, 0, kAIPlayerIOHIDDigitizerEventTouchFlag, 0,
        0, 0, 0, 0, 0, 0, true, 0);
    if (!handEvent) {
        // TEMP DIAGNOSTIC — if this fires, IOHIDEventCreateDigitizerEvent
        // itself is failing (likely a sandbox/entitlement rejection under
        // Sideloadly injection), not a downstream dispatch problem. Every
        // touch built from this point on has no real HID backing.
        static CFTimeInterval lastLog = 0;
        CFTimeInterval now = CACurrentMediaTime();
        if (now - lastLog > 2.0) {
            lastLog = now;
            os_log(AIPlayerLog(), "[DIAG] IOHIDEventCreateDigitizerEvent returned NULL — "
                  "hand event construction failed, no HID event will be attached");
        }
        return NULL;
    }
    IOHIDEventSetIntegerValue(handEvent, kAIPlayerIOHIDEventFieldDigitizerIsDisplayIntegrated, 1);

    NSUInteger idx = 0;
    for (UITouch *touch in touches) {
        uint32_t eventMask = (touch.phase == UITouchPhaseMoved)
            ? kAIPlayerIOHIDDigitizerEventPositionFlag
            : (kAIPlayerIOHIDDigitizerEventRangeFlag | kAIPlayerIOHIDDigitizerEventTouchFlag);
        BOOL isTouching = (touch.phase != UITouchPhaseEnded);
        CGPoint p = [touch locationInView:touch.window];

        IOHIDEventRef fingerEvent = IOHIDEventCreateDigitizerFingerEventWithQuality(
            kCFAllocatorDefault, timeStamp, (uint32_t)(idx + 1), 2, eventMask,
            p.x, p.y, 0, 0, 0, 5.0, 5.0, 1.0, 1.0, 1.0,
            isTouching, isTouching, 0);
        idx++;
        if (!fingerEvent) continue;
        IOHIDEventSetIntegerValue(fingerEvent, kAIPlayerIOHIDEventFieldDigitizerIsDisplayIntegrated, 1);
        IOHIDEventAppendEvent(handEvent, fingerEvent);
        CFRelease(fingerEvent);
    }
    return handEvent;
}

@interface InProcessTouchInjector : NSObject
+ (instancetype)sharedInjector;
+ (BOOL)isSupported;   // caches the capability probe; re-checks failure state each call
@property (nonatomic, readonly) BOOL hasFailedThisSession;
// start/end are in the target window's coordinate space (see -targetWindow;
// for a full-screen game window this is the same as UIScreen.mainScreen.bounds).
- (void)injectSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration;
@end

@implementation InProcessTouchInjector {
    BOOL _disabledAfterFailure;
}

- (BOOL)hasFailedThisSession { return _disabledAfterFailure; }

+ (instancetype)sharedInjector {
    static InProcessTouchInjector *inst = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ inst = [InProcessTouchInjector new]; });
    return inst;
}

+ (BOOL)isSupported {
    static BOOL supported = NO;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        UIApplication *app = [UIApplication sharedApplication];
        NSMutableArray<NSString *> *missing = [NSMutableArray array];

        // NOTE: -[UIEvent _clearTouches] and -[UIEvent _addTouch:forDelayedDelivery:]
        // were dropped from this gate. Confirmed via a live class_copyMethodList
        // dump on-device (iOS 26.5): UIEvent's touch-mutation API is gone
        // entirely -- no renamed equivalent exists, superclass chain is just
        // UIEvent -> NSObject so it didn't move to a base class either.
        // dispatchTouch:phase: below no longer calls either selector; it
        // mutates the UITouch directly and re-fetches/sends the app's
        // -_touchesEvent instead. Everything still checked here (below) was
        // confirmed PRESENT in that same dump.
        if (![app respondsToSelector:@selector(_touchesEvent)]) [missing addObject:@"-[UIApplication _touchesEvent]"];
        if (![UIEvent instancesRespondToSelector:@selector(_setHIDEvent:)]) [missing addObject:@"-[UIEvent _setHIDEvent:]"];
        if (![UITouch instancesRespondToSelector:@selector(_setLocationInWindow:resetPrevious:)]) [missing addObject:@"-[UITouch _setLocationInWindow:resetPrevious:]"];
        if (![UITouch instancesRespondToSelector:@selector(_setHidEvent:)]) [missing addObject:@"-[UITouch _setHidEvent:]"];
        if (![UITouch instancesRespondToSelector:@selector(setPhase:)]) [missing addObject:@"-[UITouch setPhase:]"];
        if (![UITouch instancesRespondToSelector:@selector(setWindow:)]) [missing addObject:@"-[UITouch setWindow:]"];

        if (missing.count > 0) {
            os_log(AIPlayerLog(), "InProcessTouchInjector: unsupported on this iOS build, missing: %{public}@",
                  [missing componentsJoinedByString:@", "]);
            supported = NO;
        } else {
            UIEvent *probe = [app _touchesEvent];
            supported = (probe != nil);
            if (!supported) os_log(AIPlayerLog(), "InProcessTouchInjector: -[UIApplication _touchesEvent] returned nil");
        }
    });
    return supported && !([InProcessTouchInjector sharedInjector].hasFailedThisSession);
}

- (nullable UIWindow *)targetWindow {
    UIApplication *app = [UIApplication sharedApplication];
    UIWindow *fallback = nil;

    NSMutableArray<UIWindow *> *candidateWindows = [NSMutableArray array];
    for (UIScene *scene in app.connectedScenes) {
        if (![scene isKindOfClass:[UIWindowScene class]]) continue;
        UIWindowScene *windowScene = (UIWindowScene *)scene;
        [candidateWindows addObjectsFromArray:windowScene.windows];
    }

    for (UIWindow *w in candidateWindows) {
        if ([w isKindOfClass:[AIOverlayWindow class]]) continue;   // that's our own toggle-button window, not the game
        if (w.isKeyWindow) return w;
        if (!fallback) fallback = w;
    }
    return fallback;
}

- (UITouch *)touchAtPoint:(CGPoint)windowPoint inWindow:(UIWindow *)window {
    UITouch *touch = [UITouch new];
    [touch setWindow:window];
    [touch setTapCount:1];
    [touch _setLocationInWindow:windowPoint resetPrevious:YES];

    UIView *hit = [window hitTest:windowPoint withEvent:nil];
    [touch setView:hit];
    [touch setPhase:UITouchPhaseBegan];
    if ([touch respondsToSelector:@selector(_setIsFirstTouchForView:)]) {
        [touch _setIsFirstTouchForView:YES];
    }
    [touch setTimestamp:[[NSProcessInfo processInfo] systemUptime]];
    if ([touch respondsToSelector:@selector(setGestureView:)]) {
        [touch setGestureView:hit];
    }

    IOHIDEventRef hidEvent = AIPlayerBuildTouchHIDEvent(@[touch]);
    if (hidEvent) {
        [touch _setHidEvent:hidEvent];
        CFRelease(hidEvent);
    }
    return touch;
}

- (void)dispatchTouch:(UITouch *)touch phase:(UITouchPhase)phase {
    [touch setTimestamp:[[NSProcessInfo processInfo] systemUptime]];
    [touch setPhase:phase];

    // iOS 26.5 CHANGE: -[UIEvent _clearTouches] and
    // -[UIEvent _addTouch:forDelayedDelivery:] no longer exist (confirmed via
    // live method-list dump -- not renamed, just gone; UIEvent's superclass
    // chain is plain UIEvent -> NSObject, so it didn't move to a base class
    // either). UIEvent's touch set is no longer externally mutable on this
    // build. -[UIEvent _setHIDEvent:] is still present, so the working
    // hypothesis is that UIKit now derives an event's touches from the HID
    // event's digitizer payload internally, rather than from explicit
    // _addTouch: calls -- which is also consistent with _setHidEvent: still
    // being present and settable on UITouch itself.
    IOHIDEventRef hidEvent = AIPlayerBuildTouchHIDEvent(@[touch]);
    if (phase == UITouchPhaseBegan) {
        os_log(AIPlayerLog(), "[DIAG] dispatchTouch Began: hidEvent=%{public}s",
              hidEvent ? "attached" : "NULL (no HID backing)");
    }
    if (hidEvent) {
        [touch _setHidEvent:hidEvent];
    }

    UIApplication *app = [UIApplication sharedApplication];
    UIEvent *event = [app _touchesEvent];
    if (hidEvent) {
        [event _setHIDEvent:hidEvent];
    }
    if (hidEvent) CFRelease(hidEvent);

    // RESULT of the previous attempt: event.allTouches.count was 0 on nearly
    // every swipe -- attaching the HID event alone does NOT make UIKit
    // auto-populate the touch set. That hypothesis is disproven.
    //
    // NEXT ATTEMPT (speculative) -- try -[UIApplication
    // _registerEstimatedTouches:event:forTouchable:]. Selector name only, no
    // confirmed signature or semantics; may be Pencil-specific. Guarded by
    // respondsToSelector: so a mismatched signature just no-ops instead of
    // crashing if the real signature differs from our guess.
    if (phase == UITouchPhaseBegan &&
        [app respondsToSelector:@selector(_registerEstimatedTouches:event:forTouchable:)]) {
        UIView *touchable = touch.view ?: (id)[self targetWindow];
        @try {
            [app _registerEstimatedTouches:[NSSet setWithObject:touch] event:event forTouchable:touchable];
            os_log(AIPlayerLog(), "[DIAG] _registerEstimatedTouches:event:forTouchable: called, no exception");
        } @catch (NSException *ex) {
            os_log(AIPlayerLog(), "[DIAG] _registerEstimatedTouches:event:forTouchable: threw: %{public}@ %{public}@",
                  ex.name, ex.reason);
        }
    }

    // TEMP DIAGNOSTIC — allTouches.count AFTER the registration attempt
    // above, so we can tell whether it changed anything.
    if (phase == UITouchPhaseBegan) {
        NSSet *all = [event respondsToSelector:@selector(allTouches)] ? [event allTouches] : nil;
        os_log(AIPlayerLog(), "[DIAG] dispatchTouch Began: event.allTouches.count=%lu (post-register attempt)",
              (unsigned long)all.count);
    }

    [app sendEvent:event];
}

// Runs entirely on the main thread (UIKit event delivery requires it), but
// this call itself returns immediately -- the down/move/up steps are
// scheduled via dispatch_after on the main queue so the capture callback
// that triggered this is never blocked.
- (void)injectSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration {
    if (_disabledAfterFailure || ![InProcessTouchInjector isSupported]) return;

    dispatch_async(dispatch_get_main_queue(), ^{
        @try {
            UIWindow *window = [self targetWindow];
            if (!window) {
                os_log(AIPlayerLog(), "InProcessTouchInjector: no target window found");
                return;
            }

            const NSInteger steps = 8;
            NSMutableArray<NSValue *> *points = [NSMutableArray arrayWithCapacity:steps + 1];
            for (NSInteger i = 0; i <= steps; i++) {
                CGFloat t = (CGFloat)i / steps;
                CGPoint p = CGPointMake(start.x + (end.x - start.x) * t,
                                         start.y + (end.y - start.y) * t);
                [points addObject:[NSValue valueWithCGPoint:p]];
            }

            UITouch *touch = [self touchAtPoint:points[0].CGPointValue inWindow:window];
            [self dispatchTouch:touch phase:UITouchPhaseBegan];

            NSTimeInterval stepInterval = duration / steps;
            for (NSInteger i = 1; i <= steps; i++) {
                int64_t delayNanos = (int64_t)(stepInterval * i * NSEC_PER_SEC);
                dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delayNanos), dispatch_get_main_queue(), ^{
                    @try {
                        [touch _setLocationInWindow:points[i].CGPointValue resetPrevious:NO];
                        [self dispatchTouch:touch phase:(i == steps ? UITouchPhaseEnded : UITouchPhaseMoved)];
                    } @catch (NSException *ex) {
                        [self handleFailure:ex];
                    }
                });
            }
        } @catch (NSException *ex) {
            [self handleFailure:ex];
        }
    });
}

- (void)handleFailure:(NSException *)ex {
    _disabledAfterFailure = YES;
    os_log(AIPlayerLog(), "InProcessTouchInjector: %{public}@ (%{public}@) -- one of the private selectors this "
          "relies on didn't behave as expected on this iOS build. Disabling in-process "
          "injection for the rest of this session; TouchInjector (system HID) remains as "
          "the fallback path, though it will itself no-op without a jailbreak.",
          ex.name, ex.reason);
}

@end

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

    // center/end are computed from the screen bounds; for a normal full-
    // screen game window this is the same coordinate space InProcessTouchInjector
    // expects (window.bounds), so no conversion is needed here.
    if ([InProcessTouchInjector isSupported]) {
        [[InProcessTouchInjector sharedInjector] injectSwipeFrom:center to:end duration:kSwipeDuration];
        __weak typeof(self) weakSelf = self;
        dispatch_async(dispatch_get_main_queue(), ^{ [weakSelf flashInjectionFeedback]; });
        return;
    }

    // Fall back to the system-HID path -- only actually delivers on a
    // jailbroken device (see TouchInjector's -init), but kept as a fallback
    // in case InProcessTouchInjector ever disables itself mid-session.
    TouchInjector *injector = [TouchInjector sharedInjector];
    if (!injector.isAvailable) {
        // Don't flash gold here -- that would falsely claim the swipe landed.
        // See TouchInjector's init: on a non-jailbroken/Sideloadly-injected
        // install, IOHIDEventSystemClientCreate returns NULL and every
        // injected touch below this point is silently dropped by design.
        // The detection/direction pipeline is working correctly (that's why
        // we got this far) -- this is strictly a "can't reach the HID
        // system from here" limitation, not a model or preprocessing bug.
        static CFTimeInterval lastUnavailableLog = 0;
        CFTimeInterval now = CACurrentMediaTime();
        if (now - lastUnavailableLog > 5.0) {
            lastUnavailableLog = now;
            os_log(AIPlayerLog(), "swipe dir=%ld conf=%.2f det=%.2f — model fired correctly but "
                  "neither InProcessTouchInjector nor TouchInjector can deliver on this "
                  "install; swipe NOT delivered", (long)dir, conf, det);
        }
        return;
    }

    [injector injectSwipeFrom:center to:end duration:kSwipeDuration];

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
