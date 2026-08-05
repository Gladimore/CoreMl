// =============================================================================
// AIPlayer.xm — standalone AI-only tweak.
//
// Shows a small floating "AI: OFF / AI: ON" button. Tapping it starts/stops
// a screen-capture + Core ML inference loop. Every high-confidence detection
// is logged AND injected into the game as a synthesized swipe (see
// DirectTouchInjector below) — the button flashes gold for ~150ms each time
// a swipe is actually sent, so you can visually confirm firing rate.
//
// TOUCH SYNTHESIS: DirectTouchInjector below is the technique validated
// on-device against Subway Surfers (GrayHueCapture/Tweak_5.xm-6.xm): build a
// real UITouch, write its private ivars directly (window, view, location,
// phase, timestamp, tapCount, touchFlags), then call
// touchesBegan/Moved/Ended/Cancelled:withEvent: DIRECTLY on the hit-tested
// view. No sendEvent:, no hand-built GSEvent/UIEvent struct, no IOHID, no
// jailbreak required — private ivar access and a plain method call are both
// available to any process in its own address space. On-device logging
// proved this is the one path that actually reaches a Unity game's
// touchesBegan handlers; sendEvent:-based delivery silently dropped
// well-formed synthetic events with no error and no nil. Validate against
// on-screen gameplay with AI: ON before trusting it in a real run — this
// still relies on private ivar names that could change across iOS versions,
// even though it needs no special entitlements.
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
// MARK: - Passthrough window (forward-declared: DirectTouchInjector below
// needs to recognize and skip it when picking a dispatch target)
// =============================================================================
@class AIOverlayWindow;

// =============================================================================
// MARK: - DirectTouchInjector — synthesizes swipes via direct responder-chain
// dispatch. No jailbreak, no IOHID, no sendEvent:.
//
// This replaces two earlier approaches that both lived in this file:
//   • A system-HID injector (IOHIDEventSystemClientDispatchEvent) — only
//     ever works on a jailbroken device with SpringBoard-level HID access;
//     silently no-ops under Sideloadly injection.
//   • An in-process sendEvent:-based injector (KIF's technique) — never got
//     a confirmed touch through on this iOS build; -[UIEvent _clearTouches]/
//     -_addTouch:forDelayedDelivery: are gone, and event.allTouches stayed
//     empty even after attaching a hand-built IOHIDEvent.
//
// This is the technique actually validated on-device against Subway
// Surfers: fully populate a real UITouch's private ivars (window, view,
// location, phase, timestamp, tapCount, touchFlags — the same set
// HybridTouchSynthesizer used in the GrayHueCapture project), then call
// touchesBegan/Moved/Ended/Cancelled:withEvent: DIRECTLY on the hit-tested
// view. On-device Console.app logging proved this is the one path that
// actually reaches a Unity game's overridden touch handlers — sendEvent:
// silently dropped every well-formed synthetic event, no error, no nil.
// Passing withEvent:nil is confirmed fine: the hit-tested view's override
// only ever reads locationInView:/phase/timestamp off the touch itself, and
// never inspects the event object.
//
// Needs no special entitlements: private ivar access and a plain method
// call are both available to any process acting on objects in its own
// address space, jailbroken or not. -isAvailable is always YES.
// =============================================================================

@interface DirectTouchInjector : NSObject
+ (instancetype)sharedInjector;
@property (nonatomic, readonly) BOOL isAvailable;   // always YES — kept for call-site symmetry with the old injectors
- (void)injectSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration;
@end

@implementation DirectTouchInjector {
    UITouch *_activeTouch;
}

+ (instancetype)sharedInjector {
    static DirectTouchInjector *inst = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ inst = [DirectTouchInjector new]; });
    return inst;
}

- (BOOL)isAvailable { return YES; }

// ---- ivar helpers ----

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
        *(NSInteger *)ivarMemory = *(NSInteger *)valuePtr;
    }
}

- (CGPoint)readCGPointIvarOnObject:(id)object name:(const char *)name fallback:(CGPoint)fallback {
    Ivar ivar = class_getInstanceVariable([object class], name);
    if (!ivar) return fallback;
    ptrdiff_t offset = ivar_getOffset(ivar);
    void *ivarMemory = (uint8_t *)(__bridge void *)object + offset;
    return *(CGPoint *)ivarMemory;
}

- (void *)rawIvarPointerOnObject:(id)object name:(const char *)name {
    Ivar ivar = class_getInstanceVariable([object class], name);
    if (!ivar) return NULL;
    ptrdiff_t offset = ivar_getOffset(ivar);
    return (uint8_t *)(__bridge void *)object + offset;
}

- (void)defensiveSetObject:(id)value forProperty:(NSString *)propName onObject:(id)target {
    NSString *ivarName = [NSString stringWithFormat:@"_%@", propName];
    Ivar ivar = class_getInstanceVariable([target class], [ivarName UTF8String]);
    if (ivar) {
        object_setIvar(target, ivar, value);
    }

    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Warc-performSelector-leaks"
    id readBack = [target performSelector:NSSelectorFromString(propName)];
    if (readBack == value) {
        #pragma clang diagnostic pop
        return;
    }

    NSString *publicSelectorString = [NSString stringWithFormat:@"set%@:", [propName capitalizedString]];
    SEL publicSelector = NSSelectorFromString(publicSelectorString);
    if ([target respondsToSelector:publicSelector]) {
        [target performSelector:publicSelector withObject:value];
        if ([target performSelector:NSSelectorFromString(propName)] == value) {
            #pragma clang diagnostic pop
            return;
        }
    }
    #pragma clang diagnostic pop

    [target setValue:value forKey:propName];
}

// Mirrors GG_KeyWindow() from the validated GrayHueCapture tweak, plus an
// explicit skip of our own overlay window (which can never become key
// anyway, since AIOverlayWindow overrides -canBecomeKeyWindow to return NO
// — this check is just an extra safety net, not load-bearing).
- (nullable UIWindow *)keyWindow {
    if (@available(iOS 13.0, *)) {
        for (UIScene *scene in [UIApplication sharedApplication].connectedScenes) {
            if (scene.activationState != UISceneActivationStateForegroundActive) continue;
            if (![scene isKindOfClass:[UIWindowScene class]]) continue;
            for (UIWindow *window in ((UIWindowScene *)scene).windows) {
                if ([window isKindOfClass:NSClassFromString(@"AIOverlayWindow")]) continue;
                if (window.isKeyWindow) return window;
            }
        }
    }
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return [UIApplication sharedApplication].keyWindow;
    #pragma clang diagnostic pop
}

// Core per-event dispatcher — ports HybridTouchSynthesizer's validated
// approach, minus the sendEvent:/GSEventProxy path entirely (confirmed
// unnecessary; direct dispatch alone is what reached UnityView on-device).
- (void)dispatchTouchAtPoint:(CGPoint)point phase:(UITouchPhase)phase inOutTouch:(UITouch **)activeTouch {
    UIWindow *keyWindow = [self keyWindow];
    if (!keyWindow) return;

    UITouch *touch = *activeTouch;
    BOOL isNewTouch = (!touch || phase == UITouchPhaseBegan);

    if (isNewTouch) {
        touch = [[UITouch alloc] init];
        *activeTouch = touch;

        UIView *targetView = [keyWindow hitTest:point withEvent:nil];
        if (!targetView) targetView = keyWindow;

        [self defensiveSetObject:keyWindow forProperty:@"window" onObject:touch];
        [self defensiveSetObject:targetView forProperty:@"view" onObject:touch];

        NSInteger tapCount = 1;
        [self writeScalarIvarOnObject:touch name:"_tapCount" type:@encode(NSInteger) valuePtr:&tapCount];
        uint16_t *touchFlagsPtr = (uint16_t *)[self rawIvarPointerOnObject:touch name:"_touchFlags"];
        if (touchFlagsPtr) *touchFlagsPtr |= 0x3;
    }

    CGPoint previousPoint = isNewTouch
        ? point
        : [self readCGPointIvarOnObject:touch name:"_locationInWindow" fallback:point];

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

    // Confirmed on-device: direct dispatch with event:nil works fine — the
    // hit-tested view's touchesXXX override never inspects the event
    // object, only the touch's own properties. No sendEvent:, ever.
    UIView *targetView = touch.view;
    NSSet *touchesSet = [NSSet setWithObject:touch];
    if (targetView) {
        switch (phase) {
            case UITouchPhaseBegan:
                if ([targetView respondsToSelector:@selector(touchesBegan:withEvent:)])
                    [targetView touchesBegan:touchesSet withEvent:nil];
                break;
            case UITouchPhaseMoved:
                if ([targetView respondsToSelector:@selector(touchesMoved:withEvent:)])
                    [targetView touchesMoved:touchesSet withEvent:nil];
                break;
            case UITouchPhaseStationary:
                break;
            case UITouchPhaseEnded:
                if ([targetView respondsToSelector:@selector(touchesEnded:withEvent:)])
                    [targetView touchesEnded:touchesSet withEvent:nil];
                break;
            case UITouchPhaseCancelled:
            default:
                if ([targetView respondsToSelector:@selector(touchesCancelled:withEvent:)])
                    [targetView touchesCancelled:touchesSet withEvent:nil];
                break;
        }
    }

    if (phase == UITouchPhaseEnded || phase == UITouchPhaseCancelled) {
        *activeTouch = nil;
    }
}

// Runs entirely on the main thread (UIKit touch delivery requires it), but
// this call itself returns immediately — the down/move/up steps are
// scheduled via dispatch_after on the main queue so the capture callback
// that triggered this is never blocked.
- (void)injectSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration {
    dispatch_async(dispatch_get_main_queue(), ^{
        // If a previous swipe never reached Ended for some reason, finalize
        // it before starting a new one instead of silently leaking it.
        if (self->_activeTouch) {
            UITouch *temp = self->_activeTouch;
            CGPoint lastPoint = [self readCGPointIvarOnObject:temp name:"_locationInWindow" fallback:start];
            [self dispatchTouchAtPoint:lastPoint phase:UITouchPhaseCancelled inOutTouch:&temp];
            self->_activeTouch = temp;
        }

        const NSInteger steps = 8;
        NSTimeInterval stepInterval = duration / steps;

        UITouch *touch = self->_activeTouch;
        [self dispatchTouchAtPoint:start phase:UITouchPhaseBegan inOutTouch:&touch];
        self->_activeTouch = touch;

        for (NSInteger i = 1; i <= steps; i++) {
            int64_t delayNanos = (int64_t)(stepInterval * i * NSEC_PER_SEC);
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delayNanos), dispatch_get_main_queue(), ^{
                CGFloat t = (CGFloat)i / steps;
                CGPoint p = CGPointMake(start.x + (end.x - start.x) * t,
                                         start.y + (end.y - start.y) * t);
                UITouch *stepTouch = self->_activeTouch;
                [self dispatchTouchAtPoint:p
                                      phase:(i == steps ? UITouchPhaseEnded : UITouchPhaseMoved)
                                 inOutTouch:&stepTouch];
                self->_activeTouch = stepTouch;
            });
        }
    });
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
    // screen game window this is the same coordinate space DirectTouchInjector
    // expects (window.bounds), so no conversion is needed here.
    [[DirectTouchInjector sharedInjector] injectSwipeFrom:center to:end duration:kSwipeDuration];

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
}
