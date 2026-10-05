// =============================================================================
// AIPlayer.m — sideloaded screen-capture + Core ML swipe player.
//
// Pipeline, once per 24 Hz tick:
//
//   ReplayKit frame -> fixed-rate clock -> antialiased 128x128 resize
//     -> gray + diff -> SwipeEncoder (CNN)     -> 128-d feature, cached in a ring
//     -> last 8 / 2 features -> SwipeHead (GRU) -> det prob + direction
//     -> detection gate -> synthesized swipe on the game window
//
// TRAIN / DEPLOY PARITY
//   The checkpoint was trained and validated on windows evaluated from a ZERO
//   hidden state (8 fast frames [t-7..t] + 2 slow frames [t-19, t-13]).
//   Carrying GRU state across a whole session is a different function that was
//   never validated. This file therefore caches per-frame CNN features and
//   re-runs the windowed head every tick, which reproduces training exactly.
//   See convert_to_coreml.py for the exported I/O contract.
//
// PREPROCESSING CONTRACT (must match the dataset builder)
//   1. Square-stretch resize to 128x128 with an antialiased filter
//      (training used torchvision resize with antialias=True; a plain
//      bilinear point-sample aliases and fabricates motion in the diff channel).
//   2. Gray = 0.2989 R + 0.5870 G + 0.1140 B (ITU-R 601-2).
//   3. diff = clamp(floorDiv((gray[t] - gray[t-1]) * 127, 255) + 127, 0, 255);
//      the first frame's diff is neutral (127).
//   4. Raw 0..255 float32 into the model; it divides by 255 internally.
//   5. Frames are resampled onto a fixed 24 Hz clock (sample-and-hold when the
//      source skips frames), because the diff channel is rate dependent.
//
// Deployment: non-jailbroken, injected with Sideloadly next to the two
// compiled models (SwipeEncoder.mlmodelc, SwipeHead.mlmodelc).
// =============================================================================

#import <UIKit/UIKit.h>
#import <ReplayKit/ReplayKit.h>
#import <CoreImage/CoreImage.h>
#import <CoreML/CoreML.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <os/log.h>
#include <stdatomic.h>

// =============================================================================
// MARK: - Configuration
// =============================================================================

// Must match the checkpoint / dataset. Enums (not static const) so they can
// size C arrays.
enum {
    AIImgSize  = 128,                     // H = W of the model input
    AIPixels   = AIImgSize * AIImgSize,
    AIFeatDim  = 128,                     // encoder output width
    AIFastLen  = 8,                       // fast window: frames [t-7 .. t]
    AISlowLen  = 2,                       // slow samples
    AIHistory  = 20,                      // newest frame + 19 back (= max slow offset + 1)
};
static const NSInteger kSlowOffsets[AISlowLen] = { 19, 13 };   // frames behind t, oldest first

static NSString *const kEncoderModelName = @"SwipeEncoder.mlmodelc";
static NSString *const kHeadModelName    = @"SwipeHead.mlmodelc";

// Fixed-rate clock (training stream was 24 fps).
static const CFTimeInterval kTickInterval     = 1.0 / 24.0;
static const CFTimeInterval kTickTolerance    = 0.008;   // accept a frame this early (60 Hz jitter)
static const NSInteger      kMaxCatchUpTicks  = 3;       // longer gaps are treated as a discontinuity

// Detection gate. det_threshold was 0.6 in training; training also used
// pos_weight=8, which inflates probabilities, so the fire threshold is higher.
static const float kFireThreshold    = 0.80f;   // fire when prob >= this and the gate is armed
static const float kReleaseThreshold = 0.50f;   // re-arm once prob drops below this
static const float kMinDirConfidence = 0.30f;   // training dir_threshold
static const CFTimeInterval kInjectCooldown = 0.15;

// Synthesized swipe.
static const NSTimeInterval kSwipeDuration  = 0.12;
static const NSInteger      kSwipeSteps     = 15;
static const NSTimeInterval kSwipeEndDelay  = 0.016;    // one frame between last move and lift
static const CGFloat        kSwipeMagnitude = 0.35;     // fraction of min(window w, h)

// Layer B (sendEvent:) alone was empirically swallowed on the target device/iOS
// build, so touches are also delivered straight to the hit-tested view. If a
// build ever delivers both, every touch phase fires twice — set this to 0.
#define AI_DELIVER_DIRECTLY_TO_VIEW 1

typedef NS_ENUM(NSInteger, AISwipeDirection) {
    AISwipeDirectionUp = 0, AISwipeDirectionDown, AISwipeDirectionLeft, AISwipeDirectionRight,
};

typedef NS_ENUM(NSInteger, AIPlayerState) {
    AIPlayerStateOff = 0, AIPlayerStateStarting, AIPlayerStateOn, AIPlayerStateStopping,
};

typedef struct {
    float           detProbability;
    AISwipeDirection direction;
    float           dirConfidence;
} AIPrediction;

// =============================================================================
// MARK: - Logging and small helpers
// =============================================================================

static os_log_t AILogHandle(void) {
    static os_log_t log;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ log = os_log_create("com.aiplayer.tweak", "AIPlayer"); });
    return log;
}
#define AILog(fmt, ...) os_log(AILogHandle(), fmt, ##__VA_ARGS__)

static NSError *AIError(NSInteger code, NSString *message) {
    return [NSError errorWithDomain:@"AIPlayer" code:code
                           userInfo:@{ NSLocalizedDescriptionKey: message }];
}

static UIWindowScene *AIActiveWindowScene(void) {
    UIWindowScene *fallback = nil;
    for (UIScene *scene in [UIApplication sharedApplication].connectedScenes) {
        if (![scene isKindOfClass:[UIWindowScene class]]) continue;
        if (scene.activationState == UISceneActivationStateForegroundActive) return (UIWindowScene *)scene;
        if (!fallback) fallback = (UIWindowScene *)scene;
    }
    return fallback;
}

// The game's key window. The overlay window can never become key, so it is
// never returned here.
static UIWindow *AIKeyWindow(void) {
    for (UIScene *scene in [UIApplication sharedApplication].connectedScenes) {
        if (scene.activationState != UISceneActivationStateForegroundActive) continue;
        if (![scene isKindOfClass:[UIWindowScene class]]) continue;
        for (UIWindow *window in ((UIWindowScene *)scene).windows) {
            if (window.isKeyWindow) return window;
        }
    }
    // Apps without a scene manifest have no connected scenes.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return [UIApplication sharedApplication].keyWindow;
#pragma clang diagnostic pop
}

// MLMultiArray raw access. dataPointer is deprecated from iOS 15.4 but remains
// the simplest correct access for arrays this code allocates itself.
static inline void *AIArrayData(MLMultiArray *array) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return array.dataPointer;
#pragma clang diagnostic pop
}

// Copies the first `n` elements of a 1-D-like output (honouring the innermost
// stride, which Core ML may pad) into `dst`.
static BOOL AICopyVector(MLMultiArray *array, float *dst, NSInteger n) {
    if (!array || array.count < n) return NO;
    NSInteger stride = array.strides.lastObject.integerValue;
    if (stride < 1) stride = 1;
    switch (array.dataType) {
        case MLMultiArrayDataTypeFloat32: {
            const float *p = (const float *)AIArrayData(array);
            for (NSInteger i = 0; i < n; i++) dst[i] = p[i * stride];
            return YES;
        }
        case MLMultiArrayDataTypeDouble: {
            const double *p = (const double *)AIArrayData(array);
            for (NSInteger i = 0; i < n; i++) dst[i] = (float)p[i * stride];
            return YES;
        }
        default:   // e.g. Float16: slow but always correct (flat logical index)
            for (NSInteger i = 0; i < n; i++) dst[i] = array[i].floatValue;
            return YES;
    }
}

// Writes `n` floats into a freshly allocated (contiguous) input array, honouring
// whichever dtype the model declared (classic neuralnetwork models may use Double).
static void AIFillArray(MLMultiArray *array, const float *src, NSInteger n) {
    void *dst = AIArrayData(array);
    if (array.dataType == MLMultiArrayDataTypeDouble) {
        double *d = (double *)dst;
        for (NSInteger i = 0; i < n; i++) d[i] = src[i];
    } else {
        memcpy(dst, src, sizeof(float) * (size_t)n);
    }
}

static MLMultiArrayDataType AIDeclaredInputType(MLModel *model, NSString *name) {
    MLMultiArrayDataType type = model.modelDescription.inputDescriptionsByName[name].multiArrayConstraint.dataType;
    return type == MLMultiArrayDataTypeDouble ? MLMultiArrayDataTypeDouble : MLMultiArrayDataTypeFloat32;
}

static inline int AIFloorDiv(int a, int b) {
    int q = a / b, r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q--;
    return q;
}

// =============================================================================
// MARK: - Model location
// =============================================================================

static NSString *AIDylibDirectory(void) {
    NSString *loose = nil;
    uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; i++) {
        const char *name = _dyld_get_image_name(i);
        if (!name) continue;
        NSString *path = [NSString stringWithUTF8String:name];
        if ([path.lastPathComponent isEqualToString:@"AIPlayer.dylib"])
            return [path stringByDeletingLastPathComponent];
        if ([path.pathExtension.lowercaseString isEqualToString:@"dylib"] &&
            [path.lastPathComponent.lowercaseString containsString:@"aiplayer"])
            loose = [path stringByDeletingLastPathComponent];
    }
    return loose;
}

// Depth-limited search for a directory called `name` (fallback only).
static NSString *AIFindDirectoryNamed(NSString *name, NSString *root, NSInteger maxDepth) {
    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL isDir = NO;
    if (![fm fileExistsAtPath:root isDirectory:&isDir] || !isDir) return nil;

    NSDirectoryEnumerator<NSURL *> *walker =
        [fm enumeratorAtURL:[NSURL fileURLWithPath:root isDirectory:YES]
 includingPropertiesForKeys:@[NSURLIsDirectoryKey]
                    options:NSDirectoryEnumerationSkipsHiddenFiles
               errorHandler:^BOOL(NSURL *url, NSError *error) { return YES; }];
    NSInteger rootDepth = root.pathComponents.count;
    for (NSURL *url in walker) {
        if ((NSInteger)url.pathComponents.count - rootDepth > maxDepth) { [walker skipDescendants]; continue; }
        NSNumber *flag = nil;
        [url getResourceValue:&flag forKey:NSURLIsDirectoryKey error:nil];
        if (flag.boolValue && [url.lastPathComponent isEqualToString:name]) return url.path;
    }
    return nil;
}

// Directory that contains BOTH compiled models, or nil.
static NSURL *AIFindModelDirectory(void) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *dylibDir  = AIDylibDirectory();
    NSString *bundleDir = [NSBundle mainBundle].bundlePath;

    NSMutableArray<NSString *> *bases = [NSMutableArray array];
    if (dylibDir) {
        [bases addObject:dylibDir];
        [bases addObject:[dylibDir stringByAppendingPathComponent:@"AIPlayer.bundle"]];
    }
    if (bundleDir) {
        [bases addObject:bundleDir];
        [bases addObject:[bundleDir stringByAppendingPathComponent:@"Frameworks"]];
        [bases addObject:[[bundleDir stringByAppendingPathComponent:@"Frameworks"]
                          stringByAppendingPathComponent:@"AIPlayer.bundle"]];
    }

    BOOL (^hasBoth)(NSString *) = ^BOOL(NSString *dir) {
        return [fm fileExistsAtPath:[dir stringByAppendingPathComponent:kEncoderModelName]] &&
               [fm fileExistsAtPath:[dir stringByAppendingPathComponent:kHeadModelName]];
    };
    for (NSString *base in bases) {
        if (hasBoth(base)) return [NSURL fileURLWithPath:base isDirectory:YES];
    }
    if (bundleDir) {
        NSString *encoder = AIFindDirectoryNamed(kEncoderModelName, bundleDir, 4);
        NSString *parent = [encoder stringByDeletingLastPathComponent];
        if (parent && hasBoth(parent)) return [NSURL fileURLWithPath:parent isDirectory:YES];
    }
    AILog("Models not found. Inject %{public}@ and %{public}@ next to AIPlayer.dylib "
          "(searched dylib dir %{public}@, app bundle %{public}@).",
          kEncoderModelName, kHeadModelName, dylibDir ?: @"(nil)", bundleDir ?: @"(nil)");
    return nil;
}

// =============================================================================
// MARK: - Inference engine
//
// Not thread safe by itself; every public method takes an internal lock.
// =============================================================================

@interface AIInferenceEngine : NSObject
@property (nonatomic, readonly) BOOL isLoaded;
@property (nonatomic, readonly) double lastInferenceMs;
- (BOOL)loadModelsFromDirectory:(NSURL *)directory error:(NSError **)error;
- (void)resetSession;
// Real frame: preprocess + encode. Returns YES and fills `out` once the history
// is full (the first 20 ticks only warm the feature ring).
- (BOOL)ingestFrame:(CVPixelBufferRef)pixelBuffer prediction:(AIPrediction *)out;
// Source skipped a tick: re-feed the previous frame with a neutral diff so the
// feature ring stays aligned with the 24 Hz timeline.
- (void)ingestHeldFrame;
@end

static BOOL AIValidateModel(MLModel *model, NSString *label,
                            NSArray<NSString *> *inputs, NSArray<NSString *> *outputs,
                            NSError **error) {
    MLModelDescription *desc = model.modelDescription;
    for (NSString *name in inputs) {
        MLFeatureDescription *f = desc.inputDescriptionsByName[name];
        if (!f || !f.multiArrayConstraint) {
            if (error) *error = AIError(2, [NSString stringWithFormat:
                @"%@ is missing multi-array input '%@' (stale model from an older export?)", label, name]);
            return NO;
        }
    }
    for (NSString *name in outputs) {
        if (!desc.outputDescriptionsByName[name]) {
            if (error) *error = AIError(3, [NSString stringWithFormat:
                @"%@ is missing output '%@' (stale model from an older export?)", label, name]);
            return NO;
        }
    }
    return YES;
}

@implementation AIInferenceEngine {
    NSLock *_lock;

    MLModel *_encoder, *_head;
    MLMultiArray *_encIn, *_headFast, *_headSlow;          // reused every tick
    id<MLFeatureProvider> _encProvider, _headProvider;

    CIContext *_ci;
    CGColorSpaceRef _rgb;

    float   _encBuf[2 * AIPixels];                         // [gray | diff], staged as float32
    float   _fastBuf[AIFastLen * AIFeatDim];
    float   _slowBuf[AISlowLen * AIFeatDim];

    uint8_t _rgba[AIPixels * 4];
    uint8_t _gray[AIPixels];
    uint8_t _prevGray[AIPixels];
    BOOL    _hasPrev;
    uint8_t _diffLUT[511];                                 // index = gray delta + 255

    float     _ring[AIHistory][AIFeatDim];                 // per-tick encoder features
    NSInteger _tick;                                       // ticks pushed this session

    double _lastInferenceMs;
}

- (instancetype)init {
    if ((self = [super init])) {
        _lock = [NSLock new];
        _rgb = CGColorSpaceCreateDeviceRGB();
        _ci = [CIContext contextWithOptions:@{ kCIContextWorkingColorSpace: (__bridge id)_rgb }];
        for (int d = -255; d <= 255; d++) {
            int s = AIFloorDiv(d * 127, 255) + 127;
            _diffLUT[d + 255] = (uint8_t)(s < 0 ? 0 : (s > 255 ? 255 : s));
        }
    }
    return self;
}

- (void)dealloc {
    if (_rgb) CGColorSpaceRelease(_rgb);
}

- (BOOL)isLoaded { return _encoder != nil && _head != nil; }
- (double)lastInferenceMs { return _lastInferenceMs; }

// ── loading ─────────────────────────────────────────────────────────────────

- (BOOL)loadModelsFromDirectory:(NSURL *)directory error:(NSError **)error {
    [_lock lock];
    BOOL ok = [self _loadLocked:directory error:error];
    [_lock unlock];
    return ok;
}

- (BOOL)_loadLocked:(NSURL *)directory error:(NSError **)error {
    MLModelConfiguration *config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsAll;

    MLModel *encoder = [MLModel modelWithContentsOfURL:
        [directory URLByAppendingPathComponent:kEncoderModelName isDirectory:YES]
                                         configuration:config error:error];
    if (!encoder) return NO;
    MLModel *head = [MLModel modelWithContentsOfURL:
        [directory URLByAppendingPathComponent:kHeadModelName isDirectory:YES]
                                       configuration:config error:error];
    if (!head) return NO;

    if (!AIValidateModel(encoder, @"SwipeEncoder", @[@"frame"], @[@"feat"], error)) return NO;
    if (!AIValidateModel(head, @"SwipeHead", @[@"fast_feats", @"slow_feats"],
                         @[@"det_logit", @"dir_logits"], error)) return NO;

    MLMultiArray *encIn = [[MLMultiArray alloc] initWithShape:@[@1, @2, @(AIImgSize), @(AIImgSize)]
                                                     dataType:AIDeclaredInputType(encoder, @"frame") error:error];
    MLMultiArray *fast  = [[MLMultiArray alloc] initWithShape:@[@1, @(AIFastLen), @(AIFeatDim)]
                                                     dataType:AIDeclaredInputType(head, @"fast_feats") error:error];
    MLMultiArray *slow  = [[MLMultiArray alloc] initWithShape:@[@1, @(AISlowLen), @(AIFeatDim)]
                                                     dataType:AIDeclaredInputType(head, @"slow_feats") error:error];
    if (!encIn || !fast || !slow) return NO;

    id<MLFeatureProvider> encProvider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"frame": [MLFeatureValue featureValueWithMultiArray:encIn] } error:error];
    id<MLFeatureProvider> headProvider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"fast_feats": [MLFeatureValue featureValueWithMultiArray:fast],
                              @"slow_feats": [MLFeatureValue featureValueWithMultiArray:slow] } error:error];
    if (!encProvider || !headProvider) return NO;

    _encoder = encoder;  _head = head;
    _encIn = encIn;      _headFast = fast;  _headSlow = slow;
    _encProvider = encProvider;  _headProvider = headProvider;
    [self _resetLocked];
    AILog("Models loaded from %{public}@", directory.path);
    return YES;
}

// ── session ─────────────────────────────────────────────────────────────────

- (void)resetSession {
    [_lock lock];
    [self _resetLocked];
    [_lock unlock];
}

- (void)_resetLocked {
    _tick = 0;
    _hasPrev = NO;
    memset(_ring, 0, sizeof(_ring));
    memset(_prevGray, 0, sizeof(_prevGray));
}

// ── per tick ────────────────────────────────────────────────────────────────

- (BOOL)ingestFrame:(CVPixelBufferRef)pixelBuffer prediction:(AIPrediction *)out {
    [_lock lock];
    BOOL produced = NO;
    if (self.isLoaded && [self _renderLocked:pixelBuffer]) {
        CFTimeInterval t0 = CACurrentMediaTime();
        [self _packLocked];
        if ([self _encodeLocked]) produced = [self _predictLocked:out];
        _lastInferenceMs = (CACurrentMediaTime() - t0) * 1000.0;
    }
    [_lock unlock];
    return produced;
}

- (void)ingestHeldFrame {
    [_lock lock];
    if (self.isLoaded && _hasPrev) {
        // Gray channel already holds the previous frame; identical frames
        // have a neutral diff.
        float *diff = _encBuf + AIPixels;
        for (NSInteger i = 0; i < AIPixels; i++) diff[i] = 127.0f;
        [self _encodeLocked];
    }
    [_lock unlock];
}

// Antialiased square-stretch resize into _rgba.
- (BOOL)_renderLocked:(CVPixelBufferRef)pixelBuffer {
    CIImage *image = [CIImage imageWithCVPixelBuffer:pixelBuffer];
    CGRect extent = image.extent;
    if (CGRectIsEmpty(extent)) return NO;
    if (extent.origin.x != 0 || extent.origin.y != 0)
        image = [image imageByApplyingTransform:
                 CGAffineTransformMakeTranslation(-extent.origin.x, -extent.origin.y)];

    // CILanczosScaleTransform: output height = scale * h, width = scale * aspect * w.
    double scale  = (double)AIImgSize / extent.size.height;
    double aspect = (double)extent.size.height / extent.size.width;
    CIImage *scaled = [image imageByApplyingFilter:@"CILanczosScaleTransform"
                               withInputParameters:@{ kCIInputScaleKey:       @(scale),
                                                      kCIInputAspectRatioKey: @(aspect) }];
    [_ci render:scaled
       toBitmap:_rgba
       rowBytes:AIImgSize * 4
         bounds:CGRectMake(0, 0, AIImgSize, AIImgSize)
         format:kCIFormatRGBA8
     colorSpace:_rgb];
    return YES;
}

// _rgba -> gray + diff staged in _encBuf; updates _prevGray.
- (void)_packLocked {
    float *grayOut = _encBuf;
    float *diffOut = _encBuf + AIPixels;
    const uint8_t *px = _rgba;
    for (NSInteger i = 0; i < AIPixels; i++, px += 4) {
        float l = 0.2989f * px[0] + 0.5870f * px[1] + 0.1140f * px[2];
        int g = (int)roundf(l);
        g = g < 0 ? 0 : (g > 255 ? 255 : g);
        int prev = _hasPrev ? _prevGray[i] : g;
        _gray[i] = (uint8_t)g;
        grayOut[i] = (float)g;
        diffOut[i] = (float)_diffLUT[g - prev + 255];
    }
    memcpy(_prevGray, _gray, sizeof(_gray));
    _hasPrev = YES;
}

// Runs the CNN on _encIn and appends the feature to the ring.
- (BOOL)_encodeLocked {
    AIFillArray(_encIn, _encBuf, 2 * AIPixels);
    NSError *error = nil;
    id<MLFeatureProvider> result = [_encoder predictionFromFeatures:_encProvider error:&error];
    MLMultiArray *feat = [result featureValueForName:@"feat"].multiArrayValue;
    if (!AICopyVector(feat, _ring[_tick % AIHistory], AIFeatDim)) {
        AILog("Encoder failed: %{public}@", error.localizedDescription ?: @"bad output");
        return NO;
    }
    _tick++;
    return YES;
}

// Windowed head over cached features; identical to training's forward().
- (BOOL)_predictLocked:(AIPrediction *)out {
    if (_tick < AIHistory) return NO;                      // still warming up

    float *fast = _fastBuf;
    float *slow = _slowBuf;
    for (NSInteger k = 0; k < AIFastLen; k++) {
        NSInteger age = AIFastLen - 1 - k;                 // oldest first
        memcpy(fast + k * AIFeatDim, _ring[(_tick - 1 - age) % AIHistory], sizeof(float) * AIFeatDim);
    }
    for (NSInteger j = 0; j < AISlowLen; j++) {
        memcpy(slow + j * AIFeatDim, _ring[(_tick - 1 - kSlowOffsets[j]) % AIHistory],
               sizeof(float) * AIFeatDim);
    }

    AIFillArray(_headFast, _fastBuf, AIFastLen * AIFeatDim);
    AIFillArray(_headSlow, _slowBuf, AISlowLen * AIFeatDim);
    NSError *error = nil;
    id<MLFeatureProvider> result = [_head predictionFromFeatures:_headProvider error:&error];
    float detLogit = 0, dirLogits[4] = { 0, 0, 0, 0 };
    if (!result ||
        !AICopyVector([result featureValueForName:@"det_logit"].multiArrayValue, &detLogit, 1) ||
        !AICopyVector([result featureValueForName:@"dir_logits"].multiArrayValue, dirLogits, 4)) {
        AILog("Head failed: %{public}@", error.localizedDescription ?: @"bad output");
        return NO;
    }

    float maxLogit = fmaxf(fmaxf(dirLogits[0], dirLogits[1]), fmaxf(dirLogits[2], dirLogits[3]));
    float e[4], sum = 0;
    for (int i = 0; i < 4; i++) { e[i] = expf(dirLogits[i] - maxLogit); sum += e[i]; }
    int best = 0;
    for (int i = 1; i < 4; i++) if (e[i] > e[best]) best = i;

    out->detProbability = 1.0f / (1.0f + expf(-detLogit));
    out->direction      = (AISwipeDirection)best;
    out->dirConfidence  = e[best] / sum;
    return YES;
}

@end

// =============================================================================
// MARK: - Touch synthesis
//
// Builds a UITouch + UITouchesEvent by hand (private API, validated on-device)
// and delivers it through the window and, because sendEvent: alone was
// swallowed on the target build, directly to the hit-tested view.
// =============================================================================

// UIKit's private -_initWithEvent:touches: reads a GSEvent-shaped struct.
// Field offsets match the decompiled initializer: flags 0x08, type 0x0C,
// x1/y1/x2/y2 0x14-0x23, sizeX/sizeY 0x68/0x6C, x3/y3 0x70/0x74.
@interface AIEventProxy : NSObject {
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
@implementation AIEventProxy
@end

// Safety net: if writing UITouch's _window/_view ivars ever fails on some iOS
// build, -window/-view fall back to an associated object.
static const void *kFallbackWindowKey = &kFallbackWindowKey;
static const void *kFallbackViewKey   = &kFallbackViewKey;

@interface UITouch (AIPlayerFallback)
- (UIWindow *)aip_window;
- (UIView *)aip_view;
@end

@implementation UITouch (AIPlayerFallback)
// After the swizzle these names run the ORIGINAL implementations.
- (UIWindow *)aip_window {
    UIWindow *window = [self aip_window];
    return window ?: objc_getAssociatedObject(self, kFallbackWindowKey);
}
- (UIView *)aip_view {
    UIView *view = [self aip_view];
    return view ?: objc_getAssociatedObject(self, kFallbackViewKey);
}
@end

static void AIInstallTouchFallbackSwizzle(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        Class cls = [UITouch class];
        Method w0 = class_getInstanceMethod(cls, @selector(window));
        Method w1 = class_getInstanceMethod(cls, @selector(aip_window));
        Method v0 = class_getInstanceMethod(cls, @selector(view));
        Method v1 = class_getInstanceMethod(cls, @selector(aip_view));
        if (w0 && w1 && v0 && v1) {
            method_exchangeImplementations(w0, w1);
            method_exchangeImplementations(v0, v1);
        } else {
            AILog("UITouch fallback swizzle unavailable (window/view selectors missing)");
        }
    });
}

// Raw ivar access for scalar fields (UITouch exposes no setters for these).
static void *AIIvarPointer(id object, const char *name) {
    Ivar ivar = class_getInstanceVariable(object_getClass(object), name);
    return ivar ? (uint8_t *)(__bridge void *)object + ivar_getOffset(ivar) : NULL;
}
static void AIWriteIvar(id object, const char *name, const void *value, size_t size) {
    void *p = AIIvarPointer(object, name);
    if (p) memcpy(p, value, size);
}
static BOOL AIReadIvar(id object, const char *name, void *value, size_t size) {
    void *p = AIIvarPointer(object, name);
    if (!p) return NO;
    memcpy(value, p, size);
    return YES;
}

// Sets the touch's window/view: shadow associated object first, then the ivar,
// then private/public setters, then KVC.
static void AISetTouchObject(UITouch *touch, id value, NSString *key, NSString *capitalized,
                             const void *assocKey) {
    objc_setAssociatedObject(touch, assocKey, value, OBJC_ASSOCIATION_RETAIN_NONATOMIC);

    Ivar ivar = class_getInstanceVariable([UITouch class],
                                          [[@"_" stringByAppendingString:key] UTF8String]);
    if (ivar) { object_setIvar(touch, ivar, value); return; }

    for (NSString *format in @[@"_set%@:", @"set%@:"]) {
        SEL setter = NSSelectorFromString([NSString stringWithFormat:format, capitalized]);
        if ([touch respondsToSelector:setter]) {
            ((void (*)(id, SEL, id))objc_msgSend)(touch, setter, value);
            return;
        }
    }
    @try { [touch setValue:value forKey:key]; } @catch (__unused NSException *e) {}
}

@interface AITouchSynthesizer : NSObject
// Starts (Began), advances (Moved) or finishes (Ended/Cancelled) a single-finger
// touch. Returns the live touch, or nil once it has ended.
- (nullable UITouch *)dispatchPhase:(UITouchPhase)phase atPoint:(CGPoint)point
                              touch:(nullable UITouch *)touch;
@end

@implementation AITouchSynthesizer

- (UITouch *)dispatchPhase:(UITouchPhase)phase atPoint:(CGPoint)point touch:(UITouch *)touch {
    UIWindow *window = AIKeyWindow();
    if (!window) return touch;

    CGPoint previous = point;
    if (!touch || phase == UITouchPhaseBegan) {
        touch = [self _newTouchInWindow:window atPoint:point];
    } else {
        AIReadIvar(touch, "_locationInWindow", &previous, sizeof previous);
        if (fabs(point.x - previous.x) > 2.0 || fabs(point.y - previous.y) > 2.0) {
            uint16_t *flags = (uint16_t *)AIIvarPointer(touch, "_touchFlags");
            if (flags) *flags &= 0xFFFD;                   // moved: clear "stationary" bit
        }
    }

    NSTimeInterval now = [[NSProcessInfo processInfo] systemUptime];
    NSInteger phaseValue = phase;
    AIWriteIvar(touch, "_locationInWindow", &point, sizeof point);
    AIWriteIvar(touch, "_previousLocationInWindow", &previous, sizeof previous);
    AIWriteIvar(touch, "_phase", &phaseValue, sizeof phaseValue);
    AIWriteIvar(touch, "_timestamp", &now, sizeof now);

    UIEvent *event = [self _eventForTouch:touch point:point previous:previous phase:phase];
    UIWindow *target = touch.window ?: window;
    if (event) [target sendEvent:event];

#if AI_DELIVER_DIRECTLY_TO_VIEW
    [self _deliverPhase:phase toView:touch.view touch:touch event:event];
#endif

    return (phase == UITouchPhaseEnded || phase == UITouchPhaseCancelled) ? nil : touch;
}

- (UITouch *)_newTouchInWindow:(UIWindow *)window atPoint:(CGPoint)point {
    UITouch *touch = [[UITouch alloc] init];
    UIView *view = [window hitTest:point withEvent:nil] ?: window;

    AISetTouchObject(touch, window, @"window", @"Window", kFallbackWindowKey);
    AISetTouchObject(touch, view,   @"view",   @"View",   kFallbackViewKey);

    NSInteger tapCount = 1;
    AIWriteIvar(touch, "_tapCount", &tapCount, sizeof tapCount);
    uint16_t *flags = (uint16_t *)AIIvarPointer(touch, "_touchFlags");
    if (flags) *flags |= 0x3;                              // a new touch always has bits 0x1|0x2
    return touch;
}

- (UIEvent *)_eventForTouch:(UITouch *)touch point:(CGPoint)point previous:(CGPoint)previous
                      phase:(UITouchPhase)phase {
    AIEventProxy *proxy = [[AIEventProxy alloc] init];
    proxy->x1 = point.x;     proxy->y1 = point.y;
    proxy->x2 = previous.x;  proxy->y2 = previous.y;
    proxy->x3 = point.x;     proxy->y3 = point.y;
    proxy->sizeX = 1.0f;     proxy->sizeY = 1.0f;
    // Began/Moved/Stationary use 0x3010180; Ended/Cancelled keep the 0x1010180 default.
    proxy->flags = (phase == UITouchPhaseEnded || phase == UITouchPhaseCancelled) ? 0x1010180 : 0x3010180;
    proxy->type  = 3001;

    // Same variable on both sides keeps ARC's retain/release balanced across the
    // init-family call made through objc_msgSend.
    Class eventClass = NSClassFromString(@"UITouchesEvent");
    UIEvent *event = [eventClass alloc];
    SEL initializer = NSSelectorFromString(@"_initWithEvent:touches:");
    if (event && [event respondsToSelector:initializer]) {
        event = ((id (*)(id, SEL, id, id))objc_msgSend)(event, initializer, proxy,
                                                        [NSSet setWithObject:touch]);
    } else {
        event = [[UIEvent alloc] init];
    }
    if (event) {
        NSTimeInterval stamp = [[NSProcessInfo processInfo] systemUptime];
        AIWriteIvar(event, "_timestamp", &stamp, sizeof stamp);
    }
    return event;
}

- (void)_deliverPhase:(UITouchPhase)phase toView:(UIView *)view touch:(UITouch *)touch
                event:(UIEvent *)event {
    if (!view) return;
    NSSet *touches = [NSSet setWithObject:touch];
    switch (phase) {
        case UITouchPhaseBegan:     [view touchesBegan:touches withEvent:event]; break;
        case UITouchPhaseMoved:     [view touchesMoved:touches withEvent:event]; break;
        case UITouchPhaseEnded:     [view touchesEnded:touches withEvent:event]; break;
        case UITouchPhaseCancelled: [view touchesCancelled:touches withEvent:event]; break;
        default: break;                                    // Stationary: nothing to deliver
    }
}

@end

// =============================================================================
// MARK: - Swipe injector (main thread)
// =============================================================================

@interface AISwipeInjector : NSObject
+ (instancetype)shared;
- (void)swipeInDirection:(AISwipeDirection)direction;   // any thread
- (void)cancel;                                         // any thread
@end

@implementation AISwipeInjector {
    AITouchSynthesizer *_synth;
    UITouch *_touch;
    CGPoint _lastPoint;
    NSUInteger _generation;          // invalidates the pending steps of a cancelled swipe
}

+ (instancetype)shared {
    static AISwipeInjector *instance;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ instance = [AISwipeInjector new]; });
    return instance;
}

- (instancetype)init {
    if ((self = [super init])) _synth = [AITouchSynthesizer new];
    return self;
}

- (void)swipeInDirection:(AISwipeDirection)direction {
    if (![NSThread isMainThread]) {
        dispatch_async(dispatch_get_main_queue(), ^{ [self swipeInDirection:direction]; });
        return;
    }
    UIWindow *window = AIKeyWindow();
    if (!window) return;

    [self cancel];                                          // never overlap two swipes

    CGRect bounds = window.bounds;
    CGPoint start = CGPointMake(CGRectGetMidX(bounds), CGRectGetMidY(bounds));
    CGFloat reach = MIN(bounds.size.width, bounds.size.height) * kSwipeMagnitude;
    CGPoint end = start;
    switch (direction) {
        case AISwipeDirectionUp:    end.y -= reach; break;
        case AISwipeDirectionDown:  end.y += reach; break;
        case AISwipeDirectionLeft:  end.x -= reach; break;
        case AISwipeDirectionRight: end.x += reach; break;
    }

    NSUInteger generation = _generation;
    [self _dispatchPhase:UITouchPhaseBegan at:start];

    NSTimeInterval stepDelay = kSwipeDuration / kSwipeSteps;
    for (NSInteger i = 1; i <= kSwipeSteps; i++) {
        CGFloat t = (CGFloat)i / (CGFloat)kSwipeSteps;
        CGPoint point = CGPointMake(start.x + (end.x - start.x) * t, start.y + (end.y - start.y) * t);
        [self _schedule:i * stepDelay generation:generation phase:UITouchPhaseMoved at:point];
    }
    [self _schedule:kSwipeDuration + kSwipeEndDelay generation:generation phase:UITouchPhaseEnded at:end];
}

- (void)cancel {
    if (![NSThread isMainThread]) {
        dispatch_async(dispatch_get_main_queue(), ^{ [self cancel]; });
        return;
    }
    _generation++;
    if (_touch) [self _dispatchPhase:UITouchPhaseCancelled at:_lastPoint];
}

- (void)_schedule:(NSTimeInterval)delay generation:(NSUInteger)generation
            phase:(UITouchPhase)phase at:(CGPoint)point {
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(delay * NSEC_PER_SEC)),
                   dispatch_get_main_queue(), ^{
        if (generation != self->_generation) return;
        [self _dispatchPhase:phase at:point];
    });
}

- (void)_dispatchPhase:(UITouchPhase)phase at:(CGPoint)point {
    if (!_touch && phase != UITouchPhaseBegan) return;      // swipe never started (no window)
    _lastPoint = point;
    _touch = [_synth dispatchPhase:phase atPoint:point touch:_touch];
}

@end

// =============================================================================
// MARK: - Player controller: capture, clock, detection gate
// =============================================================================

@interface AIPlayerController : NSObject
@property (nonatomic, readonly) AIPlayerState state;                       // main thread
@property (nonatomic, copy, nullable) void (^onStateChange)(AIPlayerState state);   // main thread
+ (instancetype)shared;
- (void)toggle;                                                            // main thread
@end

@implementation AIPlayerController {
    AIInferenceEngine *_engine;
    dispatch_queue_t _setupQueue;
    AIPlayerState _state;
    atomic_bool _active;                 // read by the capture queue

    // Touched only by the capture queue (reset before capture starts).
    BOOL _hasClock;
    CFTimeInterval _nextTick;
    BOOL _armed;
    CFTimeInterval _lastFire;
    CFTimeInterval _statsStart;
    NSInteger _statFrames, _statTicks, _statPreds, _statFires;
    double _statDetSum;
    float _statDetMax;
}

+ (instancetype)shared {
    static AIPlayerController *instance;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ instance = [AIPlayerController new]; });
    return instance;
}

- (instancetype)init {
    if ((self = [super init])) {
        _engine = [AIInferenceEngine new];
        _setupQueue = dispatch_queue_create("com.aiplayer.setup", DISPATCH_QUEUE_SERIAL);
        atomic_init(&_active, false);

        // Load models in the background so the first tap is instant.
        dispatch_async(_setupQueue, ^{ [self _ensureModelsLoaded:NULL]; });

        [[NSNotificationCenter defaultCenter]
            addObserverForName:UIApplicationDidEnterBackgroundNotification object:nil
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(NSNotification *note) {
            if (self->_state == AIPlayerStateOn) [self stop];
        }];
    }
    return self;
}

- (AIPlayerState)state { return _state; }

- (void)_setState:(AIPlayerState)state {
    _state = state;
    if (self.onStateChange) self.onStateChange(state);
}

- (void)toggle {
    if (_state == AIPlayerStateOff) [self start];
    else if (_state == AIPlayerStateOn) [self stop];
}

// ── lifecycle ───────────────────────────────────────────────────────────────

- (BOOL)_ensureModelsLoaded:(NSError **)error {          // setup queue only
    if (_engine.isLoaded) return YES;
    NSURL *directory = AIFindModelDirectory();
    if (!directory) {
        if (error) *error = AIError(1, @"SwipeEncoder/SwipeHead .mlmodelc not found");
        return NO;
    }
    NSError *loadError = nil;
    if (![_engine loadModelsFromDirectory:directory error:&loadError]) {
        AILog("Model load failed: %{public}@", loadError.localizedDescription);
        if (error) *error = loadError;
        return NO;
    }
    return YES;
}

- (void)start {
    [self _setState:AIPlayerStateStarting];
    dispatch_async(_setupQueue, ^{
        NSError *error = nil;
        if (![self _ensureModelsLoaded:&error]) {
            AILog("Cannot start: %{public}@", error.localizedDescription);
            dispatch_async(dispatch_get_main_queue(), ^{ [self _setState:AIPlayerStateOff]; });
            return;
        }
        [self->_engine resetSession];
        [self _resetRuntimeState];
        dispatch_async(dispatch_get_main_queue(), ^{ [self _beginCapture]; });
    });
}

- (void)_beginCapture {
    RPScreenRecorder *recorder = [RPScreenRecorder sharedRecorder];
    if (!recorder.isAvailable) {
        AILog("ReplayKit recorder unavailable");
        [self _setState:AIPlayerStateOff];
        return;
    }
    atomic_store(&_active, true);
    [recorder startCaptureWithHandler:^(CMSampleBufferRef buffer, RPSampleBufferType type, NSError *error) {
        if (type != RPSampleBufferTypeVideo || error || !atomic_load(&self->_active)) return;
        [self _handleSampleBuffer:buffer];
    } completionHandler:^(NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (error) {
                atomic_store(&self->_active, false);
                AILog("Capture failed: %{public}@", error.localizedDescription);
                [self _setState:AIPlayerStateOff];
            } else {
                AILog("Capture started");
                [self _setState:AIPlayerStateOn];
            }
        });
    }];
}

- (void)stop {
    [self _setState:AIPlayerStateStopping];
    atomic_store(&_active, false);
    [[AISwipeInjector shared] cancel];
    [[RPScreenRecorder sharedRecorder] stopCaptureWithHandler:^(NSError *error) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (error) AILog("Capture stop: %{public}@", error.localizedDescription);
            [self _setState:AIPlayerStateOff];
        });
    }];
}

- (void)_resetRuntimeState {
    _hasClock = NO;
    _nextTick = 0;
    _armed = NO;          // must observe a low probability before the first fire
    _lastFire = 0;
    _statsStart = 0;
    _statFrames = _statTicks = _statPreds = _statFires = 0;
    _statDetSum = 0;
    _statDetMax = 0;
}

// ── capture queue ───────────────────────────────────────────────────────────

// Number of 24 Hz ticks the frame stamped `time` accounts for (0 = drop it).
- (NSInteger)_ticksDueAt:(CFTimeInterval)time {
    if (!_hasClock) {
        _hasClock = YES;
        _nextTick = time + kTickInterval;
        return 1;
    }
    CFTimeInterval ahead = time + kTickTolerance - _nextTick;
    if (ahead < 0) return 0;
    NSInteger due = 1 + (NSInteger)floor(ahead / kTickInterval);
    if (due > kMaxCatchUpTicks) {                          // stall or discontinuity: resync
        _nextTick = time + kTickInterval;
        return 1;
    }
    _nextTick += due * kTickInterval;
    return due;
}

- (void)_handleSampleBuffer:(CMSampleBufferRef)buffer {
    _statFrames++;
    CVImageBufferRef pixels = CMSampleBufferGetImageBuffer(buffer);
    if (!pixels) return;

    CFTimeInterval time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer));
    if (!isfinite(time)) time = CACurrentMediaTime();

    NSInteger due = [self _ticksDueAt:time];
    if (due == 0) return;

    for (NSInteger i = 0; i < due - 1; i++) [_engine ingestHeldFrame];   // source skipped ticks
    _statTicks += due;

    AIPrediction prediction;
    if ([_engine ingestFrame:pixels prediction:&prediction]) [self _handlePrediction:prediction];
    [self _logStatsIfDue];
}

- (void)_handlePrediction:(AIPrediction)p {
    _statPreds++;
    _statDetSum += p.detProbability;
    if (p.detProbability > _statDetMax) _statDetMax = p.detProbability;

    if (p.detProbability < kReleaseThreshold) _armed = YES;
    if (!_armed || p.detProbability < kFireThreshold || p.dirConfidence < kMinDirConfidence) return;

    CFTimeInterval now = CACurrentMediaTime();
    if (now - _lastFire < kInjectCooldown) return;
    _lastFire = now;
    _armed = NO;                                           // one fire per detection burst
    _statFires++;
    [[AISwipeInjector shared] swipeInDirection:p.direction];
}

- (void)_logStatsIfDue {
    CFTimeInterval now = CACurrentMediaTime();
    if (_statsStart == 0) { _statsStart = now; return; }
    if (now - _statsStart < 1.0) return;
    AILog("in=%ld ticks=%ld preds=%ld det_mean=%.2f det_max=%.2f fires=%ld infer=%.1fms",
          (long)_statFrames, (long)_statTicks, (long)_statPreds,
          _statPreds ? _statDetSum / _statPreds : 0.0, _statDetMax,
          (long)_statFires, _engine.lastInferenceMs);
    _statsStart = now;
    _statFrames = _statTicks = _statPreds = _statFires = 0;
    _statDetSum = 0;
    _statDetMax = 0;
}

@end

// =============================================================================
// MARK: - Overlay UI: one draggable AI ON/OFF button
// =============================================================================

// Both overrides return nil for "hit nothing" so every touch outside the button
// falls through to the game, and the window never takes key status.
@interface AIPassthroughView : UIView @end
@implementation AIPassthroughView
- (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event {
    UIView *hit = [super hitTest:point withEvent:event];
    return hit == self ? nil : hit;
}
@end

@interface AIOverlayWindow : UIWindow @end
@implementation AIOverlayWindow
- (UIView *)hitTest:(CGPoint)point withEvent:(UIEvent *)event {
    UIView *hit = [super hitTest:point withEvent:event];
    return hit == self ? nil : hit;
}
- (BOOL)canBecomeKeyWindow { return NO; }
@end

@interface AIOverlayViewController : UIViewController @end

@implementation AIOverlayViewController {
    UIButton *_button;
    BOOL _placed;
}

- (void)loadView {
    AIPassthroughView *view = [AIPassthroughView new];
    view.backgroundColor = [UIColor clearColor];
    self.view = view;
}

- (void)viewDidLoad {
    [super viewDidLoad];

    _button = [UIButton buttonWithType:UIButtonTypeCustom];
    _button.bounds = CGRectMake(0, 0, 130, 44);
    _button.layer.cornerRadius = 12;
    _button.clipsToBounds = YES;
    _button.titleLabel.font = [UIFont monospacedSystemFontOfSize:12 weight:UIFontWeightBold];
    [_button setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    [_button addTarget:self action:@selector(_tapped) forControlEvents:UIControlEventTouchUpInside];
    [_button addGestureRecognizer:[[UIPanGestureRecognizer alloc] initWithTarget:self
                                                                          action:@selector(_dragged:)]];
    [self.view addSubview:_button];

    AIPlayerController *controller = [AIPlayerController shared];
    __weak typeof(self) weakSelf = self;
    controller.onStateChange = ^(AIPlayerState state) { [weakSelf _render:state]; };
    [self _render:controller.state];
}

- (void)viewDidLayoutSubviews {
    [super viewDidLayoutSubviews];
    CGRect bounds = self.view.bounds;
    if (_placed || CGRectGetWidth(bounds) <= 0) return;
    _placed = YES;
    _button.center = CGPointMake(CGRectGetWidth(bounds) - 80,
                                 MAX(self.view.safeAreaInsets.top, 20) + 60);
}

- (void)_render:(AIPlayerState)state {
    NSString *title;
    UIColor *color;
    switch (state) {
        case AIPlayerStateOn:
            title = @"■ AI: ON";   color = [UIColor colorWithRed:0.20 green:0.70 blue:0.35 alpha:0.95]; break;
        case AIPlayerStateStarting:
            title = @"… STARTING"; color = [UIColor colorWithRed:0.85 green:0.65 blue:0.15 alpha:0.95]; break;
        case AIPlayerStateStopping:
            title = @"… STOPPING"; color = [UIColor colorWithRed:0.85 green:0.65 blue:0.15 alpha:0.95]; break;
        default:
            title = @"▶ AI: OFF";  color = [UIColor colorWithWhite:0.15 alpha:0.92]; break;
    }
    [_button setTitle:title forState:UIControlStateNormal];
    _button.backgroundColor = color;
}

- (void)_tapped { [[AIPlayerController shared] toggle]; }

- (void)_dragged:(UIPanGestureRecognizer *)pan {
    CGPoint delta = [pan translationInView:self.view];
    CGRect bounds = self.view.bounds;
    CGSize half = CGSizeMake(_button.bounds.size.width / 2, _button.bounds.size.height / 2);
    CGPoint center = CGPointMake(_button.center.x + delta.x, _button.center.y + delta.y);
    center.x = MIN(MAX(center.x, half.width),  bounds.size.width  - half.width);
    center.y = MIN(MAX(center.y, half.height), bounds.size.height - half.height);
    _button.center = center;
    [pan setTranslation:CGPointZero inView:self.view];
}

@end

// =============================================================================
// MARK: - Entry point
// =============================================================================

static UIWindow *sOverlayWindow;

// Scene-based apps need the window attached to a scene, which may connect a
// moment after the app becomes active; legacy apps never have one.
static void AIInstallOverlay(int sceneRetriesLeft) {
    if (sOverlayWindow) return;
    UIWindowScene *scene = AIActiveWindowScene();
    if (!scene && sceneRetriesLeft > 0) {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.25 * NSEC_PER_SEC)),
                       dispatch_get_main_queue(), ^{ AIInstallOverlay(sceneRetriesLeft - 1); });
        return;
    }
    AIOverlayWindow *window = scene
        ? [[AIOverlayWindow alloc] initWithWindowScene:scene]
        : [[AIOverlayWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    window.windowLevel = UIWindowLevelAlert + 1;
    window.backgroundColor = [UIColor clearColor];
    window.rootViewController = [AIOverlayViewController new];
    window.hidden = NO;
    sOverlayWindow = window;
}

__attribute__((constructor))
static void AIPlayerInit(void) {
    AIInstallTouchFallbackSwizzle();
    dispatch_async(dispatch_get_main_queue(), ^{
        if ([UIApplication sharedApplication].applicationState == UIApplicationStateActive) {
            AIInstallOverlay(3);
            return;
        }
        __block id token = [[NSNotificationCenter defaultCenter]
            addObserverForName:UIApplicationDidBecomeActiveNotification object:nil
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(NSNotification *note) {
            [[NSNotificationCenter defaultCenter] removeObserver:token];
            AIInstallOverlay(3);
        }];
    });
}
