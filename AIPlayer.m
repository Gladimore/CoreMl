// =============================================================================
// AIPlayer.m — sideloaded screen-capture + Core ML swipe player.
//
//   ReplayKit frame
//     -> fixed-rate tick clock (tick_hz)
//     -> exact training preprocessing (AIPreprocess.c): antialiased 128x128
//        resize + torchvision grayscale, NO colour management
//     -> model ("simple": one frame; "causal": cached CNN features + GRU head)
//     -> detection gate (fire once per burst) -> synthesized swipe
//
// Everything model-specific comes from AIPlayerConfig.json, written by
// convert_to_coreml.py next to the compiled models. The dylib therefore cannot
// run a model with the wrong pipeline, and thresholds can be tuned without a
// rebuild.
//
// Deployment: non-jailbroken, injected with Sideloadly. Put AIPlayerConfig.json
// and the .mlmodelc folder(s) it lists next to AIPlayer.dylib.
// =============================================================================

#import <UIKit/UIKit.h>
#import <ReplayKit/ReplayKit.h>
#import <CoreImage/CoreImage.h>
#import <CoreML/CoreML.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <ImageIO/ImageIO.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <mach-o/dyld.h>
#include <math.h>
#include <os/log.h>
#include <stdatomic.h>

#include "AIPreprocess.h"

static NSString *const kConfigFileName = @"AIPlayerConfig.json";

// Synthesized swipe geometry (game-specific, not model-specific).
static const NSTimeInterval kSwipeDuration  = 0.12;
static const NSInteger      kSwipeSteps     = 15;
static const NSTimeInterval kSwipeEndDelay  = 0.016;   // one frame between last move and lift
static const CGFloat        kSwipeMagnitude = 0.35;    // fraction of min(window w, h)

// Clock: accept a frame slightly early (60 Hz jitter); longer gaps are a discontinuity.
static const CFTimeInterval kTickTolerance   = 0.008;
static const NSInteger      kMaxCatchUpTicks = 3;

// sendEvent: alone was empirically swallowed on the target build, so touches are
// also delivered straight to the hit-tested view. If a build ever delivers both,
// every touch phase fires twice — set this to 0.
#define AI_DELIVER_DIRECTLY_TO_VIEW 1

typedef NS_ENUM(NSInteger, AISwipeDirection) {
    AISwipeDirectionUp = 0, AISwipeDirectionDown, AISwipeDirectionLeft, AISwipeDirectionRight,
};

typedef NS_ENUM(NSInteger, AIPlayerState) {
    AIPlayerStateOff = 0, AIPlayerStateStarting, AIPlayerStateOn, AIPlayerStateStopping,
};

typedef struct {
    float            detProbability;
    float            dirProbs[4];       // UP, DOWN, LEFT, RIGHT
    AISwipeDirection direction;
    float            dirConfidence;
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

// The game's key window (the overlay window can never become key).
static UIWindow *AIKeyWindow(void) {
    for (UIScene *scene in [UIApplication sharedApplication].connectedScenes) {
        if (scene.activationState != UISceneActivationStateForegroundActive) continue;
        if (![scene isKindOfClass:[UIWindowScene class]]) continue;
        for (UIWindow *window in ((UIWindowScene *)scene).windows) {
            if (window.isKeyWindow) return window;
        }
    }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return [UIApplication sharedApplication].keyWindow;    // apps without a scene manifest
#pragma clang diagnostic pop
}

static inline void *AIArrayData(MLMultiArray *array) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    return array.dataPointer;
#pragma clang diagnostic pop
}

// Copies the first `n` elements of a 1-D-like output, honouring the innermost stride.
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
        default:
            for (NSInteger i = 0; i < n; i++) dst[i] = array[i].floatValue;
            return YES;
    }
}

// Writes `n` floats into an input array this file allocated (contiguous).
static void AIFillArray(MLMultiArray *array, const float *src, NSInteger n) {
    void *dst = AIArrayData(array);
    if (array.dataType == MLMultiArrayDataTypeDouble) {
        double *d = (double *)dst;
        for (NSInteger i = 0; i < n; i++) d[i] = src[i];
    } else {
        memcpy(dst, src, sizeof(float) * (size_t)n);
    }
}

// Allocates an input array of exactly the shape the model declares, or fails.
static MLMultiArray *AIInputArray(MLModel *model, NSString *name, NSArray<NSNumber *> *shape,
                                  NSError **error) {
    MLMultiArrayConstraint *c = model.modelDescription.inputDescriptionsByName[name].multiArrayConstraint;
    if (!c) {
        if (error) *error = AIError(10, [NSString stringWithFormat:@"model has no multi-array input '%@'", name]);
        return nil;
    }
    if (![c.shape isEqualToArray:shape]) {
        if (error) *error = AIError(11, [NSString stringWithFormat:
            @"input '%@' has shape %@, config expects %@ (model and AIPlayerConfig.json are from different exports)",
            name, c.shape, shape]);
        return nil;
    }
    MLMultiArrayDataType type = c.dataType == MLMultiArrayDataTypeDouble ? MLMultiArrayDataTypeDouble
                                                                          : MLMultiArrayDataTypeFloat32;
    return [[MLMultiArray alloc] initWithShape:shape dataType:type error:error];
}

static BOOL AIHasOutputs(MLModel *model, NSArray<NSString *> *names, NSError **error) {
    for (NSString *name in names) {
        if (!model.modelDescription.outputDescriptionsByName[name]) {
            if (error) *error = AIError(12, [NSString stringWithFormat:@"model has no output '%@'", name]);
            return NO;
        }
    }
    return YES;
}

static void AIFillPrediction(float detLogit, const float dirLogits[4], AIPrediction *out) {
    float m = fmaxf(fmaxf(dirLogits[0], dirLogits[1]), fmaxf(dirLogits[2], dirLogits[3]));
    float sum = 0;
    for (int i = 0; i < 4; i++) { out->dirProbs[i] = expf(dirLogits[i] - m); sum += out->dirProbs[i]; }
    int best = 0;
    for (int i = 0; i < 4; i++) {
        out->dirProbs[i] /= sum;
        if (out->dirProbs[i] > out->dirProbs[best]) best = i;
    }
    out->detProbability = 1.0f / (1.0f + expf(-detLogit));
    out->direction      = (AISwipeDirection)best;
    out->dirConfidence  = out->dirProbs[best];
}

// =============================================================================
// MARK: - Configuration (AIPlayerConfig.json)
// =============================================================================

@interface AIConfig : NSObject
@property (nonatomic, readonly) NSURL *directory;
@property (nonatomic, readonly) NSDictionary *raw;
@property (nonatomic, readonly) BOOL causal;
@property (nonatomic, readonly) NSArray<NSString *> *modelFiles;
@property (nonatomic, readonly) NSInteger imgSize;
@property (nonatomic, readonly) double tickHz;
@property (nonatomic, readonly) float fireThreshold, releaseThreshold, minDirConfidence;
@property (nonatomic, readonly) double cooldown;
@property (nonatomic, readonly) float inputGain, inputBias;
@property (nonatomic, readonly) NSInteger debugDumpFrames;
@property (nonatomic, readonly) MLComputeUnits computeUnits;
@property (nonatomic, readonly) NSInteger fastLen, featDim;          // causal only
@property (nonatomic, readonly) NSArray<NSNumber *> *slowOffsets;    // causal only, oldest first
+ (nullable instancetype)locateAndLoad:(NSError **)error;
@end

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

// Depth-limited search for a file called `name` (fallback only).
static NSString *AIFindFileNamed(NSString *name, NSString *root, NSInteger maxDepth) {
    NSFileManager *fm = [NSFileManager defaultManager];
    BOOL isDir = NO;
    if (![fm fileExistsAtPath:root isDirectory:&isDir] || !isDir) return nil;
    NSDirectoryEnumerator<NSURL *> *walker =
        [fm enumeratorAtURL:[NSURL fileURLWithPath:root isDirectory:YES]
 includingPropertiesForKeys:nil
                    options:NSDirectoryEnumerationSkipsHiddenFiles
               errorHandler:^BOOL(NSURL *url, NSError *error) { return YES; }];
    NSInteger rootDepth = root.pathComponents.count;
    for (NSURL *url in walker) {
        if ((NSInteger)url.pathComponents.count - rootDepth > maxDepth) { [walker skipDescendants]; continue; }
        if ([url.lastPathComponent isEqualToString:name]) return url.path;
    }
    return nil;
}

static NSString *AIFindConfigDirectory(void) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *dylibDir  = AIDylibDirectory();
    NSString *bundleDir = [NSBundle mainBundle].bundlePath;

    // Sideloadly injects AIPlayer.bundle (config + models) either next to the dylib
    // (usually App.app/Frameworks) or in the app's root; check both, then loose files.
    NSMutableArray<NSString *> *bases = [NSMutableArray array];
    if (dylibDir) {
        [bases addObject:[dylibDir stringByAppendingPathComponent:@"AIPlayer.bundle"]];
        [bases addObject:[[dylibDir stringByDeletingLastPathComponent]
                          stringByAppendingPathComponent:@"AIPlayer.bundle"]];
        [bases addObject:dylibDir];
    }
    if (bundleDir) {
        [bases addObject:[bundleDir stringByAppendingPathComponent:@"AIPlayer.bundle"]];
        [bases addObject:[bundleDir stringByAppendingPathComponent:@"Frameworks/AIPlayer.bundle"]];
        [bases addObject:[bundleDir stringByAppendingPathComponent:@"PlugIns/AIPlayer.bundle"]];
        [bases addObject:bundleDir];
        [bases addObject:[bundleDir stringByAppendingPathComponent:@"Frameworks"]];
    }
    for (NSString *base in bases) {
        if ([fm fileExistsAtPath:[base stringByAppendingPathComponent:kConfigFileName]]) return base;
    }
    // Fallback: a renamed bundle anywhere a few levels deep in the app.
    NSString *found = bundleDir ? AIFindFileNamed(kConfigFileName, bundleDir, 4) : nil;
    if (found) return [found stringByDeletingLastPathComponent];
    AILog("%{public}@ not found (searched dylib dir %{public}@ and app bundle %{public}@). "
          "Inject AIPlayer.bundle alongside AIPlayer.dylib.",
          kConfigFileName, dylibDir ?: @"(nil)", bundleDir ?: @"(nil)");
    return nil;
}

@implementation AIConfig

static double AINum(NSDictionary *d, NSString *key, double fallback) {
    id v = d[key];
    return [v isKindOfClass:[NSNumber class]] ? [v doubleValue] : fallback;
}

+ (instancetype)locateAndLoad:(NSError **)error {
    NSString *dir = AIFindConfigDirectory();
    if (!dir) {
        if (error) *error = AIError(1, @"AIPlayerConfig.json not found");
        return nil;
    }
    NSData *data = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:kConfigFileName]];
    NSDictionary *json = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:error] : nil;
    if (![json isKindOfClass:[NSDictionary class]]) {
        if (error && !*error) *error = AIError(2, @"AIPlayerConfig.json is not a JSON object");
        return nil;
    }
    AIConfig *c = [AIConfig new];
    if (![c _parse:json directory:[NSURL fileURLWithPath:dir isDirectory:YES] error:error]) return nil;
    return c;
}

- (BOOL)_parse:(NSDictionary *)j directory:(NSURL *)dir error:(NSError **)error {
    _directory = dir;
    _raw = j;

    id type = j[@"model_type"];
    if ([type isEqual:@"causal"]) _causal = YES;
    else if (![type isEqual:@"simple"]) {
        if (error) *error = AIError(3, [NSString stringWithFormat:@"unknown model_type '%@'", type]);
        return NO;
    }
    NSArray *models = j[@"models"];
    BOOL modelsOK = [models isKindOfClass:[NSArray class]] && models.count == (_causal ? 2u : 1u);
    for (id name in modelsOK ? models : @[]) modelsOK = modelsOK && [name isKindOfClass:[NSString class]];
    if (!modelsOK) {
        if (error) *error = AIError(4, @"'models' must list 1 (simple) or 2 (causal: encoder, head) .mlmodelc names");
        return NO;
    }
    _modelFiles = models;

    _imgSize          = (NSInteger)AINum(j, @"img_size", 128);
    _tickHz           = AINum(j, @"tick_hz", 24);
    _fireThreshold    = (float)AINum(j, @"fire_threshold", 0.6);
    _releaseThreshold = (float)AINum(j, @"release_threshold", _fireThreshold * 0.5);
    _minDirConfidence = (float)AINum(j, @"min_dir_confidence", 0);
    _cooldown         = AINum(j, @"cooldown_s", 0.25);
    _inputGain        = (float)AINum(j, @"input_gain", 1);
    _inputBias        = (float)AINum(j, @"input_bias", 0);
    _debugDumpFrames  = (NSInteger)AINum(j, @"debug_dump_frames", 0);

    NSString *units = j[@"compute_units"];
    _computeUnits = [units isEqualToString:@"cpu_only"]    ? MLComputeUnitsCPUOnly
                  : [units isEqualToString:@"cpu_and_gpu"] ? MLComputeUnitsCPUAndGPU
                                                           : MLComputeUnitsAll;

    if (_imgSize < 8 || _imgSize > 1024 || _tickHz < 1 || _tickHz > 120 ||
        _releaseThreshold > _fireThreshold) {
        if (error) *error = AIError(5, @"AIPlayerConfig.json values out of range "
                                        "(img_size, tick_hz, or release_threshold > fire_threshold)");
        return NO;
    }

    if (_causal) {
        _fastLen = (NSInteger)AINum(j, @"fast_len", 8);
        _featDim = (NSInteger)AINum(j, @"feat_dim", 128);
        NSArray *slow = j[@"slow_offsets"];
        _slowOffsets = [slow isKindOfClass:[NSArray class]] ? slow : @[@19, @13];
        BOOL slowOK = _slowOffsets.count >= 1;
        for (id o in _slowOffsets) slowOK = slowOK && [o isKindOfClass:[NSNumber class]] && [o integerValue] >= 0;
        if (_fastLen < 1 || _featDim < 1 || !slowOK) {
            if (error) *error = AIError(6, @"invalid causal settings in AIPlayerConfig.json");
            return NO;
        }
    }
    return YES;
}

@end

// =============================================================================
// MARK: - Inference engine
//
// Owns preprocessing and the Core ML models. Every public method takes the lock.
// =============================================================================

@interface AIInferenceEngine : NSObject
- (nullable instancetype)initWithConfig:(AIConfig *)config error:(NSError **)error;
- (void)resetSession;
// A real frame for this tick. Returns YES and fills `out` when a prediction is available
// (the causal model needs `history` ticks to warm up).
- (BOOL)ingestFrame:(CVPixelBufferRef)pixelBuffer orientation:(CGImagePropertyOrientation)orientation
         prediction:(AIPrediction *)out;
// The source skipped this tick: repeat the previous frame (identical frame => neutral diff),
// keeping the causal feature ring on the training clock. No-op for the simple model.
- (void)ingestHeldFrame;
// The model input of the most recent tick (img_size^2 uint8), for debug dumps.
- (void)copyLastGray:(uint8_t *)dst;
@property (nonatomic, readonly) double lastInferenceMs;
@end

@implementation AIInferenceEngine {
    NSLock   *_lock;
    AIConfig *_cfg;
    NSInteger _S, _pixels;

    // simple: _model only. causal: _model = encoder, _head = head.
    MLModel *_model, *_head;
    MLMultiArray *_in, *_headFast, *_headSlow;
    id<MLFeatureProvider> _provider, _headProvider;

    // preprocessing
    CIContext  *_ci;
    uint8_t    *_rgba;  size_t _rgbaCapacity;
    AIResampler _resampler;  int _rsW, _rsH;  BOOL _hasResampler;
    uint8_t     _lut[256];                     // input_gain / input_bias, applied to gray
    uint8_t    *_gray, *_prevGray;
    BOOL        _hasPrev;
    float      *_stage;                        // model input staging (float32)

    // causal feature ring
    NSInteger _history, _tick;
    float    *_ring, *_fastBuf, *_slowBuf;

    double _lastInferenceMs;
}

- (instancetype)initWithConfig:(AIConfig *)config error:(NSError **)error {
    if (!(self = [super init])) return nil;
    _lock = [NSLock new];
    _cfg = config;
    _S = config.imgSize;
    _pixels = _S * _S;

    // No colour management anywhere: the training decoder (FFmpeg/NVDEC) converted
    // YUV->RGB and nothing else, so CoreImage must not apply transfer-function or
    // gamut conversions either.
    _ci = [CIContext contextWithOptions:@{ kCIContextWorkingColorSpace: [NSNull null],
                                           kCIContextOutputColorSpace:  [NSNull null] }];

    for (int v = 0; v < 256; v++) {
        float x = roundf(v * config.inputGain + config.inputBias);
        _lut[v] = (uint8_t)(x < 0 ? 0 : (x > 255 ? 255 : x));
    }

    _gray     = calloc((size_t)_pixels, 1);
    _prevGray = calloc((size_t)_pixels, 1);
    _stage    = calloc((size_t)_pixels * (config.causal ? 2 : 1), sizeof(float));
    if (!_gray || !_prevGray || !_stage) {
        if (error) *error = AIError(20, @"out of memory");
        return nil;
    }
    if (![self _loadModels:error]) return nil;
    [self _resetLocked];
    AILog("Loaded %{public}@ model from %{public}@ (img %ld, %.0f Hz, fire %.2f / release %.2f)",
          config.causal ? @"causal" : @"simple", config.directory.path, (long)_S, config.tickHz,
          config.fireThreshold, config.releaseThreshold);
    return self;
}

- (void)dealloc {
    if (_hasResampler) AIResamplerFree(&_resampler);
    free(_rgba); free(_gray); free(_prevGray); free(_stage);
    free(_ring); free(_fastBuf); free(_slowBuf);
}

- (double)lastInferenceMs { return _lastInferenceMs; }

- (MLModel *)_open:(NSString *)file error:(NSError **)error {
    MLModelConfiguration *mc = [MLModelConfiguration new];
    mc.computeUnits = _cfg.computeUnits;
    return [MLModel modelWithContentsOfURL:[_cfg.directory URLByAppendingPathComponent:file isDirectory:YES]
                             configuration:mc error:error];
}

- (BOOL)_loadModels:(NSError **)error {
    NSArray<NSNumber *> *frameShape = @[@1, @(_cfg.causal ? 2 : 1), @(_S), @(_S)];
    _model = [self _open:_cfg.modelFiles[0] error:error];
    if (!_model) return NO;
    _in = AIInputArray(_model, @"frame", frameShape, error);
    if (!_in) return NO;
    _provider = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{ @"frame": [MLFeatureValue featureValueWithMultiArray:_in] } error:error];
    if (!_provider) return NO;

    if (!_cfg.causal) return AIHasOutputs(_model, @[@"det_logit", @"dir_logits"], error);

    if (!AIHasOutputs(_model, @[@"feat"], error)) return NO;
    _head = [self _open:_cfg.modelFiles[1] error:error];
    if (!_head || !AIHasOutputs(_head, @[@"det_logit", @"dir_logits"], error)) return NO;

    NSInteger F = _cfg.fastLen, D = _cfg.featDim, K = (NSInteger)_cfg.slowOffsets.count;
    _headFast = AIInputArray(_head, @"fast_feats", @[@1, @(F), @(D)], error);
    _headSlow = AIInputArray(_head, @"slow_feats", @[@1, @(K), @(D)], error);
    if (!_headFast || !_headSlow) return NO;
    _headProvider = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
        @"fast_feats": [MLFeatureValue featureValueWithMultiArray:_headFast],
        @"slow_feats": [MLFeatureValue featureValueWithMultiArray:_headSlow] } error:error];
    if (!_headProvider) return NO;

    _history = F;
    for (NSNumber *o in _cfg.slowOffsets) _history = MAX(_history, o.integerValue + 1);
    _ring    = calloc((size_t)(_history * D), sizeof(float));
    _fastBuf = calloc((size_t)(F * D), sizeof(float));
    _slowBuf = calloc((size_t)(K * D), sizeof(float));
    if (!_ring || !_fastBuf || !_slowBuf) {
        if (error) *error = AIError(20, @"out of memory");
        return NO;
    }
    return YES;
}

- (void)resetSession {
    [_lock lock];
    [self _resetLocked];
    [_lock unlock];
}

- (void)_resetLocked {
    _tick = 0;
    _hasPrev = NO;
    if (_ring) memset(_ring, 0, sizeof(float) * (size_t)(_history * _cfg.featDim));
}

- (void)copyLastGray:(uint8_t *)dst {
    [_lock lock];
    memcpy(dst, _gray, (size_t)_pixels);
    [_lock unlock];
}

// ── per tick ────────────────────────────────────────────────────────────────

- (BOOL)ingestFrame:(CVPixelBufferRef)pixelBuffer orientation:(CGImagePropertyOrientation)orientation
         prediction:(AIPrediction *)out {
    [_lock lock];
    BOOL produced = NO;
    CFTimeInterval t0 = CACurrentMediaTime();
    if ([self _preprocess:pixelBuffer orientation:orientation]) {
        produced = _cfg.causal ? [self _causalStepHeld:NO prediction:out]
                               : [self _simplePredict:out];
    }
    _lastInferenceMs = (CACurrentMediaTime() - t0) * 1000.0;
    [_lock unlock];
    return produced;
}

- (void)ingestHeldFrame {
    [_lock lock];
    if (_cfg.causal && _hasPrev) [self _causalStepHeld:YES prediction:NULL];
    [_lock unlock];
}

// Pixel buffer -> _gray, bit-exact with the dataset builder (see AIPreprocess.h).
- (BOOL)_preprocess:(CVPixelBufferRef)pb orientation:(CGImagePropertyOrientation)orientation {
    const uint8_t *pixels = NULL;
    size_t w = 0, h = 0, rowBytes = 0;
    int r = 0, g = 1, b = 2;
    BOOL locked = NO;

    if (CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_32BGRA &&
        orientation == kCGImagePropertyOrientationUp) {
        // Already RGB: read it directly.
        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        locked = YES;
        pixels = CVPixelBufferGetBaseAddress(pb);
        w = CVPixelBufferGetWidth(pb);
        h = CVPixelBufferGetHeight(pb);
        rowBytes = CVPixelBufferGetBytesPerRow(pb);
        r = 2; g = 1; b = 0;
    } else {
        // YUV (ReplayKit's usual 420f) or rotated: CoreImage does the YCbCr->RGB matrix and
        // the rotation at full resolution, with colour management disabled.
        CIImage *image = [CIImage imageWithCVPixelBuffer:pb options:@{ kCIImageColorSpace: [NSNull null] }];
        if (orientation != kCGImagePropertyOrientationUp)
            image = [image imageByApplyingCGOrientation:orientation];
        CGRect e = image.extent;
        if (CGRectIsEmpty(e) || CGRectIsInfinite(e)) return NO;
        image = [image imageByApplyingTransform:CGAffineTransformMakeTranslation(-e.origin.x, -e.origin.y)];
        w = (size_t)e.size.width;
        h = (size_t)e.size.height;
        rowBytes = w * 4;
        if (rowBytes * h > _rgbaCapacity) {
            free(_rgba);
            _rgba = malloc(rowBytes * h);
            _rgbaCapacity = _rgba ? rowBytes * h : 0;
            if (!_rgba) return NO;
        }
        [_ci render:image toBitmap:_rgba rowBytes:(ptrdiff_t)rowBytes bounds:CGRectMake(0, 0, w, h)
             format:kCIFormatRGBA8 colorSpace:nil];
        pixels = _rgba;
    }

    BOOL ok = pixels != NULL && w > 0 && h > 0;
    if (ok && (!_hasResampler || _rsW != (int)w || _rsH != (int)h)) {
        if (_hasResampler) AIResamplerFree(&_resampler);
        _hasResampler = AIResamplerInit(&_resampler, (int)w, (int)h, (int)_S, (int)_S) == 0;
        _rsW = (int)w;
        _rsH = (int)h;
        ok = _hasResampler;
        if (ok) AILog("Capture %zux%zu -> %ldx%ld", w, h, (long)_S, (long)_S);
    }
    if (ok) {
        memcpy(_prevGray, _gray, (size_t)_pixels);
        AIResizeToGray(&_resampler, pixels, rowBytes, r, g, b, _gray);
        for (NSInteger i = 0; i < _pixels; i++) _gray[i] = _lut[_gray[i]];
    }
    if (locked) CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    return ok;
}

- (BOOL)_readHeads:(id<MLFeatureProvider>)result error:(NSError *)error into:(AIPrediction *)out {
    float det = 0, dir[4] = { 0, 0, 0, 0 };
    if (!result ||
        !AICopyVector([result featureValueForName:@"det_logit"].multiArrayValue, &det, 1) ||
        !AICopyVector([result featureValueForName:@"dir_logits"].multiArrayValue, dir, 4)) {
        AILog("Prediction failed: %{public}@", error.localizedDescription ?: @"bad output");
        return NO;
    }
    AIFillPrediction(det, dir, out);
    return YES;
}

- (BOOL)_simplePredict:(AIPrediction *)out {
    for (NSInteger i = 0; i < _pixels; i++) _stage[i] = (float)_gray[i];
    AIFillArray(_in, _stage, _pixels);
    NSError *error = nil;
    id<MLFeatureProvider> result = [_model predictionFromFeatures:_provider error:&error];
    return [self _readHeads:result error:error into:out];
}

// Encode this tick into the ring; when warm and `out` is given, run the windowed head.
- (BOOL)_causalStepHeld:(BOOL)held prediction:(AIPrediction *)out {
    float *grayOut = _stage, *diffOut = _stage + _pixels;
    for (NSInteger i = 0; i < _pixels; i++) {
        grayOut[i] = (float)_gray[i];
        // First frame and held (repeated) frames have a neutral diff, as in training.
        diffOut[i] = (held || !_hasPrev) ? 127.0f : (float)AIDiffValue(_gray[i], _prevGray[i]);
    }
    if (!held) _hasPrev = YES;

    AIFillArray(_in, _stage, 2 * _pixels);
    NSError *error = nil;
    id<MLFeatureProvider> enc = [_model predictionFromFeatures:_provider error:&error];
    const NSInteger D = _cfg.featDim;
    if (!AICopyVector([enc featureValueForName:@"feat"].multiArrayValue, _ring + (_tick % _history) * D, D)) {
        AILog("Encoder failed: %{public}@", error.localizedDescription ?: @"bad output");
        return NO;
    }
    _tick++;
    if (!out || _tick < _history) return NO;

    const NSInteger F = _cfg.fastLen;
    for (NSInteger k = 0; k < F; k++) {                         // [t-F+1 .. t], oldest first
        NSInteger age = F - 1 - k;
        memcpy(_fastBuf + k * D, _ring + ((_tick - 1 - age) % _history) * D, sizeof(float) * (size_t)D);
    }
    NSInteger j = 0;
    for (NSNumber *offset in _cfg.slowOffsets) {                // e.g. [t-19, t-13]
        memcpy(_slowBuf + j * D, _ring + ((_tick - 1 - offset.integerValue) % _history) * D,
               sizeof(float) * (size_t)D);
        j++;
    }
    AIFillArray(_headFast, _fastBuf, F * D);
    AIFillArray(_headSlow, _slowBuf, j * D);
    id<MLFeatureProvider> result = [_head predictionFromFeatures:_headProvider error:&error];
    return [self _readHeads:result error:error into:out];
}

@end

// =============================================================================
// MARK: - Debug dump
//
// With "debug_dump_frames": N in the config, the first N ticks after AI ON are
// written to Documents/AIPlayerDump/<time>/:
//   frames.u8   N x img x img uint8, exactly what the model saw
//   ticks.csv   tick,held,has_pred,det_prob,p_up,p_down,p_left,p_right,fired
//   config.json the config in use
// tools/diagnose.py compares these against PyTorch and the training frames.
// =============================================================================

@interface AIDumpWriter : NSObject
- (nullable instancetype)initWithConfig:(AIConfig *)config;
- (BOOL)wantsMore;
- (void)appendGray:(const uint8_t *)gray held:(BOOL)held prediction:(const AIPrediction *)p fired:(BOOL)fired;
@end

@implementation AIDumpWriter {
    dispatch_queue_t _queue;
    NSFileHandle *_frames, *_csv;
    NSInteger _limit, _count, _pixels;
    NSString *_dir;
}

- (instancetype)initWithConfig:(AIConfig *)config {
    if (!(self = [super init])) return nil;
    NSString *docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES).firstObject;
    NSDateFormatter *fmt = [NSDateFormatter new];
    fmt.dateFormat = @"yyyyMMdd-HHmmss";
    _dir = [[docs stringByAppendingPathComponent:@"AIPlayerDump"]
            stringByAppendingPathComponent:[fmt stringFromDate:[NSDate date]]];
    NSFileManager *fm = [NSFileManager defaultManager];
    if (![fm createDirectoryAtPath:_dir withIntermediateDirectories:YES attributes:nil error:nil]) return nil;

    NSString *framesPath = [_dir stringByAppendingPathComponent:@"frames.u8"];
    NSString *csvPath = [_dir stringByAppendingPathComponent:@"ticks.csv"];
    [fm createFileAtPath:framesPath contents:nil attributes:nil];
    [fm createFileAtPath:csvPath
                contents:[@"tick,held,has_pred,det_prob,p_up,p_down,p_left,p_right,fired\n"
                          dataUsingEncoding:NSUTF8StringEncoding]
              attributes:nil];
    NSData *cfg = [NSJSONSerialization dataWithJSONObject:config.raw options:NSJSONWritingPrettyPrinted error:nil];
    [cfg writeToFile:[_dir stringByAppendingPathComponent:@"config.json"] atomically:YES];

    _frames = [NSFileHandle fileHandleForWritingAtPath:framesPath];
    _csv = [NSFileHandle fileHandleForWritingAtPath:csvPath];
    [_csv seekToEndOfFile];
    if (!_frames || !_csv) return nil;
    _queue = dispatch_queue_create("com.aiplayer.dump", DISPATCH_QUEUE_SERIAL);
    _limit = config.debugDumpFrames;
    _pixels = config.imgSize * config.imgSize;
    AILog("Dumping %ld ticks to %{public}@", (long)_limit, _dir);
    return self;
}

- (BOOL)wantsMore { return _count < _limit; }

- (void)appendGray:(const uint8_t *)gray held:(BOOL)held prediction:(const AIPrediction *)p fired:(BOOL)fired {
    if (_count >= _limit) return;
    NSInteger tick = _count++;
    BOOL last = _count == _limit;
    NSData *frame = [NSData dataWithBytes:gray length:(NSUInteger)_pixels];
    NSString *line = p
        ? [NSString stringWithFormat:@"%ld,%d,1,%.6f,%.6f,%.6f,%.6f,%.6f,%d\n", (long)tick, held,
           p->detProbability, p->dirProbs[0], p->dirProbs[1], p->dirProbs[2], p->dirProbs[3], fired]
        : [NSString stringWithFormat:@"%ld,%d,0,,,,,,0\n", (long)tick, held];
    dispatch_async(_queue, ^{
        [self->_frames writeData:frame];
        [self->_csv writeData:[line dataUsingEncoding:NSUTF8StringEncoding]];
        if (last) {
            [self->_frames closeFile];
            [self->_csv closeFile];
            AILog("Dump complete: %{public}@", self->_dir);
        }
    });
}

@end

// =============================================================================
// MARK: - Touch synthesis (unchanged, validated on-device)
//
// Builds a UITouch + UITouchesEvent by hand (private API) and delivers it through
// the window and, because sendEvent: alone was swallowed on the target build,
// directly to the hit-tested view.
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
        default: break;
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
    if (!_touch && phase != UITouchPhaseBegan) return;
    _lastPoint = point;
    _touch = [_synth dispatchPhase:phase atPoint:point touch:_touch];
}

@end

// =============================================================================
// MARK: - Player controller: capture, clock, detection gate
// =============================================================================

@interface AIPlayerController : NSObject
@property (nonatomic, readonly) AIPlayerState state;                               // main thread
@property (nonatomic, copy, nullable) void (^onStateChange)(AIPlayerState state);  // main thread
+ (instancetype)shared;
- (void)toggle;                                                                    // main thread
@end

@implementation AIPlayerController {
    dispatch_queue_t _setupQueue;
    AIPlayerState _state;
    atomic_bool _active;                 // read by the capture queue

    // Set on the setup queue before capture starts; read by the capture queue.
    AIConfig *_config;
    AIInferenceEngine *_engine;
    AIDumpWriter *_dump;
    uint8_t *_dumpGray;

    // Capture queue only (reset before capture starts).
    CFTimeInterval _tickInterval;
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
        _setupQueue = dispatch_queue_create("com.aiplayer.setup", DISPATCH_QUEUE_SERIAL);
        atomic_init(&_active, false);
        dispatch_async(_setupQueue, ^{ [self _ensureLoaded:NULL]; });   // first tap is instant

        [[NSNotificationCenter defaultCenter]
            addObserverForName:UIApplicationDidEnterBackgroundNotification object:nil
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(NSNotification *note) {
            if (self->_state == AIPlayerStateOn) [self stop];
        }];
    }
    return self;
}

- (void)dealloc { free(_dumpGray); }

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

- (BOOL)_ensureLoaded:(NSError **)error {          // setup queue only
    if (_engine) return YES;
    AIConfig *config = [AIConfig locateAndLoad:error];
    if (!config) return NO;
    NSError *loadError = nil;
    AIInferenceEngine *engine = [[AIInferenceEngine alloc] initWithConfig:config error:&loadError];
    if (!engine) {
        AILog("Model load failed: %{public}@", loadError.localizedDescription);
        if (error) *error = loadError;
        return NO;
    }
    _config = config;
    _engine = engine;
    _tickInterval = 1.0 / config.tickHz;
    return YES;
}

- (void)start {
    [self _setState:AIPlayerStateStarting];
    dispatch_async(_setupQueue, ^{
        NSError *error = nil;
        if (![self _ensureLoaded:&error]) {
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

- (void)_resetRuntimeState {                       // setup queue, before capture starts
    _hasClock = NO;
    _nextTick = 0;
    _armed = NO;                                   // must see a low probability before the first fire
    _lastFire = 0;
    _statsStart = 0;
    _statFrames = _statTicks = _statPreds = _statFires = 0;
    _statDetSum = 0;
    _statDetMax = 0;

    _dump = nil;
    if (_config.debugDumpFrames > 0) {
        if (!_dumpGray) _dumpGray = malloc((size_t)(_config.imgSize * _config.imgSize));
        if (_dumpGray) _dump = [[AIDumpWriter alloc] initWithConfig:_config];
    }
}

// ── capture queue ───────────────────────────────────────────────────────────

// Number of ticks the frame stamped `time` accounts for (0 = drop it).
- (NSInteger)_ticksDueAt:(CFTimeInterval)time {
    if (!_hasClock) {
        _hasClock = YES;
        _nextTick = time + _tickInterval;
        return 1;
    }
    CFTimeInterval ahead = time + kTickTolerance - _nextTick;
    if (ahead < 0) return 0;
    NSInteger due = 1 + (NSInteger)floor(ahead / _tickInterval);
    if (due > kMaxCatchUpTicks) {                  // stall or discontinuity: resync
        _nextTick = time + _tickInterval;
        return 1;
    }
    _nextTick += due * _tickInterval;
    return due;
}

static CGImagePropertyOrientation AIBufferOrientation(CMSampleBufferRef buffer) {
    CFTypeRef value = CMGetAttachment(buffer, (__bridge CFStringRef)RPVideoSampleOrientationKey, NULL);
    if (value && CFGetTypeID(value) == CFNumberGetTypeID()) {
        uint32_t o = 0;
        CFNumberGetValue((CFNumberRef)value, kCFNumberSInt32Type, &o);
        if (o >= 1 && o <= 8) return (CGImagePropertyOrientation)o;
    }
    return kCGImagePropertyOrientationUp;
}

- (void)_handleSampleBuffer:(CMSampleBufferRef)buffer {
    _statFrames++;
    CVImageBufferRef pixels = CMSampleBufferGetImageBuffer(buffer);
    if (!pixels) return;

    CFTimeInterval time = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(buffer));
    if (!isfinite(time)) time = CACurrentMediaTime();

    NSInteger due = [self _ticksDueAt:time];
    if (due == 0) return;
    _statTicks += due;

    for (NSInteger i = 0; i < due - 1; i++) {      // source skipped ticks: hold the last frame
        [_engine ingestHeldFrame];
        [self _dumpHeld:YES prediction:NULL fired:NO];
    }

    AIPrediction prediction;
    BOOL has = [_engine ingestFrame:pixels orientation:AIBufferOrientation(buffer) prediction:&prediction];
    BOOL fired = has && [self _handlePrediction:prediction];
    [self _dumpHeld:NO prediction:has ? &prediction : NULL fired:fired];
    [self _logStatsIfDue];
}

- (void)_dumpHeld:(BOOL)held prediction:(const AIPrediction *)p fired:(BOOL)fired {
    if (!_dump || ![_dump wantsMore]) return;
    [_engine copyLastGray:_dumpGray];
    [_dump appendGray:_dumpGray held:held prediction:p fired:fired];
}

// Fires at most once per detection burst: prob must drop below release_threshold
// before the next prob >= fire_threshold can fire again.
- (BOOL)_handlePrediction:(AIPrediction)p {
    _statPreds++;
    _statDetSum += p.detProbability;
    if (p.detProbability > _statDetMax) _statDetMax = p.detProbability;

    if (p.detProbability < _config.releaseThreshold) _armed = YES;
    if (!_armed || p.detProbability < _config.fireThreshold) return NO;
    if (p.dirConfidence < _config.minDirConfidence) return NO;

    CFTimeInterval now = CACurrentMediaTime();
    if (now - _lastFire < _config.cooldown) return NO;
    _lastFire = now;
    _armed = NO;
    _statFires++;
    [[AISwipeInjector shared] swipeInDirection:p.direction];
    return YES;
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
//
// ReplayKit's in-app capture records every window of the app, including this
// one, and the model never saw a button in training. While the AI is ON the
// button therefore collapses to a small, faint dot (still tappable to stop).
// =============================================================================

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
    _button.center = CGPointMake(CGRectGetWidth(bounds) - 80, MAX(self.view.safeAreaInsets.top, 20) + 60);
}

- (void)_render:(AIPlayerState)state {
    CGPoint center = _button.center;
    if (state == AIPlayerStateOn) {
        _button.bounds = CGRectMake(0, 0, 22, 22);
        _button.layer.cornerRadius = 11;
        _button.alpha = 0.35;
        [_button setTitle:nil forState:UIControlStateNormal];
        _button.backgroundColor = [UIColor colorWithRed:0.20 green:0.70 blue:0.35 alpha:1.0];
    } else {
        NSString *title;
        UIColor *color;
        switch (state) {
            case AIPlayerStateStarting:
                title = @"… STARTING"; color = [UIColor colorWithRed:0.85 green:0.65 blue:0.15 alpha:0.95]; break;
            case AIPlayerStateStopping:
                title = @"… STOPPING"; color = [UIColor colorWithRed:0.85 green:0.65 blue:0.15 alpha:0.95]; break;
            default:
                title = @"▶ AI: OFF";  color = [UIColor colorWithWhite:0.15 alpha:0.92]; break;
        }
        _button.bounds = CGRectMake(0, 0, 130, 44);
        _button.layer.cornerRadius = 12;
        _button.alpha = 1.0;
        [_button setTitle:title forState:UIControlStateNormal];
        _button.backgroundColor = color;
    }
    _button.center = center;
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
