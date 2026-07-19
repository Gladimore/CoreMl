// =============================================================================
// AIPlayer.xm — standalone AI-only tweak.
//
// Shows a small floating "AI: OFF / AI: ON" button. Tapping it starts/stops
// a screen-capture + Core ML inference loop. Currently LOGS predictions
// (NSLog) rather than acting on them — touch injection is a separate,
// not-yet-built piece (see the note near ai_processCapturedBuffer:).
//
// PREPROCESSING CONTRACT — must exactly match build_dataset_v3_gpu.py:
//   1. Resize to (kImgSize, kImgSize) via SQUARE STRETCH (not
//      aspect-preserving) — torchvision's TF.resize(..., antialias=True)
//   2. RGB -> grayscale via ITU-R 601-2 luma:
//        gray = 0.2989*R + 0.5870*G + 0.1140*B
//   3. diff[t] = clamp( floorDiv((frame[t]-frame[t-1])*127, 255) + 127, 0, 255 )
//      diff[0] = 127 (neutral). floorDiv is FLOOR division, not truncating —
//      see floorDiv() below, do not replace with a plain `/`.
//
// If any of this drifts from the training pipeline, inference silently
// produces wrong (but plausible-looking) predictions — no crash, no error.
// =============================================================================

#import <UIKit/UIKit.h>
#import <ReplayKit/ReplayKit.h>
#import <CoreImage/CoreImage.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#include <math.h>

// ── Model config — MUST match your checkpoint's arch dict + dataset meta ───
static const NSInteger kImgSize     = 128;   // meta['img_size'] — CONFIRM against your dataset
static const NSInteger kFastLayers  = 2;     // arch['fast_layers']
static const NSInteger kSlowLayers  = 1;     // arch['slow_layers']
static const NSInteger kHidden      = 192;   // arch['hidden']
static const NSInteger kSlowHidden  = kHidden / 2;

static const NSInteger kSlowBranchEveryNTicks = 6;  // VALIDATE against SLOW_OFFSETS spacing
static const float     kDetectionThreshold    = 0.80f;
static const int       kTargetFPS             = 24;

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
    NSError *err = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:@[@1, @2, @(kImgSize), @(kImgSize)]
                                                     dataType:MLMultiArrayDataTypeFloat32
                                                        error:&err];
    float *dst = (float *)arr.dataPointer;
    const uint8_t *g = (const uint8_t *)gray.bytes, *d = (const uint8_t *)diff.bytes;
    NSInteger n = kImgSize * kImgSize;
    for (NSInteger i = 0; i < n; i++) {
        dst[i]     = g[i] * (1.0f / 255.0f);
        dst[n + i] = d[i] * (1.0f / 255.0f);
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

@interface AIOverlayVC : UIViewController
@property (nonatomic, strong) UIButton *toggleButton;
@property (nonatomic, assign) BOOL isPlaying;
@property (nonatomic, assign) NSInteger frameTick;
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

    // Load the compiled model from this tweak's own resource bundle.
    NSBundle *tweakBundle = [NSBundle bundleForClass:[InferenceEngine class]];
    NSURL *modelURL = [tweakBundle URLForResource:@"SwipeAnnotator" withExtension:@"mlmodelc"];
    if (modelURL) {
        NSError *err = nil;
        if (![[InferenceEngine sharedEngine] loadModelAtURL:modelURL error:&err]) {
            NSLog(@"[AIPlayer] Failed to load model: %@", err);
        } else {
            NSLog(@"[AIPlayer] Model loaded from %@", modelURL);
        }
    } else {
        NSLog(@"[AIPlayer] SwipeAnnotator.mlmodelc not found in bundle %@", tweakBundle.bundlePath);
    }
}

- (void)handleDrag:(UIPanGestureRecognizer *)pan {
    CGPoint d = [pan translationInView:self.view];
    pan.view.center = CGPointMake(pan.view.center.x + d.x, pan.view.center.y + d.y);
    [pan setTranslation:CGPointZero inView:self.view];
}

- (void)toggleTapped {
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

    __weak typeof(self) weakSelf = self;
    [[RPScreenRecorder sharedRecorder]
        startCaptureWithHandler:^(CMSampleBufferRef buf, RPSampleBufferType type, NSError *err) {
            if (type != RPSampleBufferTypeVideo || err) return;
            __strong typeof(weakSelf) strongSelf = weakSelf;
            if (!strongSelf || !strongSelf.isPlaying) return;
            [strongSelf processCapturedBuffer:buf];
        }
        completionHandler:^(NSError *err) {
            if (err) NSLog(@"[AIPlayer] startCapture error: %@", err);
            else NSLog(@"[AIPlayer] Capture started.");
        }];
}

- (void)stopCapture {
    [[RPScreenRecorder sharedRecorder] stopCaptureWithHandler:^(NSError *err) {
        NSLog(@"[AIPlayer] Capture stopped%@", err ? [NSString stringWithFormat:@" (error: %@)", err] : @"");
    }];
}

// Runs on RPScreenRecorder's capture queue, not main thread.
- (void)processCapturedBuffer:(CMSampleBufferRef)buf {
    static CFTimeInterval lastTime = 0;
    CFTimeInterval now = CACurrentMediaTime();
    if (now - lastTime < (1.0 / kTargetFPS)) return;
    lastTime = now;

    CVImageBufferRef px = CMSampleBufferGetImageBuffer(buf);
    if (!px) return;

    self.frameTick++;
    BOOL hasNewSlow = (self.frameTick % kSlowBranchEveryNTicks) == 0;

    NSError *err = nil;
    SwipePrediction *pred = [[InferenceEngine sharedEngine] predictWithPixelBuffer:px
                                                                   hasNewSlowFrame:hasNewSlow
                                                                              error:&err];
    if (!pred) {
        if (err) NSLog(@"[AIPlayer] predict error: %@", err);
        return;
    }

    if (pred.detProbability >= kDetectionThreshold) {
        NSLog(@"[AIPlayer] swipe dir=%ld conf=%.2f det=%.2f",
              (long)pred.direction, pred.dirConfidence, pred.detProbability);
        // Touch injection goes here once built. For now: log-only, so you
        // can validate predictions against on-screen gameplay before
        // wiring in anything that actually acts on the game.
    }
}

@end

// =============================================================================
// MARK: - Entry point
// =============================================================================

__attribute__((constructor))
static void AIPlayerInit(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        static UIWindow *win = nil;
        win = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
        win.windowLevel = UIWindowLevelAlert + 1;
        win.backgroundColor = [UIColor clearColor];
        win.rootViewController = [AIOverlayVC new];
        win.hidden = NO;
    });
}
