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
#include <mach/mach_time.h>
#include <mach-o/dyld.h>

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
// start/end are in points, in the same coordinate space as UIScreen.mainScreen.bounds.
- (void)injectSwipeFrom:(CGPoint)start to:(CGPoint)end duration:(NSTimeInterval)duration;
@end

@implementation TouchInjector {
    IOHIDEventSystemClientRef _client;
}

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
        if (!_client) NSLog(@"[AIPlayer] TouchInjector: IOHIDEventSystemClientCreate returned NULL "
                            @"(expected on a non-jailbroken device — injection will no-op)");
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
// Theos has two distinct resource-staging mechanisms, and this Makefile
// uses the plain one:
//
//   AIPlayer_RESOURCE_DIRS       (no BUNDLE_ prefix) -- what this Makefile sets.
//   AIPlayer_BUNDLE_RESOURCE_DIRS (WITH BUNDLE_ prefix) -- a different variable.
//
// Per Theos's own tweak.mk, only the BUNDLE_-prefixed variable triggers
// staging into a separate "<jb root>/Library/Application Support/
// AIPlayer.bundle/" bundle. Plain _RESOURCE_DIRS (what we use) copies
// resources directly alongside AIPlayer.dylib itself, in
// ".../Library/MobileSubstrate/DynamicLibraries/". There is no
// AIPlayer.bundle on disk at all with this Makefile -- searching for one,
// under any candidate root, will always fail. That was the actual bug
// behind "No resource bundle found" / "Exhausted all candidate roots".
//
// Fix: look for SwipeAnnotator.mlmodelc in the same directory as
// AIPlayer.dylib, found via dyld (works unmodified across rootful and
// rootless/var-jb layouts, since we ask the loader where the dylib
// actually came from rather than guessing a prefix).
// =============================================================================

static NSString *AIPlayerDylibDirectory(void) {
    uint32_t count = _dyld_image_count();
    NSString *lastResortMatch = nil;
    for (uint32_t i = 0; i < count; i++) {
        const char *name = _dyld_get_image_name(i);
        if (!name) continue;
        NSString *path = [NSString stringWithUTF8String:name];
        // Primary match: exact filename.
        if ([path.lastPathComponent isEqualToString:@"AIPlayer.dylib"])
            return path.stringByDeletingLastPathComponent;
        // Fallback: some jailbreak loaders (ElleKit, TrollStore-style
        // substrate replacements, symlinked DynamicLibraries dirs) report a
        // dyld image path that doesn't match the exact on-disk filename
        // case-for-case, or resolves through a symlink first. Case-insensitive
        // containment catches those without falsely matching an unrelated image.
        if ([path.lastPathComponent.lowercaseString containsString:@"aiplayer"] &&
            [path.pathExtension.lowercaseString isEqualToString:@"dylib"]) {
            lastResortMatch = path.stringByDeletingLastPathComponent;
        }
    }
    if (lastResortMatch) {
        NSLog(@"[AIPlayer] Matched AIPlayer.dylib via fallback (case/symlink mismatch on exact match)");
        return lastResortMatch;
    }
    return nil;
}

// Returns the directory that should contain SwipeAnnotator.mlmodelc: the
// same directory AIPlayer.dylib was loaded from. Falls back to the
// well-known DynamicLibraries paths (rootful and rootless) if dyld lookup
// fails for some reason, so this still works even if AIPlayerDylibDirectory
// can't find a match.
static NSArray<NSString *> *AIPlayerCandidateResourceDirs(void) {
    NSMutableArray<NSString *> *dirs = [NSMutableArray array];
    NSString *dylibDir = AIPlayerDylibDirectory();
    if (dylibDir) [dirs addObject:dylibDir];
    for (NSString *d in @[
        @"/var/jb/Library/MobileSubstrate/DynamicLibraries",
        @"/Library/MobileSubstrate/DynamicLibraries",
    ]) {
        if (![dirs containsObject:d]) [dirs addObject:d];
    }
    return dirs;
}

// Returns the URL to the compiled model, or nil with diagnostic logging if
// it can't be found in any candidate location.
static NSURL *AIPlayerModelURL(void) {
    for (NSString *dir in AIPlayerCandidateResourceDirs()) {
        NSString *modelPath = [dir stringByAppendingPathComponent:@"SwipeAnnotator.mlmodelc"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:modelPath]) {
            NSLog(@"[AIPlayer] Found SwipeAnnotator.mlmodelc at: %@", modelPath);
            return [NSURL fileURLWithPath:modelPath isDirectory:YES];
        }
        NSLog(@"[AIPlayer] SwipeAnnotator.mlmodelc not found at: %@", modelPath);
    }
    NSLog(@"[AIPlayer] Exhausted all candidate directories -- model not found anywhere");
    return nil;
}

@interface AIOverlayVC : UIViewController
@property (nonatomic, strong) UIButton *toggleButton;
@property (nonatomic, assign) BOOL isPlaying;
@property (nonatomic, assign) NSInteger frameTick;
@property (nonatomic, assign) CFTimeInterval lastInjectTime;
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
            NSLog(@"[AIPlayer] Failed to load model: %@", err);
        } else {
            NSLog(@"[AIPlayer] Model loaded from %@", modelURL);
        }
    } else {
        NSLog(@"[AIPlayer] Model not loaded -- SwipeAnnotator.mlmodelc was not found (see preceding log lines for paths checked)");
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
        CFTimeInterval nowTime = CACurrentMediaTime();
        if (nowTime - self.lastInjectTime < kInjectCooldown) {
            // Still inside the previous swipe's cooldown window — almost
            // certainly the same physical swipe crossing threshold on a
            // second consecutive tick. Log it for visibility but don't
            // double-fire.
            NSLog(@"[AIPlayer] swipe dir=%ld conf=%.2f det=%.2f — suppressed (cooldown)",
                  (long)pred.direction, pred.dirConfidence, pred.detProbability);
            return;
        }
        self.lastInjectTime = nowTime;

        NSLog(@"[AIPlayer] swipe dir=%ld conf=%.2f det=%.2f — injecting",
              (long)pred.direction, pred.dirConfidence, pred.detProbability);
        [self injectSwipeForDirection:pred.direction];
    }
}

- (void)injectSwipeForDirection:(SwipeDirection)dir {
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

    [[TouchInjector sharedInjector] injectSwipeFrom:center to:end duration:kSwipeDuration];

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
