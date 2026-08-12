// ===========================================================
// TouchSynthesis.xm
//
// Standalone dylib version of the touch-synthesis engine. This is the
// SAME engine/logic as the original combined tweak — every method and
// every fix are preserved. Two things are deliberately NOT here because
// they belong at the consumer level, not in a generic reusable module:
//   - FloatingSwipeButtonManager and the UIWindow -makeKeyAndVisible
//     hook that showed its overlay button (see FloatingSwipeExample/Tweak.xm).
//   - The UnityView diagnostic touch-logging hook (Unity-specific, and
//     depends on how a given consumer is allowed to hook — see the note
//     above GG_TouchSynthesisBuildMarker further down for why, and
//     AIPlayer.xm for a working non-jailbroken-safe implementation of it).
//
// Public API is declared in TouchSynthesis.h. Any other tweak can link
// against this dylib and call:
//   TouchSynthesisKeyWindow()
//   [[HybridTouchSynthesizer sharedInstance] dispatchHybridTouchAtPoint:phase:inOutTouch:]
//   [[HybridTouchSynthesizer sharedInstance] finalizeAnyActiveTouch:]
//   ...etc (see header)
//
// See the original combined file's changelog for the full fix history
// (flags mapping, GSEventProxy offsets, associated-object fallback,
// _tapCount/_touchFlags initialization, event-creation-failure fallback,
// gesture-recognizer safety gating, the empirical Layer-A-unconditional
// fix, etc.) — all of that logic is unchanged here, just relocated.
// ===========================================================

#import "TouchSynthesis.h"
#import <objc/runtime.h>
#import <math.h>
#import <os/log.h>

// ---------------------------------------------------------
// LOGGING — os_log (not NSLog: iOS 26 redacts %@ args under NSLog almost
// unconditionally; os_log with an explicit handle is the supported way
// to get %{public}@ honored). Initialized via constructor so gg_log is
// valid before any +load method in this file can fire.
// ---------------------------------------------------------
static os_log_t gg_log;

__attribute__((constructor))
static void GG_InitLog(void) {
    gg_log = os_log_create("com.gg.touchsynthesis", "touch");
}

// ---------------------------------------------------------
// Public key-window helper (declared in TouchSynthesis.h).
// -[UIApplication keyWindow] is deprecated (iOS 13+) and ignores
// multi-scene apps. This walks connectedScenes for the foreground-active
// UIWindowScene's key window, falling back to the deprecated accessor
// only pre-iOS 13 or if no active scene is found yet.
// ---------------------------------------------------------
UIWindow *TouchSynthesisKeyWindow(void) {
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

// Memory layout mimicking Apple's internal GSEvent struct. Required to
// prevent _initWithEvent:touches: from crashing. Offsets confirmed
// against __UIEvent_Synthesize__initWithTouch__ disassembly: flags(0x08),
// type(0x0C), x1/y1/x2/y2(0x14-0x23), sizeX/sizeY(0x68/0x6C), x3/y3(0x70/0x74).
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
// Associated-object shadow storage + swizzled -window/-view.
// The binary unconditionally stashes the intended window/view via
// objc_setAssociatedObject BEFORE attempting any ivar write, and
// swizzles UITouch's -window/-view at +load so that if the real
// accessor ever returns nil, it falls back to the associated object.
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

        os_log(gg_log, "[Touch]: TouchSynthesis dylib loaded, -window/-view swizzle %{public}@ (window: %{public}@, view: %{public}@)",
              (originalWindow && swizzledWindow && originalView && swizzledView) ? @"installed" : @"FAILED — check selector names",
              originalWindow ? @"ok" : @"MISSING",
              originalView ? @"ok" : @"MISSING");
    });
}

// After the +load swap, sending -gg_touchSynthesis_window to self
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
// SYNTHESIS ENGINE
// ---------------------------------------------------------
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
        *(NSInteger *)ivarMemory = *(NSInteger *)valuePtr;
    }
}

// Companion reader — captures a touch's location BEFORE overwriting it,
// to use as the true "previous" point (needed for GSEventProxy's x2/y2).
- (CGPoint)readCGPointIvarOnObject:(id)object name:(const char *)name fallback:(CGPoint)fallback {
    Ivar ivar = class_getInstanceVariable([object class], name);
    if (!ivar) return fallback;
    ptrdiff_t offset = ivar_getOffset(ivar);
    void *ivarMemory = (uint8_t *)(__bridge void *)object + offset;
    return *(CGPoint *)ivarMemory;
}

// Raw pointer helper for _touchFlags (a 16-bit bitfield needing OR/AND
// bit manipulation, not a full-width overwrite). Bits 0x1/0x2 are set on
// every new touch; bit 0x2 clears when a location update moves the touch
// more than 2.0pt in x or y.
- (void *)rawIvarPointerOnObject:(id)object name:(const char *)name {
    Ivar ivar = class_getInstanceVariable([object class], name);
    if (!ivar) return NULL;
    ptrdiff_t offset = ivar_getOffset(ivar);
    return (uint8_t *)(__bridge void *)object + offset;
}

// 2. Defensive fallback chain for OBJECTS ONLY (_window, _view)
- (void)defensiveSetObject:(id)value forProperty:(NSString *)propName onObject:(id)target {
    // Store the associated-object shadow copy first, unconditionally.
    if ([propName isEqualToString:@"window"]) {
        objc_setAssociatedObject(target, kTouchSynthesisFallbackWindowKey, value, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    } else if ([propName isEqualToString:@"view"]) {
        objc_setAssociatedObject(target, kTouchSynthesisFallbackViewKey, value, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }

    NSString *ivarName = [NSString stringWithFormat:@"_%@", propName];
    Ivar ivar = class_getInstanceVariable([target class], [ivarName UTF8String]);

    // Attempt A: raw runtime ivar assignment
    if (ivar) {
        object_setIvar(target, ivar, value);
    } else {
        os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ — no ivar named %{public}@ found on %{public}@", propName, ivarName, [target class]);
    }

    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Warc-performSelector-leaks"
    id readBack = [target performSelector:NSSelectorFromString(propName)];
    if (readBack == value) {
        os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ = %{public}@ succeeded via raw ivar", propName, value);
        return;
    }

    // Attempt B: private setter (_setWindow: / _setView:)
    NSString *privateSelectorString = [NSString stringWithFormat:@"_set%@:", [propName capitalizedString]];
    SEL privateSelector = NSSelectorFromString(privateSelectorString);
    if ([target respondsToSelector:privateSelector]) {
        [target performSelector:privateSelector withObject:value];
        if ([target performSelector:NSSelectorFromString(propName)] == value) {
            os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ = %{public}@ succeeded via private setter", propName, value);
            return;
        }
    }

    // Attempt C: public setter (setWindow: / setView:)
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

    // Attempt D: KVC fallback of last resort (uses the PUBLIC key).
    [target setValue:value forKey:propName];
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Warc-performSelector-leaks"
    os_log(gg_log, "[Touch]: defensiveSetObject %{public}@ = %{public}@ — raw ivar/private/public setters all failed, used KVC fallback (readback now: %{public}@)",
          propName, value, [target performSelector:NSSelectorFromString(propName)]);
    #pragma clang diagnostic pop
}

// Shared event builder, used by both the normal per-event dispatch and
// -finalizeAnyActiveTouch: so Layer-B event construction stays identical.
- (UIEvent *)buildSyntheticEventForTouch:(UITouch *)touch atPoint:(CGPoint)point previousPoint:(CGPoint)previousPoint phase:(UITouchPhase)phase {
    GSEventProxy *gsProxy = [[GSEventProxy alloc] init];
    gsProxy->x1 = point.x;         gsProxy->y1 = point.y;
    // x1/y1 = locationInView:, x2/y2 = previousLocationInView: (confirmed
    // via disassembly of -[UIEvent(Synthesize) _initWithTouch:] — not a
    // duplicate of the current point).
    gsProxy->x2 = previousPoint.x; gsProxy->y2 = previousPoint.y;
    gsProxy->x3 = point.x;         gsProxy->y3 = point.y;
    gsProxy->sizeX = 1.0;          gsProxy->sizeY = 1.0;

    // Began(0)/Moved(1)/Stationary(2) -> 0x3010180 override.
    // Ended(3)/Cancelled(4) -> 0x1010180 default. type = 3001, constant.
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
        // Ground truth writes _timestamp onto the constructed EVENT object
        // itself immediately after _initWithEvent:touches: returns non-nil.
        NSTimeInterval eventTimestamp = [[NSProcessInfo processInfo] systemUptime];
        [self writeScalarIvarOnObject:syntheticEvent name:"_timestamp" type:@encode(NSTimeInterval) valuePtr:&eventTimestamp];
    }
    return syntheticEvent;
}

// Send to the touch's window directly, UIApplication only as a nil-window
// fallback. Applies the exclusion-toggle behavior: never skip the send,
// just briefly disable userInteractionEnabled on the excluded view when
// it currently owns the dispatch window and is interactive.
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

// 3. The core per-event dispatcher.
- (void)dispatchHybridTouchAtPoint:(CGPoint)point phase:(UITouchPhase)phase inOutTouch:(UITouch **)activeTouch {
    os_log(gg_log, "[Touch]: dispatchHybridTouchAtPoint (%.1f, %.1f) phase=%ld", point.x, point.y, (long)phase);
    UIWindow *keyWindow = TouchSynthesisKeyWindow();
    if (!keyWindow) {
        os_log(gg_log, "[Touch]: dispatchHybridTouchAtPoint aborted — TouchSynthesisKeyWindow() returned nil");
        return;
    }

    UITouch *touch = *activeTouch;
    BOOL isNewTouch = (!touch || phase == UITouchPhaseBegan);

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

    UIEvent *syntheticEvent = [self buildSyntheticEventForTouch:touch atPoint:point previousPoint:previousPoint phase:phase];

    UIWindow *dispatchWindow = touch.window ?: keyWindow;
    os_log(gg_log, "[Touch]: touch.window=%{public}@ touch.view=%{public}@ syntheticEvent=%{public}@ dispatchWindow=%{public}@",
          touch.window, touch.view, syntheticEvent ? @"built OK" : @"NIL", dispatchWindow);

    if (syntheticEvent) {
        [self sendSyntheticEvent:syntheticEvent toWindow:dispatchWindow];
    } else {
        os_log(gg_log, "[Touch]: syntheticEvent was nil — routing to performEventCreationFailureFallbackForTouch (Layer A)");
    }

    // Layer A is called UNCONDITIONALLY here (empirical fix — see the
    // original changelog's "SEVENTH PASS" note): sendEvent: was observed
    // silently dropping well-formed synthetic events on-device even
    // though event construction always succeeded. Direct
    // touchesBegan/Moved/Ended:withEvent: calls are the one path
    // confirmed to actually reach the target view's handlers.
    // KNOWN RISK: if sendEvent: DOES work correctly on some other
    // device/iOS build, this double-fires touchesBegan/Moved/Ended for
    // the same touch. If you see duplicate delivery, gate this back to
    // an else-branch (only on nil event, or a delivery-confirmation
    // timeout) instead of unconditional.
    [self performEventCreationFailureFallbackForTouch:touch targetView:touch.view event:syntheticEvent phase:phase];

    if (phase == UITouchPhaseEnded || phase == UITouchPhaseCancelled) {
        *activeTouch = nil;
    }
}

// 3b. Event-creation-failure fallback / Layer A. Confirmed exact
// phase->selector mapping: 0->Began, 1->Moved, 2->no-op, 3->Ended,
// 4->Cancelled. The (possibly-nil) event is passed straight through as
// the withEvent: argument in every case.
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
            // Confirmed no-op in the binary.
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

// Session-teardown cleanup path (distinct call site from the per-event
// dispatcher above). Call this whenever a swipe/replay sequence is
// interrupted mid-flight so the app doesn't end up thinking a finger is
// still down.
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

    // Layer B first.
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

// 4b. UIControl/cell action dispatcher decision logic.
- (NSInteger)actionCodeForTouch:(UITouch *)touch
                candidateControl:(UIView *)candidate
                 boundsContainsPoint:(BOOL)boundsContainsPoint
                 alreadyConsumedFlag:(BOOL)alreadyConsumedFlag
                   allTargetsPresent:(BOOL)allTargetsPresent
                hasSecondaryTargets:(BOOL)hasSecondaryTargets
                   cellAncestorFound:(BOOL)cellAncestorFound
                        isKeyWindow:(BOOL)isKeyWindow
                       touchFlagsBit1:(BOOL)touchFlagsBit1 {
    // Gate 1: re-entrancy guard, not a match requirement.
    if (alreadyConsumedFlag) return 0;
    // Gate 2: touch point must be inside target bounds.
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

// Bounds/cache-gate helper matching the semantics above, for callers
// that just need the pass/fail without the full 0-4 action code.
- (BOOL)shouldAllowControlDispatchForTouch:(UITouch *)touch onView:(UIView *)targetView {
    if (!targetView) return NO;
    CGPoint pointInTarget = [touch locationInView:targetView];
    if (!CGRectContainsPoint(targetView.bounds, pointInTarget)) {
        return NO;
    }
    return YES;
}

// 5. Gesture-recognizer fallback (Layer C) — fires target/action
// directly, bypassing sendEvent:, gated by a bundle-ownership safety
// check. Not called automatically anywhere — kept as a standalone,
// correctly-gated method you can wire up explicitly.
- (BOOL)classNameLooksUnsafeForGestureFallback:(NSString *)className {
    if ([className hasPrefix:@"WK"]) return YES;
    if ([className hasPrefix:@"_WK"]) return YES;
    if ([className hasPrefix:@"Web"]) return YES;
    if ([className containsString:@"WebKit"]) return YES;
    return NO;
}

- (BOOL)isSafeGestureFallbackTargetEntry:(id)target {
    if (!target) return NO;
    NSBundle *targetBundle = [NSBundle bundleForClass:[target class]];
    if (targetBundle == [NSBundle mainBundle]) return YES;
    NSString *mainPath = [[NSBundle mainBundle] bundlePath];
    NSString *targetPath = targetBundle.bundlePath;
    if (targetPath.length > 0 && mainPath.length > 0 && [targetPath hasPrefix:mainPath]) {
        return YES; // app-owned via path containment
    }

    NSString *className = NSStringFromClass([target class]);
    if ([self classNameLooksUnsafeForGestureFallback:className]) return NO;

    if (targetPath.length == 0) return NO;
    if ([targetPath hasPrefix:@"/System/Library"]) return NO;
    if ([targetPath containsString:@"PrivateFrameworks"]) return NO;
    return YES;
}

- (BOOL)isSafeGestureFallbackTarget:(id)target {
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
    id target = [entry valueForKey:@"_target"];
    if (!target) target = [entry valueForKey:@"target"];
    return target;
}

- (SEL)extractGestureAction:(id)entry {
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
        if (![self isSafeGestureFallbackTarget:recognizer]) continue;

        NSArray *targets = nil;
        @try {
            id value = [recognizer valueForKey:@"_targets"];
            if ([value isKindOfClass:[NSArray class]]) targets = value;
        } @catch (__unused NSException *exception) {
            targets = nil;
        }
        if (!targets) targets = @[];

        // All-or-nothing per recognizer: if ANY valid target/action entry
        // fails the per-entry safety check, abort the whole recognizer.
        NSMutableArray *validEntries = [NSMutableArray array];
        BOOL allSafe = YES;
        for (id targetEntry in targets) {
            id targetObj = [self extractGestureTarget:targetEntry];
            SEL action = [self extractGestureAction:targetEntry];
            if (!targetObj || !action) continue;
            if (![targetObj respondsToSelector:action]) continue;
            if (![self isSafeGestureFallbackTargetEntry:targetObj]) {
                allSafe = NO;
                break;
            }
            [validEntries addObject:@[targetObj, [NSValue valueWithPointer:action]]];
        }
        if (!allSafe) continue;
        if (validEntries.count == 0) continue;

        // Set state ONCE, fire ALL entries, restore ONCE.
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
// NOTE ON WHAT'S DELIBERATELY *NOT* HERE:
//
// An earlier revision of this module included a Logos %hook/%group/%init
// block that swizzled UnityView's touchesBegan/Moved/Ended/Cancelled for
// diagnostic logging. It has been REMOVED, for two reasons:
//
// 1. Logos %hook compiles to a call through MSHookMessageEx, which needs
//    MobileSubstrate (or an ABI-compatible equivalent — Substitute,
//    ElleKit) present on-device at runtime, and also emits a
//    `.linker_option "-framework CydiaSubstrate"` directive at BUILD
//    time regardless of Makefile target type. Any consumer of this
//    dylib that deploys non-jailbroken (e.g. Sideloadly + dylib
//    injection, no MobileSubstrate on device) would fail to link and/or
//    fail to resolve that hook at load time. Since this module is meant
//    to be linkable from either a jailbroken Substrate-tweak consumer
//    (see FloatingSwipeExample) or a non-jailbroken one (see
//    AIPlayer.xm), it cannot itself depend on Substrate anywhere.
// 2. UnityView is app/engine-specific (only relevant if your target is a
//    Unity game), so it doesn't belong in a generic synthesis module
//    regardless of the above. If you need this diagnostic, add it in
//    YOUR consuming tweak using method_setImplementation-based manual
//    swizzling (no Substrate required) — see AIPlayer.xm's
//    GG_SwizzleClassMethod + GG_TryInstallUnityViewDiagnosticHook for a
//    working, non-jailbroken-safe reference implementation of exactly
//    this pattern, applied at the consumer level instead of here.
//
// Everything above this note (HybridTouchSynthesizer itself, the
// UITouch window/view fallback swizzle, GSEventProxy) uses only
// method_exchangeImplementations/method_setImplementation — plain
// Objective-C runtime calls that need nothing beyond libobjc, which is
// always present. Nothing in this file requires MobileSubstrate.
// ---------------------------------------------------------

__attribute__((constructor))
static void GG_TouchSynthesisBuildMarker(void) {
    // Logs unconditionally, first thing, with the compile-time date/time
    // baked in via __DATE__/__TIME__. If this line (or a fresher
    // timestamp than expected) never shows up after a rebuild+reinstall,
    // the running binary is stale.
    os_log(gg_log, "[Touch]: ===== TouchSynthesis BUILD MARKER: compiled %{public}s %{public}s =====", __DATE__, __TIME__);
}
