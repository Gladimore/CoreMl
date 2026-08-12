// ===========================================================
// TouchSynthesis.h
// Public interface for the TouchSynthesis dylib.
// Import this from any other tweak that wants to synthesize touches.
// ===========================================================
#ifndef TOUCHSYNTHESIS_H
#define TOUCHSYNTHESIS_H

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

// Returns the current key window, correctly handling multi-scene apps
// (iOS 13+) by walking connectedScenes for the foreground-active
// UIWindowScene's key window. Falls back to the deprecated
// UIApplication.keyWindow accessor only pre-iOS 13 or if no active scene
// can be found yet (e.g. very early app launch).
FOUNDATION_EXPORT UIWindow * _Nullable TouchSynthesisKeyWindow(void);

__attribute__((visibility("default")))
@interface HybridTouchSynthesizer : NSObject

// View that should have touches routed around it (e.g. your own overlay
// button) so a synthetic swipe never lands on the button itself. The
// synthesizer briefly disables userInteractionEnabled on this view for
// the duration of any dispatch whose window matches it.
@property (nonatomic, strong, nullable) UIView *excludedTouchView;

// Simplified single-instance cache of "the control we currently believe
// the active touch is over," consulted by the optional UIControl/cell
// gating helpers below (-actionCodeForTouch:... / -shouldAllowControlDispatchForTouch:onView:).
@property (nonatomic, weak, nullable) UIControl *currentControl;

+ (instancetype)sharedInstance;

// ---------------------------------------------------------
// Core dispatch
// ---------------------------------------------------------

// Call once per touch phase (Began, then Moved N times, then Ended or
// Cancelled). `activeTouch` is an in/out UITouch* pointer:
//   - On UITouchPhaseBegan, pass a pointer to a nil-initialized UITouch*
//     (typically an ivar on your caller).
//   - On every subsequent call for the SAME touch, pass a pointer to that
//     same variable — this method reads the previously-created UITouch
//     back out of it and updates it in place.
//   - On UITouchPhaseEnded/Cancelled, the pointed-to variable is cleared
//     back to nil automatically after dispatch.
- (void)dispatchHybridTouchAtPoint:(CGPoint)point
                              phase:(UITouchPhase)phase
                         inOutTouch:(UITouch * _Nullable * _Nonnull)activeTouch;

// Call this if a swipe/replay sequence is interrupted or torn down
// mid-flight (e.g. -dealloc on whatever object was driving it, or an
// app backgrounding event) so the app isn't left believing a finger is
// still down. Dispatches the touch's current phase one final time via
// both delivery layers, then clears activeTouch back to nil.
- (void)finalizeAnyActiveTouch:(UITouch * _Nullable * _Nonnull)activeTouch;

// ---------------------------------------------------------
// Optional Layer C — gesture recognizer fallback
// ---------------------------------------------------------

// Fires a view's gesture recognizers' target/action pairs directly,
// bypassing sendEvent: entirely. Gated by an app-bundle ownership safety
// check on both the recognizer and each individual target (rejects
// WebKit/system-framework targets). NOT called automatically by
// -dispatchHybridTouchAtPoint:phase:inOutTouch: — wire it up yourself
// if a particular view needs this fallback.
- (void)performGestureRecognizerFallbackOnView:(UIView *)view;

// ---------------------------------------------------------
// Optional UIControl/cell dispatch helpers
// ---------------------------------------------------------

// Bounds-check helper: does the touch currently land inside targetView's
// bounds (in targetView's own coordinate space)? Useful before deciding
// whether to fire a UIControl action fallback yourself.
- (BOOL)shouldAllowControlDispatchForTouch:(UITouch *)touch
                                     onView:(nullable UIView *)targetView;

// Full 0-4 action-code decision matching the underlying gesture/control
// dispatch gate. See TouchSynthesis.xm for the meaning of each input and
// each returned code — exposed here in case a consuming tweak wants to
// replicate the exact control/cell-tap gating logic itself.
- (NSInteger)actionCodeForTouch:(UITouch *)touch
                candidateControl:(nullable UIView *)candidate
             boundsContainsPoint:(BOOL)boundsContainsPoint
             alreadyConsumedFlag:(BOOL)alreadyConsumedFlag
               allTargetsPresent:(BOOL)allTargetsPresent
              hasSecondaryTargets:(BOOL)hasSecondaryTargets
                cellAncestorFound:(BOOL)cellAncestorFound
                      isKeyWindow:(BOOL)isKeyWindow
                   touchFlagsBit1:(BOOL)touchFlagsBit1;

@end

NS_ASSUME_NONNULL_END

#endif
