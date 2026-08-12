ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

# =============================================================================
# Two independent library instances, ONE Makefile -- NOT Theos SUBPROJECTS.
#
# An earlier revision of this repo used SUBPROJECTS (a root Makefile that
# only orchestrates, plus a separate Makefile in TouchSynthesis/ and
# AIPlayer/ each). That hit two real problems in CI:
#   1. Wrong assumption about where build output lands -- Theos's
#      SUBPROJECTS mechanism shares ONE .theos/obj/<arch> directory rooted
#      at the top-level project, not a separate .theos tree per subproject
#      directory (confirmed against a real linker error).
#   2. After fixing #1, a second failure: while ostensibly building "in
#      TouchSynthesis" (per SUBPROJECTS' own per-directory log line),
#      Theos's internal per-instance dispatch (which re-invokes the
#      current Makefile via `-f` with a target like
#      "AIPlayer.all.library.variables" to build a specific named
#      instance -- see master/library.mk) somehow resolved to the WRONG
#      instance/directory, looking for AIPlayer.xm while cwd was
#      TouchSynthesis/. Neither TouchSynthesis/Makefile's content nor its
#      hash changed between the run where this worked and the run where
#      it didn't, which points at Theos's own subproject/instance-dispatch
#      interaction (possibly compounded by the Theos-core cache's
#      restore-keys prefix fallback) rather than anything wrong in what
#      was written here -- but rather than keep guessing at Theos
#      internals blind, this switches to the simpler, far more common
#      pattern instead of continuing to debug the nested-directory one.
#
# This pattern -- one Makefile, `LIBRARY_NAME = A B`, per-instance
# `A_FILES`/`B_FILES` etc. -- is what most real multi-product Theos
# projects actually use (the same structure as `TWEAK_NAME = foo bar`
# tweaks-with-a-preferences-bundle examples throughout Theos's own docs
# and the wild), and avoids cross-directory recursion entirely: there is
# only one Makefile, read once, in one directory (the repo root). Source
# files stay organized in TouchSynthesis/ and AIPlayer/ via relative
# paths in each instance's _FILES variable; build output for both lands
# in this repo root's .theos/obj/arm64/ (confirmed Theos behavior -- see
# build.yml's "Locate build output" step).
#
# Order matters: TouchSynthesis is listed first because AIPlayer links
# against its output. Theos's own per-instance loop in master/library.mk
# processes $(LIBRARY_NAME) sequentially in listed order (same structural
# pattern as SUBPROJECTS' own loop) -- and this CI runner's GNU Make
# doesn't have jobserver-based parallelism active anyway, per its own
# "Build may be slow... isn't using all available CPU cores" notice, so
# there's no ordering ambiguity to worry about here.
# =============================================================================
LIBRARY_NAME = TouchSynthesis AIPlayer

# ── TouchSynthesis: the reusable touch-synthesis engine ─────────────────────
# No Logos %hook/%group/%init directives anywhere in it (confirmed -- see
# TouchSynthesis/TouchSynthesis.xm's own header comment), so no
# MobileSubstrate dependency at build or run time.
TouchSynthesis_FILES = TouchSynthesis/TouchSynthesis.xm
TouchSynthesis_FRAMEWORKS = UIKit
TouchSynthesis_CFLAGS = -fobjc-arc -Wall -fvisibility=default

# -exported_symbols_list: without this, a release/FINALPACKAGE build can
# strip HybridTouchSynthesizer's symbols entirely, leaving AIPlayer.dylib
# with unresolved symbols at link time. -install_name @rpath/...: so
# AIPlayer.dylib's own recorded dependency reads "@rpath/TouchSynthesis.dylib"
# rather than this build machine's absolute path -- resolvable on-device
# as long as TouchSynthesis.dylib ships in the same directory as
# AIPlayer.dylib inside the app bundle (see AIPlayer_LDFLAGS below).
TouchSynthesis_LDFLAGS = -Wl,-exported_symbols_list,TouchSynthesis/touchsynthesis-exports.txt -install_name @rpath/TouchSynthesis.dylib

# ── AIPlayer: the AI capture/inference/injection tweak itself ───────────────
# Plain dylib target, not a Theos "tweak" -- same rationale as before: real
# deployment is Sideloadly's "inject dylib" option on a non-jailbroken
# device, and AIPlayer.xm contains zero %hook/%new/%ctor Logos directives.
AIPlayer_FILES = AIPlayer/AIPlayer.xm
AIPlayer_FRAMEWORKS = UIKit ReplayKit CoreImage QuartzCore CoreML IOKit
AIPlayer_CFLAGS = -fobjc-arc -Wno-unused-parameter -ITouchSynthesis

# Touch synthesis now lives in TouchSynthesis/ (built first -- see
# LIBRARY_NAME ordering above) instead of being copy-pasted inline into
# AIPlayer.xm. Link directly against the built dylib's path rather than
# `-lTouchSynthesis`: Theos's library.mk names the product exactly
# "TouchSynthesis.dylib" (no "lib" prefix), which `-l<name>`'s
# lib<name>.dylib lookup convention would not find. Both instances' output
# lands in the SAME .theos/obj/arm64/ directory (see the note above), so
# this is a same-directory reference, not a relative ../ path.
#
# -rpath @loader_path: lets the "@rpath/TouchSynthesis.dylib" dependency
# resolve relative to wherever AIPlayer.dylib itself ends up loaded from
# on device, rather than depending on the host app's own rpath entries --
# safer for a Sideloadly injection, where you control neither. Ship
# TouchSynthesis.dylib in the SAME directory as AIPlayer.dylib inside the
# IPA for this to resolve at launch.
AIPlayer_LDFLAGS = .theos/obj/arm64/TouchSynthesis.dylib -Wl,-rpath,@loader_path -install_name @rpath/AIPlayer.dylib

# No XXX_BUNDLE_RESOURCE_DIRS here on purpose. That variable only mattered
# for staging SwipeAnnotator.mlmodelc into a .deb's
# Library/Application Support/AIPlayer/AIPlayer.bundle/ tree -- a path that
# never existed on the actual (non-jailbroken) deployment target. The
# compiled model instead ships as its own separate zip, dropped in
# Sideloadly alongside the dylibs as a sibling file/bundle. See
# AIPlayer.xm's AIPlayerModelURL() for the exact candidate paths it checks
# at runtime.

include $(THEOS_MAKE_PATH)/library.mk
