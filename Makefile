# =============================================================================
# Root Makefile — pure SUBPROJECTS aggregator.
#
# This repo now builds TWO dylibs instead of one:
#   TouchSynthesis/  the reusable touch-synthesis engine (HybridTouchSynthesizer,
#                     TouchSynthesisKeyWindow()). No Logos %hook/%group/%init
#                     directives anywhere in it, so it has no MobileSubstrate
#                     dependency — see TouchSynthesis/TouchSynthesis.xm's own
#                     header comment. Must build BEFORE AIPlayer, since
#                     AIPlayer links against its output.
#   AIPlayer/         the AI capture/inference/injection tweak itself. Same
#                     plain-dylib, no-Substrate rationale as before (see
#                     AIPlayer/Makefile) — now imports TouchSynthesis.h and
#                     links against TouchSynthesis.dylib instead of carrying
#                     its own copy of the synthesis engine inline.
#
# SUBPROJECTS builds each listed directory as its own complete, independent
# Theos project via a recursive `make`, IN THE ORDER LISTED, one fully
# finishing before the next starts. That ordering guarantee is exactly why
# TouchSynthesis is listed first: AIPlayer/Makefile's link step reads
# ../.theos/obj/arm64/TouchSynthesis.dylib directly (Theos's SUBPROJECTS
# share ONE .theos/obj/<arch> build directory rooted at THIS file's own
# directory, not a separate .theos tree per subproject -- confirmed
# against an actual build failure, see AIPlayer/Makefile's comment for
# the full explanation), which must already exist by the time AIPlayer
# links.
#
# `make FINALPACKAGE=1 V=1` at this root level (see build.yml) propagates
# both variables into each subproject's recursive make automatically —
# that's standard GNU Make behavior for command-line variable assignments,
# nothing subproject-specific needed here.
# =============================================================================
export ARCHS = arm64
export TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

SUBPROJECTS = TouchSynthesis AIPlayer
include $(THEOS_MAKE_PATH)/aggregate.mk