ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

# =============================================================================
# Plain dylib target -- NOT a Theos "tweak" (TWEAK_NAME + tweak.mk).
#
# A `tweak` target exists to be loaded by MobileSubstrate/MobileHooker based
# on a Filter, then dpkg-installed on a jailbroken device. Neither applies
# here: the real deployment path is Sideloadly's "inject dylib" option on a
# non-jailbroken device, and AIPlayer.xm contains zero %hook/%new/%ctor
# Logos directives -- it's plain Objective-C with a single
# __attribute__((constructor)) entry point that runs the instant dyld loads
# the dylib into the host process. That's exactly what Sideloadly's
# injection does (adds an LC_LOAD_DYLIB load command, no Substrate
# involved), so a plain `library` target is both correct and sufficient.
# =============================================================================
LIBRARY_NAME = AIPlayer

AIPlayer_FILES = AIPlayer.xm
AIPlayer_FRAMEWORKS = UIKit ReplayKit CoreImage QuartzCore CoreML IOKit
AIPlayer_CFLAGS = -fobjc-arc -Wno-unused-parameter -I../TouchSynthesis

# Touch synthesis now lives in ../TouchSynthesis (built first -- see the
# root Makefile's SUBPROJECTS ordering) instead of being copy-pasted
# inline into this file. Link directly against its build output rather
# than `-lTouchSynthesis`: Theos's library.mk names the product exactly
# "TouchSynthesis.dylib" (no "lib" prefix), which `-l<name>`'s
# lib<name>.dylib lookup convention would not find.
#
# -rpath @loader_path: lets the "@rpath/TouchSynthesis.dylib" dependency
# (recorded via TouchSynthesis/Makefile's own -install_name) resolve
# relative to wherever AIPlayer.dylib itself ends up loaded from on
# device, rather than depending on the host app's own rpath entries --
# safer for a Sideloadly injection, where you control neither. Ship
# TouchSynthesis.dylib in the SAME directory as AIPlayer.dylib for this
# to resolve at launch.
AIPlayer_LDFLAGS = ../TouchSynthesis/.theos/obj/arm64/TouchSynthesis.dylib -Wl,-rpath,@loader_path -install_name @rpath/AIPlayer.dylib

# No XXX_BUNDLE_RESOURCE_DIRS here on purpose. That variable only mattered
# for staging SwipeAnnotator.mlmodelc into a .deb's
# Library/Application Support/AIPlayer/AIPlayer.bundle/ tree -- a path that
# never existed on the actual (non-jailbroken) deployment target. The
# compiled model instead ships as its own separate zip, dropped in Sideloadly
# alongside the dylibs as a sibling file/bundle. See AIPlayer.xm's
# AIPlayerModelURL() for the exact candidate paths it checks at runtime.

include $(THEOS_MAKE_PATH)/library.mk
