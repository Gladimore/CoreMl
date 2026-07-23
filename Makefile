ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

# =============================================================================
# Plain dylib target -- NOT a Theos "tweak" (TWEAK_NAME + tweak.mk).
#
# A `tweak` target exists to be loaded by MobileSubstrate/MobileHooker based
# on AIPlayer.plist's Filter, then dpkg-installed on a jailbroken device.
# Neither of those applies to this project: the real deployment path is
# Sideloadly's "inject dylib" option on a non-jailbroken device, and
# AIPlayer.xm contains zero %hook/%new/%ctor Logos directives -- it's plain
# Objective-C with a single __attribute__((constructor)) entry point that
# runs the instant dyld loads the dylib into the host process. That's
# exactly what Sideloadly's injection does (adds an LC_LOAD_DYLIB load
# command, no Substrate involved), so a plain `library` target is both
# correct and sufficient. control and AIPlayer.plist are no longer used by
# anything in this build and can be deleted from the repo.
# =============================================================================
LIBRARY_NAME = AIPlayer

AIPlayer_FILES = AIPlayer.xm
AIPlayer_FRAMEWORKS = UIKit ReplayKit CoreImage QuartzCore CoreML IOKit
AIPlayer_CFLAGS = -fobjc-arc -Wno-unused-parameter

# No XXX_BUNDLE_RESOURCE_DIRS here on purpose. That variable only mattered
# for staging SwipeAnnotator.mlmodelc into a .deb's
# Library/Application Support/AIPlayer/AIPlayer.bundle/ tree -- a path that
# never existed on the actual (non-jailbroken) deployment target. The
# compiled model instead ships as its own separate zip, dropped in Sideloadly
# alongside the dylib as a sibling file/bundle. See AIPlayer.xm's
# AIPlayerModelURL() for the exact candidate paths it checks at runtime.

include $(THEOS_MAKE_PATH)/library.mk