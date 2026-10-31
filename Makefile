ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

# Plain dylib, injected with Sideloadly ("inject dylib"). No Logos/Substrate:
# AIPlayer.dylib's constructor starts everything when dyld loads it.
#
# The models are NOT linked in. CI packages AIPlayerConfig.json plus the .mlmodelc
# folder(s) it lists into AIPlayer.bundle (AIPlayerBundle.zip); inject that bundle
# alongside the dylib in Sideloadly.
LIBRARY_NAME = AIPlayer

AIPlayer_FILES = AIPlayer.m AIPreprocess.c
AIPlayer_FRAMEWORKS = UIKit ReplayKit CoreImage CoreML CoreMedia CoreVideo QuartzCore ImageIO
AIPlayer_CFLAGS = -fobjc-arc -O2 -Wall -Wno-unused-parameter -Wno-deprecated-declarations

include $(THEOS_MAKE_PATH)/library.mk
