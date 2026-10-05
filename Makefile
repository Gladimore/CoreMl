ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

# Plain dylib, injected with Sideloadly ("inject dylib"). There is no Logos,
# MobileSubstrate or %hook anywhere, so the source is a plain .m and Theos does
# not run it through Logos. AIPlayer.dylib's constructor starts everything
# when dyld loads it; see AIPlayer.m.
#
# The two compiled models (SwipeEncoder.mlmodelc, SwipeHead.mlmodelc) are NOT
# bundled: they ship as a separate zip (see the CI workflow) and are injected
# alongside the dylib. AIPlayer.m searches the dylib's directory and the app
# bundle for them.
LIBRARY_NAME = AIPlayer

AIPlayer_FILES = AIPlayer.m
AIPlayer_FRAMEWORKS = UIKit ReplayKit CoreImage CoreML CoreMedia CoreVideo QuartzCore
AIPlayer_CFLAGS = -fobjc-arc -Wall -Wno-unused-parameter -Wno-deprecated-declarations

include $(THEOS_MAKE_PATH)/library.mk
