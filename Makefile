ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = AIPlayer

AIPlayer_FILES = AIPlayer.xm
AIPlayer_FRAMEWORKS = UIKit ReplayKit CoreImage QuartzCore CoreML IOKit
AIPlayer_CFLAGS = -fobjc-arc -Wno-unused-parameter

# Copies the whole SwipeAnnotator.mlmodelc DIRECTORY into AIPlayer.bundle at
# package time. This directory must already exist (produced by
# `xcrun coremlcompiler compile` in the CI workflow) before `make` runs.
AIPlayer_RESOURCE_DIRS = SwipeAnnotator.mlmodelc

include $(THEOS_MAKE_PATH)/tweak.mk
