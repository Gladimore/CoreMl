ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = AIPlayer

AIPlayer_FILES = AIPlayer.xm
AIPlayer_FRAMEWORKS = UIKit ReplayKit CoreImage QuartzCore CoreML IOKit
AIPlayer_CFLAGS = -fobjc-arc -Wno-unused-parameter

# Stages the compiled model into Library/Application Support/AIPlayer.bundle/
# at package time. This directory must already exist (produced by
# `xcrun coremlcompiler compile` in the CI workflow) before `make` runs.
#
# MUST be BUNDLE_RESOURCE_DIRS, not RESOURCE_DIRS. Per Theos's own
# instance/tweak.mk, a `tweak`-type target (TWEAK_NAME + include tweak.mk,
# what this Makefile is) only wires up XXX_BUNDLE_RESOURCE_DIRS /
# XXX_BUNDLE_RESOURCE_FILES -- there is no branch anywhere in tweak.mk that
# reads plain XXX_RESOURCE_DIRS. That variable name only does something for
# a `bundle`-type target (BUNDLE_NAME + include bundle.mk), which this
# isn't. Using the un-prefixed name here was a silent no-op: `make`
# succeeds, no error, no warning -- the .mlmodelc just never gets copied
# into the .deb at all, so the tweak has nothing to load at runtime no
# matter how the lookup code searches for it on-device.
AIPlayer_BUNDLE_RESOURCE_DIRS = SwipeAnnotator.mlmodelc

include $(THEOS_MAKE_PATH)/tweak.mk