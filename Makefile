ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = AIPlayer

AIPlayer_FILES = AIPlayer.xm
AIPlayer_FRAMEWORKS = UIKit ReplayKit CoreImage QuartzCore CoreML IOKit
AIPlayer_CFLAGS = -fobjc-arc -Wno-unused-parameter

# Stages the compiled model into
# Library/Application Support/AIPlayer/AIPlayer.bundle/ at package time.
# mlmodel_bundle/ must already exist (produced by the CI workflow's
# "Convert checkpoint to Core ML" step) before `make` runs.
#
# MUST be BUNDLE_RESOURCE_DIRS, not RESOURCE_DIRS. Per Theos's own
# instance/tweak.mk, a `tweak`-type target (TWEAK_NAME + include tweak.mk,
# what this Makefile is) only wires up XXX_BUNDLE_RESOURCE_DIRS /
# XXX_BUNDLE_RESOURCE_FILES -- there is no branch anywhere in tweak.mk that
# reads plain XXX_RESOURCE_DIRS. That variable name only does something for
# a `bundle`-type target (BUNDLE_NAME + include bundle.mk), which this
# isn't. Using the un-prefixed name here was a silent no-op: `make`
# succeeds, no error, no warning -- nothing gets copied into the .deb at
# all.
#
# Points at mlmodel_bundle/ (a WRAPPER directory containing
# SwipeAnnotator.mlmodelc), not at SwipeAnnotator.mlmodelc directly.
# XXX_BUNDLE_RESOURCE_DIRS copies a directory's CONTENTS into the bundle,
# flattening it -- it does not preserve the directory itself. Pointing it
# straight at SwipeAnnotator.mlmodelc would spill model.espresso.net /
# .weights / .shape / coremldata.bin loose into AIPlayer.bundle/, destroying
# the .mlmodelc as a loadable unit (Core ML's MLModel loader expects to
# open an actual folder by that name). Flattening the WRAPPER's contents
# instead correctly leaves SwipeAnnotator.mlmodelc intact one level down:
#   mlmodel_bundle/SwipeAnnotator.mlmodelc/...  --(flatten mlmodel_bundle)-->
#   AIPlayer.bundle/SwipeAnnotator.mlmodelc/...
AIPlayer_BUNDLE_RESOURCE_DIRS = mlmodel_bundle

include $(THEOS_MAKE_PATH)/tweak.mk