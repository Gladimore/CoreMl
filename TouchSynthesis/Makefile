ARCHS = arm64
TARGET = iphone:clang:latest:15.0

include $(THEOS)/makefiles/common.mk

# Plain library target, same rationale as AIPlayer/Makefile: this project's
# real deployment is Sideloadly dylib injection on a non-jailbroken device,
# not a Substrate tweak install. TouchSynthesis.xm contains zero Logos
# %hook/%group/%init directives (confirmed — see its own header comment),
# so there is nothing here that needs MobileSubstrate at build or run time.
LIBRARY_NAME = TouchSynthesis

TouchSynthesis_FILES = TouchSynthesis.xm
TouchSynthesis_FRAMEWORKS = UIKit
TouchSynthesis_CFLAGS = -fobjc-arc -Wall -fvisibility=default

# -exported_symbols_list: without this, a release/FINALPACKAGE build can
# strip HybridTouchSynthesizer's symbols entirely, leaving AIPlayer.dylib
# with unresolved symbols at link time (or worse, at dyld load time on
# device if the strip happens post-link). -install_name @rpath/...: so
# AIPlayer.dylib's own recorded dependency reads "@rpath/TouchSynthesis.dylib"
# rather than this build machine's absolute path — resolvable on-device as
# long as TouchSynthesis.dylib ships in the same directory as AIPlayer.dylib
# inside the app bundle (see AIPlayer/Makefile's -rpath @loader_path).
TouchSynthesis_LDFLAGS = -Wl,-exported_symbols_list,touchsynthesis-exports.txt -install_name @rpath/TouchSynthesis.dylib

include $(THEOS_MAKE_PATH)/library.mk
