default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -ICatch/include

CPP_FILES=$(wildcard test/*.cpp)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d)

$(eval $(call test_folder_compile,))

$(eval $(call add_test_executable,etl_test,etl.cpp))

$(eval $(call add_executable_set,etl_test,etl_test))

release: release_etl_test
debug: debug_etl_test

all: release debug
test: all

clean:
	rm -rf release/
	rm -rf debug/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)