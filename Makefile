default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -ICatch/include

CPP_FILES=$(wildcard test/*.cpp)
TEST_FILES=$(CPP_FILES:test/%=%)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d)

$(eval $(call test_folder_compile,))

$(eval $(call add_test_executable,etl_test,$(TEST_FILES)))

$(eval $(call add_executable_set,etl_test,etl_test))

release: release_etl_test
debug: debug_etl_test

all: release debug
test: all

v:
	@ echo $(CPP_FILES)
	@ echo $(TEST_FILES)

clean:
	rm -rf release/
	rm -rf debug/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)