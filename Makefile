default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Ilib/include -ICatch/include -Werror

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -stdlib=libc++
endif

CPP_FILES=$(wildcard test/*.cpp)
TEST_FILES=$(CPP_FILES:test/%=%)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d)

$(eval $(call folder_compile,workbench))
$(eval $(call test_folder_compile,))

# Create executables
$(eval $(call add_executable,test_asm_1,workbench/test.cpp))
$(eval $(call add_executable,test_asm_2,workbench/test_dim.cpp))
$(eval $(call add_executable,benchmark,workbench/benchmark.cpp))
$(eval $(call add_executable,mmul,workbench/mmul.cpp))
$(eval $(call add_test_executable,etl_test,$(TEST_FILES)))

$(eval $(call add_executable_set,etl_test,etl_test))

test_asm: release/bin/test_asm_1 release/bin/test_asm_2

release: release_etl_test release/bin/benchmark
debug: debug_etl_test debug/bin/benchmark

all: release debug

debug_test: debug
	./debug/bin/etl_test

release_test: release
	./release/bin/etl_test

test: all
	./debug/bin/etl_test
	./release/bin/etl_test

benchmark: release/bin/benchmark
	./release/bin/benchmark

cppcheck:
	cppcheck -I include/ --platform=unix64 --suppress=missingIncludeSystem --enable=all --std=c++11 workbench/*.cpp include/etl/*.hpp

v:
	@ echo $(CPP_FILES)
	@ echo $(TEST_FILES)

clean:
	rm -rf release/
	rm -rf debug/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)
