default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

WARNINGS_FLAGS += -pedantic

CXX_FLAGS += -Ilib/include -ICatch/include -Werror

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -stdlib=libc++
endif

ifneq (,$(findstring c++-analyzer,$(CXX)))
CXX_FLAGS += -stdlib=libc++
endif

ifneq (,$(ETL_MKL))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)
CXX_FLAGS += -Wno-tautological-compare
else
ifneq (,$(ETL_BLAS))
CXX_FLAGS += -DETL_BLAS_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)
endif
endif

# Enable coverage if not disabled by the user
ifeq (,$(ETL_NO_COVERAGE))
ifneq (,$(findstring clang,$(CXX)))
DEBUG_FLAGS += -fprofile-arcs -ftest-coverage
else
ifneq (,$(findstring g++,$(CXX)))
DEBUG_FLAGS += --coverage
endif
endif
endif

#DEBUG_FLAGS=-fsanitize=address -fsanitize=undefined

CXX_FLAGS += -DETL_VECTORIZE

CPP_FILES=$(wildcard test/*.cpp)
TEST_FILES=$(CPP_FILES:test/%=%)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d)

$(eval $(call folder_compile,workbench,-Wno-error))
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
release_debug: release_debug_etl_test release_debug/bin/benchmark
debug: debug_etl_test debug/bin/benchmark

all: release release_debug debug

debug_test: debug
	./debug/bin/etl_test

release_debug_test: release_debug
	./release_debug/bin/etl_test

release_test: release
	./release/bin/etl_test

test: all
	./debug/bin/etl_test
	./release_debug/bin/etl_test
	./release/bin/etl_test

benchmark: release/bin/benchmark
	./release/bin/benchmark

cppcheck:
	cppcheck -I include/ --platform=unix64 --suppress=missingIncludeSystem --enable=all --std=c++11 workbench/*.cpp include/etl/*.hpp

coverage: debug_test
	lcov -b . --directory debug/test --capture --output-file debug/bin/app.info
	@ mkdir -p reports/coverage
	genhtml --output-directory reports/coverage debug/bin/app.info

coverage_view: coverage
	firefox reports/coverage/index.html

doc:
	doxygen Doxyfile

clean: base_clean
	rm -rf reports
	rm -rf latex/ html/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)
