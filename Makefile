default: release

.PHONY: default release debug all clean test debug_test release_debug_test release_test
.PHONY: valgrind_test benchmark cppcheck coverage coverage_view format modernize tidy tidy_all doc
.PHONY: full_bench

BLAS_PKG = mkl

include make-utils/flags.mk
include make-utils/cpp-utils.mk

# Build with libc++ if configured
ifneq (,$(CLANG_LIBCXX))
$(eval $(call use_libcxx))
endif

# Be stricter
CXX_FLAGS += -pedantic -Werror -Winvalid-pch

# Add includes
CXX_FLAGS += -Ilib/include -ICatch/include -Itest/include

ifneq (,$(ETL_MKL))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags $(BLAS_PKG))
LD_FLAGS += $(shell pkg-config --libs $(BLAS_PKG))

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif

else
ifneq (,$(ETL_BLAS))
CXX_FLAGS += -DETL_BLAS_MODE $(shell pkg-config --cflags $(BLAS_PKG))
LD_FLAGS += $(shell pkg-config --libs $(BLAS_PKG))
endif
endif

ifneq (,$(ETL_CUBLAS))
CXX_FLAGS += -DETL_CUBLAS_MODE $(shell pkg-config --cflags cublas)
LD_FLAGS += $(shell pkg-config --libs cublas)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

LD_FLAGS += -pthread

# Enable coverage if not disabled by the user
ifeq (,$(ETL_NO_COVERAGE))
$(eval $(call enable_coverage))
endif

# Enable Clang sanitizers in debug mode
ifneq (,$(findstring clang,$(CXX)))
ifeq (,$(ETL_CUBLAS))
DEBUG_FLAGS += -fsanitize=address,undefined
endif
endif

# Enable advanced vectorization for release modes (unset by the benchmark)
ifeq (,$(ETL_NO_DEFAULT))
RELEASE_FLAGS 		+= -DETL_VECTORIZE_FULL
RELEASE_DEBUG_FLAGS += -DETL_VECTORIZE_FULL
endif

# Enable configurable default options (set by the benchmark)
ifneq (,$(ETL_DEFAULTS))
RELEASE_FLAGS 		+= $(ETL_DEFAULTS)
RELEASE_DEBUG_FLAGS += $(ETL_DEFAULTS)
endif

# Add support for precompiler headers for GCC
ifeq (,$(findstring clang,$(CXX)))
$(eval $(call precompile_init,test/include))
$(eval $(call precompile_header,test/include,test.hpp))
$(eval $(call precompile_header,test/include,test_light.hpp))
$(eval $(call precompile_finalize))
endif

# Compile folders
$(eval $(call auto_folder_compile,workbench,-Icpm/include))
$(eval $(call auto_folder_compile,test/src))

# Collect files for the test executable
CPP_FILES=$(wildcard test/src/*.cpp)
TEST_FILES=$(CPP_FILES:test/%=%)

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

debug_test: debug_etl_test
	./debug/bin/etl_test

release_debug_test: release_debug_etl_test
	./release_debug/bin/etl_test

release_test: release_etl_test
	./release/bin/etl_test

test: all
	./debug/bin/etl_test
	./release_debug/bin/etl_test
	./release/bin/etl_test

valgrind_test: debug
	valgrind --leak-check=full ./debug/bin/etl_test

benchmark: release/bin/benchmark
	./release/bin/benchmark --tag=`git rev-list HEAD --count`-`git rev-parse HEAD`

full_bench:
	bash scripts/bench_runner.sh

cppcheck:
	cppcheck -I include/ --platform=unix64 --suppress=missingIncludeSystem --enable=all --std=c++11 workbench/*.cpp include/etl/*.hpp

coverage: debug_test
	lcov -b . --directory debug/test --capture --output-file debug/bin/app.info
	@ mkdir -p reports/coverage
	genhtml --output-directory reports/coverage debug/bin/app.info

coverage_view: coverage
	firefox reports/coverage/index.html

format:
	find include/etl/ test -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i -style=file

modernize:
	clang-modernize -add-override -loop-convert -pass-by-value -use-auto -use-nullptr -p ${PWD} -include *

# clang-tidy with some false positive checks removed
tidy:
	clang-tidy -checks='*,-llvm-include-order,-clang-analyzer-alpha.core.PointerArithm,-clang-analyzer-alpha.deadcode.UnreachableCode,-clang-analyzer-alpha.core.IdenticalExpr' -p ${PWD} test/*.cpp -header-filter='include/etl/*'

# clang-tidy with all the checks
tidy_all:
	clang-tidy -checks='*' -p ${PWD} test/*.cpp -header-filter='include/etl/*'

doc:
	doxygen Doxyfile

clean: base_clean
	rm -rf reports
	rm -rf latex/ html/

version:
	@echo `git rev-parse HEAD`

tag:
	@echo `git rev-list HEAD --count`-`git rev-parse HEAD`

include make-utils/cpp-utils-finalize.mk
