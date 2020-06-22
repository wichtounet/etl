default: release

.PHONY: default release debug all clean test debug_test release_debug_test release_test
.PHONY: valgrind_test benchmark cppcheck coverage coverage_view format modernize tidy tidy_all doc
.PHONY: full_bench

include make-utils/flags.mk
include make-utils/cpp-utils.mk

# Use C++20
$(eval $(call use_cpp20))

# Configure the BLAS package to use
ifneq (,$(ETL_BLAS_PKG))
BLAS_PKG = $(ETL_BLAS_PKG)
else
BLAS_PKG = mkl
endif

# Try to detect parallel mkl
ifneq (,$(findstring threads,$(BLAS_PKG)))
CXX_FLAGS += -DETL_BLAS_THREADS
endif

# Build with libc++ if configured
ifneq (,$(CLANG_LIBCXX))
$(eval $(call use_libcxx))
endif

# Be stricter
CXX_FLAGS += -pedantic -Werror -Winvalid-pch -Wno-uninitialized

# Add includes
CXX_FLAGS += -Ilib/include -Idoctest -Itest/include

# Support for extra flags
CXX_FLAGS += $(EXTRA_CXX_FLAGS)

CXX_FLAGS += -ftemplate-backtrace-limit=0

# Tune clang warnings
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wpessimizing-move

# False positives for all variable templates with \tparam documentation
CXX_FLAGS += -Wno-documentation
endif

# Tune GCC warnings
ifeq (,$(findstring clang,$(CXX)))
ifneq (,$(findstring g++,$(CXX)))
CXX_FLAGS += -Wno-ignored-attributes -Wno-misleading-indentation
endif
endif

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

ifneq (,$(ETL_PARALLEL))
CXX_FLAGS += -DETL_PARALLEL
endif

ifneq (,$(ETL_EXTENDED))
CXX_FLAGS += -DETL_EXTENDED_BENCH
endif

ifneq (,$(ETL_THESIS))
CXX_FLAGS += -DETL_THESIS_BENCH
endif

# On demand activation of full GPU support
ifneq (,$(ETL_GPU))
CXX_FLAGS += -DETL_GPU

CXX_FLAGS += $(shell pkg-config --cflags cublas)
CXX_FLAGS += $(shell pkg-config --cflags cufft)
CXX_FLAGS += $(shell pkg-config --cflags cudnn)
CXX_FLAGS += $(shell pkg-config --cflags curand)

LD_FLAGS += $(shell pkg-config --libs cublas)
LD_FLAGS += $(shell pkg-config --libs cufft)
LD_FLAGS += $(shell pkg-config --libs cudnn)
LD_FLAGS += $(shell pkg-config --libs curand)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
else

# On demand activation of cublas support
ifneq (,$(ETL_CUBLAS))
CXX_FLAGS += -DETL_CUBLAS_MODE $(shell pkg-config --cflags cublas)
LD_FLAGS += $(shell pkg-config --libs cublas)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

# On demand activation of cufft support
ifneq (,$(ETL_CUFFT))
CXX_FLAGS += -DETL_CUFFT_MODE $(shell pkg-config --cflags cufft)
LD_FLAGS += $(shell pkg-config --libs cufft)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

# On demand activation of curand support
ifneq (,$(ETL_CURAND))
CXX_FLAGS += -DETL_CURAND_MODE $(shell pkg-config --cflags curand)
LD_FLAGS += $(shell pkg-config --libs curand)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

# On demand activation of cudnn support
ifneq (,$(ETL_CUDNN))
CXX_FLAGS += -DETL_CUDNN_MODE $(shell pkg-config --cflags cudnn)
LD_FLAGS += $(shell pkg-config --libs cudnn)

ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-documentation
endif
endif

endif

# On demand activation of egblas support
ifneq (,$(ETL_EGBLAS))
CXX_FLAGS += -DETL_EGBLAS_MODE $(shell pkg-config --cflags egblas)
LD_FLAGS += $(shell pkg-config --libs egblas)
endif

LD_FLAGS += -pthread

# Enable coverage if not disabled by the user
ifneq (,$(ETL_COVERAGE))
$(eval $(call enable_coverage))
endif

# Enable sonar workarounds
ifneq (,$(ETL_SONAR))
CXX_FLAGS += -DSONAR_ANALYSIS
endif

# Enable Clang sanitizers in debug mode
ifneq (,$(ETL_SANITIZE))
ifneq (,$(findstring clang,$(CXX)))
ifeq (,$(ETL_CUBLAS))
DEBUG_FLAGS += -fsanitize=address,undefined
endif
endif
endif

# Enable advanced vectorization for release modes (unset by the benchmark and the test runner)
ifeq (,$(ETL_NO_DEFAULT))
RELEASE_FLAGS 		+= -DETL_VECTORIZE_FULL
RELEASE_DEBUG_FLAGS += -DETL_VECTORIZE_FULL
endif

# Enable configurable default options (set by the benchmark and the test runner)
ifneq (,$(ETL_DEFAULTS))
DEBUG_FLAGS 		+= $(ETL_DEFAULTS)
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
$(eval $(call auto_folder_compile,workbench/src,-Icpm/include))
$(eval $(call auto_folder_compile,benchmark/src,-DETL_MANUAL_SELECT -Ibenchmark/include -Icpm/include))
$(eval $(call auto_folder_compile,test/src,-DETL_MANUAL_SELECT))

# Collect files for the test executable
CPP_FILES=$(wildcard test/src/*.cpp)
TEST_FILES=$(CPP_FILES:test/%=%)

# Create the main test executable
# Should not be build at the same time as the sub executables
#$(eval $(call add_test_executable,etl_test,$(TEST_FILES)))
#$(eval $(call add_executable_set,etl_test,etl_test))

# Create the sub test executable
$(eval $(call add_test_executable,etl_test_alias,src/test.cpp src/alias.cpp))
$(eval $(call add_test_executable,etl_test_alignment,src/test.cpp src/alignment.cpp))
$(eval $(call add_test_executable,etl_test_assert,src/test.cpp src/assert.cpp))
$(eval $(call add_test_executable,etl_test_avg_pool_2d,src/test.cpp src/avg_pool_2d.cpp))
$(eval $(call add_test_executable,etl_test_avg_pool_3d,src/test.cpp src/avg_pool_3d.cpp))
$(eval $(call add_test_executable,etl_test_avg_pool_upsample,src/test.cpp src/avg_pool_upsample.cpp))
$(eval $(call add_test_executable,etl_test_batch_hint_2d,src/test.cpp src/batch_hint_2d.cpp))
$(eval $(call add_test_executable,etl_test_batch_hint_4d,src/test.cpp src/batch_hint_4d.cpp))
$(eval $(call add_test_executable,etl_test_bias_add,src/test.cpp src/bias_add.cpp))
$(eval $(call add_test_executable,etl_test_big,src/test.cpp src/big.cpp))
$(eval $(call add_test_executable,etl_test_binary,src/test.cpp src/binary.cpp))
$(eval $(call add_test_executable,etl_test_column_major,src/test.cpp src/column_major.cpp))
$(eval $(call add_test_executable,etl_test_compare,src/test.cpp src/compare.cpp))
$(eval $(call add_test_executable,etl_test_complex,src/test.cpp src/complex.cpp))
$(eval $(call add_test_executable,etl_test_conv_1d,src/test.cpp src/conv_1d.cpp))
$(eval $(call add_test_executable,etl_test_conv_2d_backward,src/test.cpp src/conv_2d_backward.cpp))
$(eval $(call add_test_executable,etl_test_conv_2d_full,src/test.cpp src/conv_2d_full.cpp))
$(eval $(call add_test_executable,etl_test_conv_2d_same,src/test.cpp src/conv_2d_same.cpp))
$(eval $(call add_test_executable,etl_test_conv_2d_stride,src/test.cpp src/conv_2d_stride.cpp))
$(eval $(call add_test_executable,etl_test_conv_2d_valid,src/test.cpp src/conv_2d_valid.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_backward,src/test.cpp src/conv_4d_backward.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_backward_filter,src/test.cpp src/conv_4d_backward_filter.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_full,src/test.cpp src/conv_4d_full.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_full_mixed,src/test.cpp src/conv_4d_full_mixed.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_stride,src/test.cpp src/conv_4d_stride.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_valid,src/test.cpp src/conv_4d_valid.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_valid_back,src/test.cpp src/conv_4d_valid_back.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_valid_filter,src/test.cpp src/conv_4d_valid_filter.cpp))
$(eval $(call add_test_executable,etl_test_conv_4d_valid_mixed,src/test.cpp src/conv_4d_valid_mixed.cpp))
$(eval $(call add_test_executable,etl_test_conv_deep,src/test.cpp src/conv_deep.cpp))
$(eval $(call add_test_executable,etl_test_conv_multi,src/test.cpp src/conv_multi.cpp))
$(eval $(call add_test_executable,etl_test_conv_multi_multi,src/test.cpp src/conv_multi_multi.cpp))
$(eval $(call add_test_executable,etl_test_convmtx,src/test.cpp src/convmtx.cpp))
$(eval $(call add_test_executable,etl_test_cross,src/test.cpp src/cross.cpp))
$(eval $(call add_test_executable,etl_test_decomposition,src/test.cpp src/decomposition.cpp))
$(eval $(call add_test_executable,etl_test_diagonal,src/test.cpp src/diagonal.cpp))
$(eval $(call add_test_executable,etl_test_direct,src/test.cpp src/direct.cpp))
$(eval $(call add_test_executable,etl_test_dot,src/test.cpp src/dot.cpp))
$(eval $(call add_test_executable,etl_test_dyn_conv_2d_backward,src/test.cpp src/dyn_conv_2d_backward.cpp))
$(eval $(call add_test_executable,etl_test_dyn_conv_4d_backward,src/test.cpp src/dyn_conv_4d_backward.cpp))
$(eval $(call add_test_executable,etl_test_dyn_conv_4d_backward_filter,src/test.cpp src/dyn_conv_4d_backward_filter.cpp))
$(eval $(call add_test_executable,etl_test_dyn_matrix,src/test.cpp src/dyn_matrix.cpp))
$(eval $(call add_test_executable,etl_test_dyn_pooling_derivative,src/test.cpp src/dyn_pooling_derivative.cpp))
$(eval $(call add_test_executable,etl_test_dyn_prob_max_pool,src/test.cpp src/dyn_prob_max_pool.cpp))
$(eval $(call add_test_executable,etl_test_dyn_rep,src/test.cpp src/dyn_rep.cpp))
$(eval $(call add_test_executable,etl_test_dyn_upsample,src/test.cpp src/dyn_upsample.cpp))
$(eval $(call add_test_executable,etl_test_dyn_vector,src/test.cpp src/dyn_vector.cpp))
$(eval $(call add_test_executable,etl_test_elt_compare,src/test.cpp src/elt_compare.cpp))
$(eval $(call add_test_executable,etl_test_elt_logical,src/test.cpp src/elt_logical.cpp))
$(eval $(call add_test_executable,etl_test_embedding_lookup,src/test.cpp src/embedding_lookup.cpp))
$(eval $(call add_test_executable,etl_test_fast_dyn_matrix,src/test.cpp src/fast_dyn_matrix.cpp))
$(eval $(call add_test_executable,etl_test_fast_matrix,src/test.cpp src/fast_matrix.cpp))
$(eval $(call add_test_executable,etl_test_fast_vector,src/test.cpp src/fast_vector.cpp))
$(eval $(call add_test_executable,etl_test_fft,src/test.cpp src/fft.cpp))
$(eval $(call add_test_executable,etl_test_fft2,src/test.cpp src/fft2.cpp))
$(eval $(call add_test_executable,etl_test_flipping,src/test.cpp src/flipping.cpp))
$(eval $(call add_test_executable,etl_test_gemm,src/test.cpp src/gemm.cpp))
$(eval $(call add_test_executable,etl_test_gemm_cm,src/test.cpp src/gemm_cm.cpp))
$(eval $(call add_test_executable,etl_test_gemm_expr,src/test.cpp src/gemm_expr.cpp))
$(eval $(call add_test_executable,etl_test_gemm_mixed,src/test.cpp src/gemm_mixed.cpp))
$(eval $(call add_test_executable,etl_test_gemm_nt,src/test.cpp src/gemm_nt.cpp))
$(eval $(call add_test_executable,etl_test_gemm_nt_cm,src/test.cpp src/gemm_nt_cm.cpp))
$(eval $(call add_test_executable,etl_test_gemm_tn,src/test.cpp src/gemm_tn.cpp))
$(eval $(call add_test_executable,etl_test_gemm_tn_cm,src/test.cpp src/gemm_tn_cm.cpp))
$(eval $(call add_test_executable,etl_test_gemm_tt,src/test.cpp src/gemm_tt.cpp))
$(eval $(call add_test_executable,etl_test_gemm_tt_cm,src/test.cpp src/gemm_tt_cm.cpp))
$(eval $(call add_test_executable,etl_test_gemm_types,src/test.cpp src/gemm_types.cpp))
$(eval $(call add_test_executable,etl_test_gemv,src/test.cpp src/gemv.cpp))
$(eval $(call add_test_executable,etl_test_gemv_cm,src/test.cpp src/gemv_cm.cpp))
$(eval $(call add_test_executable,etl_test_gemv_mixed,src/test.cpp src/gemv_mixed.cpp))
$(eval $(call add_test_executable,etl_test_gemv_types,src/test.cpp src/gemv_types.cpp))
$(eval $(call add_test_executable,etl_test_generators,src/test.cpp src/generators.cpp))
$(eval $(call add_test_executable,etl_test_gevm,src/test.cpp src/gevm.cpp))
$(eval $(call add_test_executable,etl_test_gevm_cm,src/test.cpp src/gevm_cm.cpp))
$(eval $(call add_test_executable,etl_test_gevm_mixed,src/test.cpp src/gevm_mixed.cpp))
$(eval $(call add_test_executable,etl_test_gevm_types,src/test.cpp src/gevm_types.cpp))
$(eval $(call add_test_executable,etl_test_globals,src/test.cpp src/globals.cpp))
$(eval $(call add_test_executable,etl_test_gpu,src/test.cpp src/gpu.cpp))
$(eval $(call add_test_executable,etl_test_hermitian,src/test.cpp src/hermitian.cpp))
$(eval $(call add_test_executable,etl_test_ifft,src/test.cpp src/ifft.cpp))
$(eval $(call add_test_executable,etl_test_ifft2,src/test.cpp src/ifft2.cpp))
$(eval $(call add_test_executable,etl_test_im2col,src/test.cpp src/im2col.cpp))
$(eval $(call add_test_executable,etl_test_integers,src/test.cpp src/integers.cpp))
$(eval $(call add_test_executable,etl_test_inv,src/test.cpp src/inv.cpp))
$(eval $(call add_test_executable,etl_test_iterations,src/test.cpp src/iterations.cpp))
$(eval $(call add_test_executable,etl_test_iterators,src/test.cpp src/iterators.cpp))
$(eval $(call add_test_executable,etl_test_lower,src/test.cpp src/lower.cpp))
$(eval $(call add_test_executable,etl_test_max_pool_2d,src/test.cpp src/max_pool_2d.cpp))
$(eval $(call add_test_executable,etl_test_max_pool_3d,src/test.cpp src/max_pool_3d.cpp))
$(eval $(call add_test_executable,etl_test_max_pool_upsample,src/test.cpp src/max_pool_upsample.cpp))
$(eval $(call add_test_executable,etl_test_memory_slice,src/test.cpp src/memory_slice.cpp))
$(eval $(call add_test_executable,etl_test_merge,src/test.cpp src/merge.cpp))
$(eval $(call add_test_executable,etl_test_ml,src/test.cpp src/ml.cpp))
$(eval $(call add_test_executable,etl_test_noise,src/test.cpp src/noise.cpp))
$(eval $(call add_test_executable,etl_test_optimize_1,src/test.cpp src/optimize_1.cpp))
$(eval $(call add_test_executable,etl_test_optimize_2,src/test.cpp src/optimize_2.cpp))
$(eval $(call add_test_executable,etl_test_outer,src/test.cpp src/outer.cpp))
$(eval $(call add_test_executable,etl_test_parallel,src/test.cpp src/parallel.cpp))
$(eval $(call add_test_executable,etl_test_pooling_derivative,src/test.cpp src/pooling_derivative.cpp))
$(eval $(call add_test_executable,etl_test_print,src/test.cpp src/print.cpp))
$(eval $(call add_test_executable,etl_test_prob_max_pool,src/test.cpp src/prob_max_pool.cpp))
$(eval $(call add_test_executable,etl_test_reduc,src/test.cpp src/reduc.cpp))
$(eval $(call add_test_executable,etl_test_rep,src/test.cpp src/rep.cpp))
$(eval $(call add_test_executable,etl_test_scalar_op,src/test.cpp src/scalar_op.cpp))
$(eval $(call add_test_executable,etl_test_selected,src/test.cpp src/selected.cpp))
$(eval $(call add_test_executable,etl_test_serial,src/test.cpp src/serial.cpp))
$(eval $(call add_test_executable,etl_test_serializer,src/test.cpp src/serializer.cpp))
$(eval $(call add_test_executable,etl_test_slice,src/test.cpp src/slice.cpp))
$(eval $(call add_test_executable,etl_test_softmax,src/test.cpp src/softmax.cpp))
$(eval $(call add_test_executable,etl_test_sparse_complex,src/test.cpp src/sparse_complex.cpp))
$(eval $(call add_test_executable,etl_test_sparse_matrix,src/test.cpp src/sparse_matrix.cpp))
$(eval $(call add_test_executable,etl_test_special_cases,src/test.cpp src/special_cases.cpp))
$(eval $(call add_test_executable,etl_test_stop,src/test.cpp src/stop.cpp))
$(eval $(call add_test_executable,etl_test_strictly_lower,src/test.cpp src/strictly_lower.cpp))
$(eval $(call add_test_executable,etl_test_strictly_upper,src/test.cpp src/strictly_upper.cpp))
$(eval $(call add_test_executable,etl_test_sub_matrix_2d,src/test.cpp src/sub_matrix_2d.cpp))
$(eval $(call add_test_executable,etl_test_sub_matrix_3d,src/test.cpp src/sub_matrix_3d.cpp))
$(eval $(call add_test_executable,etl_test_sub_matrix_4d,src/test.cpp src/sub_matrix_4d.cpp))
$(eval $(call add_test_executable,etl_test_symmetric,src/test.cpp src/symmetric.cpp))
$(eval $(call add_test_executable,etl_test_timed,src/test.cpp src/timed.cpp))
$(eval $(call add_test_executable,etl_test_tmp,src/test.cpp src/tmp.cpp))
$(eval $(call add_test_executable,etl_test_traits,src/test.cpp src/traits.cpp))
$(eval $(call add_test_executable,etl_test_transpose,src/test.cpp src/transpose.cpp))
$(eval $(call add_test_executable,etl_test_trigo,src/test.cpp src/trigo.cpp))
$(eval $(call add_test_executable,etl_test_unaligned,src/test.cpp src/unaligned.cpp))
$(eval $(call add_test_executable,etl_test_unary,src/test.cpp src/unary.cpp))
$(eval $(call add_test_executable,etl_test_uni_lower,src/test.cpp src/uni_lower.cpp))
$(eval $(call add_test_executable,etl_test_uni_upper,src/test.cpp src/uni_upper.cpp))
$(eval $(call add_test_executable,etl_test_unmanaged,src/test.cpp src/unmanaged.cpp))
$(eval $(call add_test_executable,etl_test_upper,src/test.cpp src/upper.cpp))
$(eval $(call add_test_executable,etl_test_upsample,src/test.cpp src/upsample.cpp))
$(eval $(call add_test_executable,etl_test_views,src/test.cpp src/views.cpp))
$(eval $(call add_test_executable,etl_test_virtual_views,src/test.cpp src/virtual_views.cpp))

debug_test_all: $(DEBUG_TEST_EXECUTABLES)
release_debug_test_all: $(RELEASE_DEBUG_TEST_EXECUTABLES)
release_test_all: $(RELEASE_TEST_EXECUTABLES)

run_debug_test_all: $(DEBUG_TEST_TARGETS)
run_release_debug_test_all: $(RELEASE_DEBUG_TEST_TARGETS)
run_release_test_all: $(RELEASE_TEST_TARGETS)

# Create the benchmark executables
BENCH_FILES=$(wildcard benchmark/src/benchmark*cpp)

$(eval $(call add_executable,benchmark,$(BENCH_FILES)))
$(eval $(call add_executable,benchmark_benchmark,benchmark/src/benchmark.cpp))
$(eval $(call add_executable,benchmark_cdbn,benchmark/src/benchmark.cpp benchmark/src/benchmark_cdbn.cpp))
$(eval $(call add_executable,benchmark_conv,benchmark/src/benchmark.cpp benchmark/src/benchmark_conv.cpp))
$(eval $(call add_executable,benchmark_conv_extended,benchmark/src/benchmark.cpp benchmark/src/benchmark_conv_extended.cpp))
$(eval $(call add_executable,benchmark_fft,benchmark/src/benchmark.cpp benchmark/src/benchmark_fft.cpp))
$(eval $(call add_executable,benchmark_gemm,benchmark/src/benchmark.cpp benchmark/src/benchmark_gemm.cpp))
$(eval $(call add_executable,benchmark_pool,benchmark/src/benchmark.cpp benchmark/src/benchmark_pool.cpp))
$(eval $(call add_executable,benchmark_thesis,benchmark/src/benchmark.cpp benchmark/src/benchmark_thesis.cpp))
$(eval $(call add_executable,benchmark_trigo,benchmark/src/benchmark.cpp benchmark/src/benchmark_trigo.cpp))

# Create various executables
$(eval $(call add_executable,test_asm_1,workbench/src/test.cpp))
$(eval $(call add_executable,test_asm_2,workbench/src/test_dim.cpp))
$(eval $(call add_executable,mmul,workbench/src/mmul.cpp))
$(eval $(call add_executable,parallel,workbench/src/parallel.cpp))
$(eval $(call add_executable,multi,workbench/src/multi.cpp))
$(eval $(call add_executable,locality,workbench/src/locality.cpp))
$(eval $(call add_executable,counters,workbench/src/counters.cpp))
$(eval $(call add_executable,verify_cpm,workbench/src/verify_cpm.cpp))
$(eval $(call add_executable,benchmark_paper,workbench/src/benchmark_paper.cpp))

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
	cppcheck -I include/ --platform=unix64 --suppress=missingIncludeSystem --enable=all --std=c++11 benchmark/*.cpp workbench/*.cpp include/etl/*.hpp

coverage: debug_test
	lcov -b . --directory debug/test --capture --output-file debug/bin/app.info
	@ mkdir -p reports/coverage
	genhtml --output-directory reports/coverage debug/bin/app.info

coverage_view: coverage
	firefox reports/coverage/index.html

# Note: workbench / benchmark is no included on purpose because of poor macro alignment
format:
	find include test -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i -style=file

# Note: test are not included on purpose (we want to force to test some operators on matrix/vector)
modernize:
	find include benchmark workbench -name "*.hpp" -o -name "*.cpp" > etl_file_list
	clang-modernize -add-override -loop-convert -pass-by-value -use-auto -use-nullptr -p ${PWD} -include-from=etl_file_list
	rm etl_file_list

clang-tidy:
	@ /usr/share/clang/run-clang-tidy.py -p . -header-filter '^include/etl' -checks='cert-*,cppcoreguidelines-*,google-*,llvm-*,misc-*,modernize-*,performance-*,readility-*,-cppcoreguidelines-pro-type-reinterpret-cast,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-google-readability-namespace-comments,-llvm-namespace-comment,-llvm-include-order,-google-runtime-references,-misc-unconventional-assign-operator,-cppcoreguidelines-c-copy-assignment-signature,-google-readability-todo,-cppcoreguidelines-pro-type-vararg,-cppcoreguidelines-pro-type-const-cast,-cppcoreguidelines-pro-bounds-array-to-pointer-decay,-cppcoreguidelines-pro-bounds-constant-array-index,-cert-err58-cpp' -j9 2>/dev/null  | /usr/bin/zgrep -v "^clang-tidy"

clang-tidy-all:
	@ /usr/share/clang/run-clang-tidy.py -header-filter '^include/etl' -checks='*' -j9

clang-tidy-mono:
	clang-tidy -p . -header-filter '^include/etl' -checks='cert-*,cppcoreguidelines-*,google-*,llvm-*,misc-*,modernize-*,performance-*,readility-*,-cppcoreguidelines-pro-type-reinterpret-cast,-cppcoreguidelines-pro-bounds-pointer-arithmetic,-google-readability-namespace-comments,-llvm-namespace-comment,-llvm-include-order,-google-runtime-references,-misc-unconventional-assign-operator,-cppcoreguidelines-c-copy-assignment-signature,-google-readability-todo,-cppcoreguidelines-pro-type-vararg,-cppcoreguidelines-pro-type-const-cast,-cppcoreguidelines-pro-bounds-array-to-pointer-decay,-cppcoreguidelines-pro-bounds-constant-array-index,-cert-err58-cpp' test/*.cpp

clang-tidy-mono-all:
	clang-tidy -p . -header-filter '^include/etl' -checks='*' test/*.cpp

doc:
	doxygen Doxyfile

clean: base_clean
	rm -rf reports
	rm -rf latex/ html/

# Fight against my fingers
clena: clean
claen: clean

version:
	@echo `git rev-parse HEAD`

tag:
	@echo `git rev-list HEAD --count`-`git rev-parse HEAD`

include make-utils/cpp-utils-finalize.mk
