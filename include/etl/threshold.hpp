//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains thresholds to select implementations based
 * on the expression size
 */

#pragma once

namespace etl {

#ifdef ETL_DEBUG_THRESHOLDS

constexpr size_t gemm_std_max    = 75 * 75;   ///< The maximum number of elements to be handled by std algorithm
constexpr size_t gemm_cublas_min = 180 * 180; ///< The minimum number or elements before considering cublas

constexpr size_t gemm_rr_small_threshold    = 1000; ///< The number of elements of B after which we use BLAS-like kernel (for GEMM)
constexpr size_t gemm_rr_medium_threshold   = 2000; ///< The number of elements of B after which we use BLAS-like kernel (for GEMM)
constexpr size_t gemm_nt_rr_small_threshold = 1000; ///< The number of elements of B after which we use BLAS-like kernel (for GEMM)
constexpr size_t gemm_cc_small_threshold    = 1000; ///< The number of elements of B after which we use BLAS-like kernel (for GEMM)

constexpr size_t gevm_rm_small_threshold = 1000; ///< The number of elements of b after which we use BLAS-like kernel
constexpr size_t gevm_cm_small_threshold = 1000; ///< The number of elements of b after which we use BLAS-like kernel

constexpr size_t gemv_rm_small_threshold = 1000; ///< The number of elements of A after which we use BLAS-like kernel
constexpr size_t gemv_cm_small_threshold = 1000; ///< The number of elements of A after which we use BLAS-like kernel

constexpr size_t parallel_threshold = 2 * 1024; ///< The minimum number of elements before considering parallel implementation

constexpr size_t sum_parallel_threshold     = 1024 * 2; ///< The minimum number of elements before considering parallel acc implementation
constexpr size_t vec_sum_parallel_threshold = 1024 * 2; ///< The minimum number of elements before considering parallel acc implementation

constexpr size_t conv1_parallel_threshold_conv   = 100; ///< The mimum output size before considering parallel convolution
constexpr size_t conv1_parallel_threshold_kernel = 16;  ///< The mimum kernel size before considering parallel convolution

constexpr size_t fft1_many_threshold_transforms = 16;  ///< The mimum number of transforms to parallelize them
constexpr size_t fft1_many_threshold_n          = 768; ///< The mimum size of the transforms to parallelize them

constexpr size_t fft2_many_threshold_transforms = 16;   ///< The mimum number of transforms to parallelize them
constexpr size_t fft2_many_threshold_n          = 1024; ///< The mimum size of the transforms to parallelize them

constexpr size_t stream_threshold = 1024; ///< The threshold at which stream is used

#else

constexpr size_t gemm_std_max    = 75 * 75;   ///< The maximum number of elements to be handled by std algorithm
constexpr size_t gemm_cublas_min = 180 * 180; ///< The minimum number or elements before considering cublas

constexpr size_t gemm_rr_small_threshold    = 100 * 100; ///< The number of elements of B after which we use BLAS-like kernel (for GEMM)
constexpr size_t gemm_rr_medium_threshold   = 400 * 400; ///< The number of elements of B after which we use BLAS-like kernel (for GEMM)
constexpr size_t gemm_nt_rr_small_threshold = 500 * 500; ///< The number of elements of B after which we use BLAS-like kernel (for GEMM)
constexpr size_t gemm_cc_small_threshold    = 40000;     ///< The number of elements of B after which we use BLAS-like kernel (for GEMM)

constexpr size_t gevm_rm_small_threshold = 72000;   ///< The number of elements of b after which we use BLAS-like kernel
constexpr size_t gevm_cm_small_threshold = 4000000; ///< The number of elements of b after which we use BLAS-like kernel

constexpr size_t gemv_rm_small_threshold = 4500000; ///< The number of elements of A after which we use BLAS-like kernel
constexpr size_t gemv_cm_small_threshold = 2400000; ///< The number of elements of A after which we use BLAS-like kernel

constexpr size_t parallel_threshold = 128 * 1024; ///< The minimum number of elements before considering parallel implementation

constexpr size_t sum_parallel_threshold     = 1024 * 32;  ///< The minimum number of elements before considering parallel acc implementation
constexpr size_t vec_sum_parallel_threshold = 1024 * 128; ///< The minimum number of elements before considering parallel acc implementation

constexpr size_t conv1_parallel_threshold_conv   = 100; ///< The mimum output size before considering parallel convolution
constexpr size_t conv1_parallel_threshold_kernel = 16;  ///< The mimum kernel size before considering parallel convolution

constexpr size_t fft1_many_threshold_transforms = 16;  ///< The mimum number of transforms to parallelize them
constexpr size_t fft1_many_threshold_n          = 768; ///< The mimum size of the transforms to parallelize them

constexpr size_t fft2_many_threshold_transforms = 16;   ///< The mimum number of transforms to parallelize them
constexpr size_t fft2_many_threshold_n          = 1024; ///< The mimum size of the transforms to parallelize them

constexpr size_t stream_threshold = cache_size; ///< The threshold at which stream is used

#endif

} //end of namespace etl
