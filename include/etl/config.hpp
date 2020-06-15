//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/config_impl.hpp"

namespace etl {

/*!
 * \brief Indicates if the selection can be manually set
 */
constexpr bool manual_select = ETL_MANUAL_SELECT_BOOL;

/*!
 * \brief Indicates if the expressions can be automatically
 * vectorized by ETL.
 *
 * Warning: Disabling this flag can incur a very significant
 * slowdown.
 */
constexpr bool vectorize_expr = ETL_VECTORIZE_EXPR_BOOL;

/*!
 * \brief Indicates if the implementations can be automatically
 * vectorized by ETL.
 *
 * Warning: Disabling this flag can incur a very significant
 * slowdown.
 */
constexpr bool vectorize_impl = ETL_VECTORIZE_IMPL_BOOL;

/*!
 * \brief Indicates if conv_valid_multi can use FFT.
 */
constexpr bool conv_valid_fft = ETL_CONV_VALID_FFT_BOOL;

/*!
 * \brief The number of threads ETL can use in parallel mode
 */
const size_t threads = ETL_PARALLEL_THREADS;

/*!
 * \brief Indicates if support for parallelization is integrated
 * into the framework
 */
constexpr bool parallel_support = ETL_PARALLEL_SUPPORT_BOOL;

/*!
 * \brief Indicates if the expressions and implementations can
 * be automaticallly parallelized.
 *
 * Note: It is generally a good idea to enable this flag.
 */
constexpr bool is_parallel = ETL_PARALLEL_BOOL;

/*!
 * \brief Indicates if the MKL library is available for ETL
 */
constexpr bool mkl_enabled = ETL_MKL_MODE_BOOL;

/*!
 * \brief Indicates if there is a fast FFT routine available.
 *
 * This flag is currently only set to true if MKL or CUFFT is available
 */
constexpr bool has_fast_fft = ETL_MKL_MODE_BOOL || ETL_CUFFT_MODE_BOOL;

/*!
 * \brief Indicates if a BLAS library is available for ETL.
 */
constexpr bool cblas_enabled = ETL_BLAS_MODE_BOOL;

/*!
 * \brief Indicates if the BLAS library is parallel.
 */
constexpr bool is_blas_parallel = ETL_BLAS_THREADS_BOOL;

/*!
 * \brief Indicates if the BLAS library is parallel and we are able to disable
 * the parallel.
 *
 * Currently, this only works for MKL
 */
constexpr bool is_blas_parallel_config = is_blas_parallel && mkl_enabled;

/*!
 * \brief Indicates if CUDA is available.
 */
constexpr bool cuda_enabled = ETL_CUDA_BOOL;

/*!
 * \brief Indicates if the NVIDIA CUBLAS library is available for ETL.
 */
constexpr bool cublas_enabled = ETL_CUBLAS_MODE_BOOL;

/*!
 * \brief Indicates if the NVIDIA CUFFT library is available for ETL.
 */
constexpr bool cufft_enabled = ETL_CUFFT_MODE_BOOL;

/*!
 * \brief Indicates if the NVIDIA CURAND library is available for ETL.
 */
constexpr bool curand_enabled = ETL_CURAND_MODE_BOOL;

/*!
 * \brief Indicates if the NVIDIA CUDNN library is available for ETL.
 */
constexpr bool cudnn_enabled = ETL_CUDNN_MODE_BOOL;

/*!
 * \brief Indicates if the EGBLAS library is available for ETL.
 */
constexpr bool egblas_enabled = ETL_EGBLAS_MODE_BOOL;

/*!
 * \brief Boolean flag indicating if division can be done by
 * multiplication (false) or not (true)
 */
constexpr bool is_div_strict = ETL_STRICT_DIV_BOOL;

/*!
 * \brief Indicates if ETL is allowed to perform streaming (non-temporal writes).
 */
constexpr bool streaming = !ETL_NO_STREAMING_BOOL;

/*!
 * \brief Indicates if ETL is allowed to pad matrices and vectors.
 */
constexpr bool padding = !ETL_NO_PADDING_BOOL;

/*!
 * \brief Indicates if ETL is allowed to pad matrices and vectors.
 *
 * Warning: this flag is still highly experimental
 */
constexpr bool advanced_padding = ETL_ADVANCED_PADDING_BOOL;

/*!
 * \brief Indicates if ETL is allowed to pad matrices and vectors.
 */
constexpr bool padding_impl = !ETL_NO_PADDING_IMPL_BOOL;

/*!
 * \brief Indicates if ETL is allowed tor unroll non-vectorized
 * loops.
 */
constexpr bool unroll_normal_loops = ETL_NO_UNROLL_NON_VECT_BOOL;

/*!
 * \brief Indicates if ETL is relaxed.
 */
constexpr bool relaxed = ETL_RELAXED_BOOL;

/*!
 * \brief Cache size of the machine.
 */
constexpr size_t cache_size = ETL_CACHE_SIZE;

/*!
 * \brief Maximum workspace that ETL is allowed to allocate.
 */
constexpr size_t max_workspace = ETL_MAX_WORKSPACE;

/*!
 * \brief Maximum workspace that ETL is allowed to allocate on the
 * GPU.
 */
constexpr size_t cudnn_max_workspace = ETL_CUDNN_MAX_WORKSPACE;

/*!
 * \brief Vectorization mode
 */
enum class vector_mode_t {
    NONE,  ///< No vectorization is available
    SSE3,  ///< SSE3 is the max vectorization available
    AVX,   ///< AVX is the max vectorization available
    AVX512 ///< AVX-512F is the max vectorization available
};

/*!
 * \brief The default vector mode for vectorization by the
 * evaluator engine.
 */
constexpr vector_mode_t vector_mode = ETL_VECTOR_MODE;

/*!
 * \brief Indicates if AVX512 is available
 */
constexpr bool avx512_enabled = ETL_AVX512_BOOL;

/*!
 * \brief Indicates if AVX is available
 */
constexpr bool avx_enabled = ETL_AVX_BOOL;

/*!
 * \brief Indicates if AVX2 is available
 */
constexpr bool avx2_enabled = ETL_AVX2_BOOL;

/*!
 * \brief Indicates if SSE3 is available
 */
constexpr bool sse3_enabled = ETL_SSE3_BOOL;

/*!
 * \brief Indicates if vectorization is available in any format.
 */
constexpr bool vec_enabled = avx512_enabled || avx_enabled || sse3_enabled;

/*!
 * \brief Indicates if the projectis compiled with intel compiler.
 */
constexpr bool intel_compiler = ETL_INTEL_COMPILER_BOOL;

/* Checks for parameters */

static_assert(!is_parallel || parallel_support, "is_parallel can only work with parallel_support");

} //end of namespace etl
