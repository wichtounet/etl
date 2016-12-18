//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

#ifdef ETL_VECTORIZE_FULL

//VECTORIZE_FULL enables VECTORIZE_EXPR
#ifndef ETL_VECTORIZE_EXPR
#define ETL_VECTORIZE_EXPR
#endif

//VECTORIZE_FULL enables VECTORIZE_IMPL
#ifndef ETL_VECTORIZE_IMPL
#define ETL_VECTORIZE_IMPL
#endif

#endif //ETL_VECTORIZE_FULL

//Flag to enable auto-vectorization of expressions
#ifdef ETL_VECTORIZE_EXPR
constexpr bool vectorize_expr = true;
#else
constexpr bool vectorize_expr   = false;                               ///< Boolean flag indicating if the expression are automatically vectorized
#endif

//Flag to enable vectorized implementation of algorithms
#ifdef ETL_VECTORIZE_IMPL
constexpr bool vectorize_impl = true;
#else
constexpr bool vectorize_impl   = false;                               ///< Boolean flag indicating if the implementations are automatically vectorized
#endif

//Flag to allow conv_valid_multi to use FFT
#ifdef ETL_CONV_VALID_FFT
constexpr bool conv_valid_fft = true;
#else
constexpr bool conv_valid_fft   = false;                               ///< Boolean flag indicating if temporaries are created
#endif

//Select the number of threads
#ifdef ETL_PARALLEL_THREADS
constexpr std::size_t threads = ETL_PARALLEL_THREADS;
#else
const std::size_t threads     = std::thread::hardware_concurrency(); ///< Number of threads
#endif

//Indicate that ETL should run in parallel
#ifdef ETL_PARALLEL
constexpr bool is_parallel = true;
#else
constexpr bool is_parallel = false;                               ///< Boolean flag indicating if expressions and implementations are parallelized (alpha)
#endif

#ifdef ETL_MKL_MODE

//MKL mode enables BLAS mode
#ifndef ETL_BLAS_MODE
#define ETL_BLAS_MODE
#endif

constexpr bool mkl_enabled = true;
constexpr bool has_fast_fft   = true;

#else

constexpr bool mkl_enabled              = false; ///< Boolean flag indicating if MKL is enabled
constexpr bool has_fast_fft                = false; ///< Boolean flag indicating if a fast FFT implementation is available

#endif

//Flag to enable the use of CBLAS library
#ifdef ETL_BLAS_MODE
constexpr bool cblas_enabled = true;
#else
constexpr bool cblas_enabled            = false; ///< Boolean flag indicating if CBLAS is available
#endif

//Flag to indicate that blas is multithreaded
#ifdef ETL_BLAS_THREADS
constexpr bool is_blas_parallel = true;
#else
constexpr bool is_blas_parallel            = false; ///< Boolean flag indicating if CBLAS is running parallel
#endif

#ifdef ETL_CUDA
static_assert(false, "ETL_CUDA should never be set directly");
#endif

#ifdef ETL_GPU

#ifndef ETL_CUBLAS_MODE
#define ETL_CUBLAS_MODE
#endif

#ifndef ETL_CUFFT_MODE
#define ETL_CUFFT_MODE
#endif

#ifndef ETL_CUDNN_MODE
#define ETL_CUDNN_MODE
#endif

#endif

#ifdef ETL_CUBLAS_MODE
constexpr bool cublas_enabled = true;
#define ETL_CUDA
#else
constexpr bool cublas_enabled           = false; ///< Boolean flag indicating if CUBLAS is available
#endif

#ifdef ETL_CUFFT_MODE
#define ETL_CUDA
constexpr bool cufft_enabled = true;
#else
constexpr bool cufft_enabled            = false; ///< Boolean flag indicating if CUFFT is available
#endif

#ifdef ETL_CUDNN_MODE
constexpr bool cudnn_enabled = true;
#define ETL_CUDA
#else
constexpr bool cudnn_enabled            = false; ///< Boolean flag indicating if CUDNN is available
#endif

//Flag to perform elementwise multiplication by default (operator*)
//instead of matrix(vector) multiplication
#ifdef ETL_ELEMENT_WISE_MULTIPLICATION
constexpr bool is_element_wise_mul_default = true;
#else
constexpr bool is_element_wise_mul_default = false; ///< Boolean flag indicating if multiplication of two expression means matrix multiplication (false) or element-wise multiplication (true)
#endif

//Flag to prevent division to be done by multiplication
#ifdef ETL_STRICT_DIV
constexpr bool is_div_strict = true;
#else
constexpr bool is_div_strict               = false; ///< Boolean flag indicating if division can be done by multiplication (false) or not (true)
#endif

// Flag to disable streaming operations
#ifdef ETL_NO_STREAMING
constexpr bool streaming = false;
#else
constexpr bool streaming = true; ///< Booling indicating if streaming can be used
#endif

// Flag to disable padding operations
#ifdef ETL_NO_PADDING
constexpr bool padding = false;
#else
constexpr bool padding = true; ///< Booling indicating if padding can be used
#endif

// Flag to enabled padding operations
#ifdef ETL_ADVANCED_PADDING
constexpr bool advanced_padding = true;
#else
constexpr bool advanced_padding = false; ///< Booling indicating if advanced padding can be used
#endif

// Flag to disable padding implementations
#ifdef ETL_NO_PADDING_IMPL
constexpr bool padding_impl = false;
#else
constexpr bool padding_impl = true; ///< Booling indicating if padding implementations can be used
#endif

// Flag to set the cache size
#ifdef ETL_CACHE_SIZE
constexpr size_t cache_size = ETL_CACHE_SIZE;
#else
constexpr size_t cache_size = 3 * 1024 * 1024; ///< Cache size on the machine
#endif

//Flag to disable unrolling of non-vectorized loops
#ifdef ETL_NO_UNROLL_NON_VECT
constexpr bool unroll_normal_loops = false;
#else
constexpr bool unroll_normal_loops         = true;  ///< Boolean flag indicating if normal loops are getting unrolled
#endif

//Flag to configure the maximum workspace size for ETL
#ifdef ETL_MAX_WORKSPACE
constexpr std::size_t max_workspace = ETL_MAX_WORKSPACE;
#else
constexpr std::size_t max_workspace = 2UL * 1024 * 1024 * 1024; ///< The max workspace we allocate for ETL (2GiB by default)
#endif

//Flag to configure the maximum workspace size for CUDA
#ifdef ETL_CUDNN_MAX_WORKSPACE
constexpr std::size_t cudnn_max_workspace = ETL_CUDNN_MAX_WORKSPACE;
#else
constexpr std::size_t cudnn_max_workspace = 2UL * 1024 * 1024 * 1024; ///< The max workspace we allocate for CUDNN (2GiB by default)
#endif

/*!
 * \brief Vectorization mode
 */
enum class vector_mode_t {
    NONE,  ///< No vectorization is available
    SSE3,  ///< SSE3 is the max vectorization available
    AVX,   ///< AVX is the max vectorization available
    AVX512 ///< AVX-512F is the max vectorization available
};

#ifdef __AVX512F__
constexpr vector_mode_t vector_mode = vector_mode_t::AVX512;
#elif defined(__AVX__)
constexpr vector_mode_t vector_mode        = vector_mode_t::AVX;
#elif defined(__SSE3__)
constexpr vector_mode_t vector_mode = vector_mode_t::SSE3;
#else
constexpr vector_mode_t vector_mode = vector_mode_t::NONE; ///< The vector mode in use
#endif

#ifdef __AVX512F__
constexpr bool avx512_enabled = true;
#else
constexpr bool avx512_enabled              = false; ///< Indicates if AVX512F is available
#endif

#ifdef __AVX__
constexpr bool avx_enabled = true;
#else
constexpr bool avx_enabled                 = false; ///< Indicates if AVX is available
#endif

#ifdef __SSE3__
constexpr bool sse3_enabled = true;
#else
constexpr bool sse3_enabled                = false; ///< Indicates if sse3 is available
#endif

constexpr bool vec_enabled = avx512_enabled || avx_enabled || sse3_enabled; ///< Indicates if any vectorization is available

#ifdef __INTEL_COMPILER
constexpr bool intel_compiler = true;
#else
constexpr bool intel_compiler              = false; ///< Indicates if the project is compiled with intel
#endif

//TODO: Once there is a good selection for conv4_valid, this should be removed
#ifdef ETL_CONV4_PREFER_BLAS
constexpr bool conv4_prefer_blas = true;
#else
constexpr bool conv4_prefer_blas           = false;
#endif

} //end of namespace etl
