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
constexpr const bool vectorize_expr = true;
#else
constexpr const bool vectorize_expr   = false;                               ///< Boolean flag indicating if the expression are automatically vectorized
#endif

//Flag to enable vectorized implementation of algorithms
#ifdef ETL_VECTORIZE_IMPL
constexpr const bool vectorize_impl = true;
#else
constexpr const bool vectorize_impl   = false;                               ///< Boolean flag indicating if the implementations are automatically vectorized
#endif

//Flag to disable the creation of temporary in expressions
#ifdef ETL_NO_TEMPORARY
constexpr const bool create_temporary = false;
#else
constexpr const bool create_temporary = true;                                ///< Boolean flag indicating if temporaries are created
#endif

//Flag to allow conv_valid_multi to use FFT
#ifdef ETL_CONV_VALID_FFT
constexpr const bool conv_valid_fft = true;
#else
constexpr const bool conv_valid_fft   = false;                               ///< Boolean flag indicating if temporaries are created
#endif

//Select the number of threads
#ifdef ETL_PARALLEL_THREADS
constexpr const std::size_t threads = ETL_PARALLEL_THREADS;
#else
const std::size_t threads             = std::thread::hardware_concurrency(); ///< Number of threads
#endif

//Indicate that ETL should run in parallel
#ifdef ETL_PARALLEL
constexpr const bool parallel = true;
#else
constexpr const bool parallel         = false;                               ///< Boolean flag indicating if expressions and implementations are parallelized (alpha)
#endif

#ifdef ETL_MKL_MODE

//MKL mode enables BLAS mode
#ifndef ETL_BLAS_MODE
#define ETL_BLAS_MODE
#endif

constexpr const bool is_mkl_enabled = true;
constexpr const bool has_fast_fft   = true;

#else

constexpr const bool is_mkl_enabled              = false;
constexpr const bool has_fast_fft                = false;

#endif

//Flag to enable the use of CBLAS library
#ifdef ETL_BLAS_MODE
constexpr const bool is_cblas_enabled = true;
#else
constexpr const bool is_cblas_enabled            = false;
#endif

//Flag to indicate that blas is multithreaded
#ifdef ETL_BLAS_THREADS
constexpr const bool is_blas_parallel = true;
#else
constexpr const bool is_blas_parallel            = false;
#endif

#ifdef ETL_CUDA
static_assert(false, "ETL_CUDA should never be set directly");
#endif

#ifdef ETL_CUBLAS_MODE
constexpr const bool is_cublas_enabled = true;
#define ETL_CUDA
#else
constexpr const bool is_cublas_enabled           = false;
#endif

#ifdef ETL_CUFFT_MODE
#define ETL_CUDA
constexpr const bool is_cufft_enabled = true;
#else
constexpr const bool is_cufft_enabled            = false;
#endif

//Flag to perform elementwise multiplication by default (operator*)
//instead of matrix(vector) multiplication
#ifdef ETL_ELEMENT_WISE_MULTIPLICATION
constexpr const bool is_element_wise_mul_default = true;
#else
constexpr const bool is_element_wise_mul_default = false; ///< Boolean flag indicating if multiplication of two expression means matrix multiplication (false) or element-wise multiplication (true)
#endif

//Flag to prevent division to be done by multiplication
#ifdef ETL_STRICT_DIV
constexpr const bool is_div_strict = true;
#else
constexpr const bool is_div_strict               = false; ///< Boolean flag indicating if division can be done by multiplication (false) or not (true)
#endif

//Flag to disable unrolling of vectorized loops
#ifdef ETL_NO_UNROLL_VECT
constexpr const bool unroll_vectorized_loops = false;
#else
constexpr const bool unroll_vectorized_loops     = true;  ///< Boolean flag indicating if vectorized loops are getting unrolled
#endif

//Flag to disable unrolling of non-vectorized loops
#ifdef ETL_NO_UNROLL_NON_VECT
constexpr const bool unroll_normal_loops = false;
#else
constexpr const bool unroll_normal_loops         = true;  ///< Boolean flag indicating if normal loops are getting unrolled
#endif

/*!
 * \brief Vectorization mode
 */
enum class vector_mode_t {
    NONE,
    SSE3,
    AVX,
    AVX512
};

#ifdef __AVX512F__
constexpr const vector_mode_t vector_mode = vector_mode_t::AVX512;
#elif defined(__AVX__)
constexpr const vector_mode_t vector_mode        = vector_mode_t::AVX;
#elif defined(__SSE3__)
constexpr const vector_mode_t vector_mode = vector_mode_t::SSE3;
#else
constexpr const vector_mode_t vector_mode = vector_mode_t::NONE; ///< The vector mode in use
#endif

#ifdef __AVX512F__
constexpr const bool avx512_enabled = true;
#else
constexpr const bool avx512_enabled              = false;
#endif

#ifdef __AVX__
constexpr const bool avx_enabled = true;
#else
constexpr const bool avx_enabled                 = false;
#endif

#ifdef __SSE3__
constexpr const bool sse3_enabled = true;
#else
constexpr const bool sse3_enabled                = false;
#endif

#ifdef __INTEL_COMPILER
constexpr const bool intel_compiler = true;
#else
constexpr const bool intel_compiler              = false; ///< Indicates if the project is compiled with intel
#endif

} //end of namespace etl
