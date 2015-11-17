//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <thread>

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
constexpr const bool vectorize_expr   = false;
#endif

//Flag to enable vectorized implementation of algorithms
#ifdef ETL_VECTORIZE_IMPL
constexpr const bool vectorize_impl = true;
#else
constexpr const bool vectorize_impl   = false;
#endif

//Flag to disable the creation of temporary in expressions
#ifdef ETL_NO_TEMPORARY
constexpr const bool create_temporary = false;
#else
constexpr const bool create_temporary = true;
#endif

#ifdef ETL_PARALLEL_THREADS
constexpr const std::size_t threads = ETL_PARALLEL_THREADS;
#else
const std::size_t threads = std::thread::hardware_concurrency();
#endif

#ifdef ETL_PARALLEL
constexpr const bool parallel = true;
#else
constexpr const bool parallel = false;
#endif

#ifdef ETL_MKL_MODE

//MKL mode enables BLAS mode
#ifndef ETL_BLAS_MODE
#define ETL_BLAS_MODE
#endif

struct is_mkl_enabled : std::true_type {};
struct has_fast_fft : std::true_type {};

#else

struct is_mkl_enabled : std::false_type {};
struct has_fast_fft : std::false_type {};

#endif

//Flag to enable the use of CBLAS library
#ifdef ETL_BLAS_MODE
struct is_cblas_enabled : std::true_type {};
#else
struct is_cblas_enabled : std::false_type {};
#endif

#ifdef ETL_CUBLAS_MODE
struct is_cublas_enabled : std::true_type {};
#else
struct is_cublas_enabled : std::false_type {};
#endif

#ifdef ETL_CUFFT_MODE
struct is_cufft_enabled : std::true_type {};
#else
struct is_cufft_enabled : std::false_type {};
#endif

//Flag to perform elementwise multiplication by default (operator*)
//instead of matrix(vector) multiplication
#ifdef ETL_ELEMENT_WISE_MULTIPLICATION
constexpr const bool is_element_wise_mul_default = true;
#else
constexpr const bool is_element_wise_mul_default = false;
#endif

//Flag to prevent division to be done by multiplication
#ifdef ETL_STRICT_DIV
constexpr const bool is_div_strict = true;
#else
constexpr const bool is_div_strict               = false;
#endif

//Flag to disable unrolling of vectorized loops
#ifdef ETL_NO_UNROLL_VECT
constexpr const bool unroll_vectorized_loops = false;
#else
constexpr const bool unroll_vectorized_loops     = true;
#endif

//Flag to disable unrolling of non-vectorized loops
#ifdef ETL_NO_UNROLL_NON_VECT
constexpr const bool unroll_normal_loops = false;
#else
constexpr const bool unroll_normal_loops         = true;
#endif

enum class vector_mode_t {
    NONE,
    SSE3,
    AVX
};

#ifdef __AVX__
constexpr const vector_mode_t vector_mode = vector_mode_t::AVX;
#elif defined(__SSE3__)
constexpr const vector_mode_t vector_mode        = vector_mode_t::SSE3;
#else
constexpr const vector_mode_t vector_mode = vector_mode_t::NONE;
#endif

#ifdef __INTEL_COMPILER
constexpr const bool intel_compiler = true;
#else
constexpr const bool intel_compiler = false;
#endif

} //end of namespace etl
