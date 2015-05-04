//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONFIG_HPP
#define ETL_CONFIG_HPP

namespace etl {

#ifdef ETL_VECTORIZE_FULL

#ifndef ETL_VECTORIZE_EXPR
#define ETL_VECTORIZE_EXPR
#endif

#ifndef ETL_VECTORIZE_IMPL
#define ETL_VECTORIZE_IMPL
#endif

#endif //ETL_VECTORIZE_FULL

#ifdef ETL_VECTORIZE_EXPR
constexpr const bool vectorize_expr = true;
#else
constexpr const bool vectorize_expr = false;
#endif

#ifdef ETL_VECTORIZE_IMPL
struct vectorize : std::false_type {};
#else
struct vectorize : std::true_type {};
#endif

//Flag to disable the creation of temporary in expressions
#ifdef ETL_NO_TEMPORARY
constexpr const bool create_temporary = false;
#else
constexpr const bool create_temporary = true;
#endif

#ifdef ETL_MKL_MODE

//MKL mode implies BLAS mode
#ifndef ETL_BLAS_MODE
#define ETL_BLAS_MODE
#endif

struct is_mkl_enabled : std::true_type {};

#else

struct is_mkl_enabled : std::false_type {};

#endif

//Flag to enable the use of CBLAS library
#ifdef ETL_BLAS_MODE
struct is_cblas_enabled : std::true_type {};
#else
struct is_cblas_enabled : std::false_type {};
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
constexpr const bool is_div_strict = false;
#endif

//Flag to enable unrolling of vectorized loops
#ifdef ETL_UNROLL_VECT
constexpr const bool unroll_vectorized_loops = true;
#else
constexpr const bool unroll_vectorized_loops = false;
#endif

//Flag to enable unrolling of non-vectorized loops
#ifdef ETL_UNROLL_NON_VECT
constexpr const bool unroll_normal_loops = true;
#else
constexpr const bool unroll_normal_loops = false;
#endif

} //end of namespace etl

#endif
