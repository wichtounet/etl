//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONFIG_HPP
#define ETL_CONFIG_HPP

namespace etl {

#ifdef ETL_VECTORIZE_EXPR

#ifndef ETL_VECTORIZE
#define ETL_VECTORIZE
#endif

struct vectorize_expr : std::false_type {};

#else

struct vectorize_expr : std::true_type {};

#endif

#ifdef ETL_VECTORIZE

struct vectorize : std::false_type {};

#else

struct vectorize : std::true_type {};

#endif

//Flag to disable the creation of temporary in expressions
#ifdef ETL_NO_TEMPORARY
static constexpr const bool create_temporary = false;
#else
static constexpr const bool create_temporary = true;
#endif

#ifdef ETL_MKL_MODE

#ifndef ETL_BLAS_MODE
#define ETL_BLAS_MODE
#endif

struct is_mkl_enabled : std::true_type {};

#else

struct is_mkl_enabled : std::false_type {};

#endif

#ifdef ETL_BLAS_MODE

struct is_cblas_enabled : std::true_type {};

#else

struct is_cblas_enabled : std::false_type {};

#endif

#ifdef ETL_CONV1_MMUL

struct is_conv1_mmul_enabled : std::true_type {};

#else

struct is_conv1_mmul_enabled : std::false_type {};

#endif

#ifdef ETL_CONV2_MMUL

struct is_conv2_mmul_enabled : std::true_type {};

#else

struct is_conv2_mmul_enabled : std::false_type {};

#endif

#ifdef ETL_ELEMENT_WISE_MULTIPLICATION

struct is_element_wise_mul_default : std::true_type {};

#else

struct is_element_wise_mul_default : std::false_type {};

#endif

#ifdef ETL_STRICT_DIV

struct is_div_strict : std::true_type {};

#else

struct is_div_strict : std::false_type {};

#endif

#ifdef ETL_UNROLL_VECT
constexpr const bool unroll_vectorized_loops = true;
#else
constexpr const bool unroll_vectorized_loops = false;
#endif

#ifdef ETL_UNROLL_NON_VECT
constexpr const bool unroll_normal_loops = true;
#else
constexpr const bool unroll_normal_loops = false;
#endif

} //end of namespace etl

#endif
