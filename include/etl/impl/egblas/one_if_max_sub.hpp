//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the one_if_max_sub operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SONE_IF_MAX_SUB
static constexpr bool has_sone_if_max_sub = true;
#else
static constexpr bool has_sone_if_max_sub = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas one_if_max_sub operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void one_if_max_sub([[maybe_unused]] size_t b,
                           [[maybe_unused]] size_t n,
                           [[maybe_unused]] float alpha,
                           [[maybe_unused]] float* A,
                           [[maybe_unused]] size_t lda,
                           [[maybe_unused]] float* B,
                           [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SONE_IF_MAX_SUB
    inc_counter("egblas");
    egblas_sone_if_max_sub(b, n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::one_if_max_sub");
#endif
}

#ifdef EGBLAS_HAS_DONE_IF_MAX_SUB
static constexpr bool has_done_if_max_sub = true;
#else
static constexpr bool has_done_if_max_sub = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas one_if_max_sub operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void one_if_max_sub([[maybe_unused]] size_t  b,
                           [[maybe_unused]] size_t  n,
                           [[maybe_unused]] double  alpha,
                           [[maybe_unused]] double* A,
                           [[maybe_unused]] size_t  lda,
                           [[maybe_unused]] double* B,
                           [[maybe_unused]] size_t  ldb) {
#ifdef EGBLAS_HAS_DONE_IF_MAX_SUB
    inc_counter("egblas");
    egblas_done_if_max_sub(b, n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::one_if_max_sub");
#endif
}

} //end of namespace etl::impl::egblas
