//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the bce operations.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision BCE loss
 */
#ifdef EGBLAS_HAS_BCE_SLOSS
static constexpr bool has_bce_sloss = true;
#else
static constexpr bool has_bce_sloss = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline float bce_loss(size_t n, float alpha, const float* A, size_t lda, const float* B, size_t ldb) {
#ifdef EGBLAS_HAS_BCE_SLOSS
    inc_counter("egblas");
    return egblas_bce_sloss(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::bce_loss");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision BCE loss
 */
#ifdef EGBLAS_HAS_BCE_DLOSS
static constexpr bool has_bce_dloss = true;
#else
static constexpr bool has_bce_dloss = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline double bce_loss(size_t n, double alpha, const double* A, size_t lda, const double* B, size_t ldb) {
#ifdef EGBLAS_HAS_BCE_DLOSS
    inc_counter("egblas");
    return egblas_bce_dloss(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::bce_loss");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has single-precision BCE error
 */
#ifdef EGBLAS_HAS_BCE_SERROR
static constexpr bool has_bce_serror = true;
#else
static constexpr bool has_bce_serror = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline float bce_error(size_t n, float alpha, const float* A, size_t lda, const float* B, size_t ldb) {
#ifdef EGBLAS_HAS_BCE_SERROR
    inc_counter("egblas");
    return egblas_bce_serror(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::bce_error");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision BCE error
 */
#ifdef EGBLAS_HAS_BCE_DERROR
static constexpr bool has_bce_derror = true;
#else
static constexpr bool has_bce_derror = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline double bce_error(size_t n, double alpha, const double* A, size_t lda, const double* B, size_t ldb) {
#ifdef EGBLAS_HAS_BCE_DERROR
    inc_counter("egblas");
    return egblas_bce_derror(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::bce_error");

    return 0.0;
#endif
}

} //end of namespace etl::impl::egblas
