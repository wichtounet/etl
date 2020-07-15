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
inline float bce_loss([[maybe_unused]] size_t n,
                      [[maybe_unused]] float alpha,
                      [[maybe_unused]] const float* A,
                      [[maybe_unused]] size_t lda,
                      [[maybe_unused]] const float* B,
                      [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_BCE_SLOSS
    inc_counter("egblas");
    return egblas_bce_sloss(n, alpha, A, lda, B, ldb);
#else
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
inline double bce_loss([[maybe_unused]] size_t n,
                       [[maybe_unused]] double alpha,
                       [[maybe_unused]] const double* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const double* B,
                       [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_BCE_DLOSS
    inc_counter("egblas");
    return egblas_bce_dloss(n, alpha, A, lda, B, ldb);
#else
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
inline float bce_error([[maybe_unused]] size_t n,
                       [[maybe_unused]] float alpha,
                       [[maybe_unused]] const float* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const float* B,
                       [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_BCE_SERROR
    inc_counter("egblas");
    return egblas_bce_serror(n, alpha, A, lda, B, ldb);
#else
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
inline double bce_error([[maybe_unused]] size_t n,
                        [[maybe_unused]] double alpha,
                        [[maybe_unused]] const double* A,
                        [[maybe_unused]] size_t lda,
                        [[maybe_unused]] const double* B,
                        [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_BCE_DERROR
    inc_counter("egblas");
    return egblas_bce_derror(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::bce_error");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has single-precision BCE
 */
#ifdef EGBLAS_HAS_SBCE
static constexpr bool has_sbce = true;
#else
static constexpr bool has_sbce = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bce operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline std::pair<float, float> bce([[maybe_unused]] size_t n,
                                   [[maybe_unused]] float alpha,
                                   [[maybe_unused]] float beta,
                                   [[maybe_unused]] const float* A,
                                   [[maybe_unused]] size_t lda,
                                   [[maybe_unused]] const float* B,
                                   [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SBCE
    inc_counter("egblas");
    return egblas_sbce(n, alpha, beta, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::bce");

    return std::make_pair(0.0f, 0.0f);
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision BCE
 */
#ifdef EGBLAS_HAS_DBCE
static constexpr bool has_dbce = true;
#else
static constexpr bool has_dbce = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bce operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline std::pair<double, double> bce([[maybe_unused]] size_t n,
                                     [[maybe_unused]] double alpha,
                                     [[maybe_unused]] double beta,
                                     [[maybe_unused]] const double* A,
                                     [[maybe_unused]] size_t lda,
                                     [[maybe_unused]] const double* B,
                                     [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DBCE
    inc_counter("egblas");
    return egblas_dbce(n, alpha, beta, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::bce");

    return std::make_pair(0.0, 0.0);
#endif
}

} //end of namespace etl::impl::egblas
