//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the cce operations.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision CCE loss
 */
#ifdef EGBLAS_HAS_CCE_SLOSS
static constexpr bool has_cce_sloss = true;
#else
static constexpr bool has_cce_sloss = false;
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
inline float cce_loss([[maybe_unused]] size_t n,
                      [[maybe_unused]] float alpha,
                      [[maybe_unused]] float* A,
                      [[maybe_unused]] size_t lda,
                      [[maybe_unused]] float* B,
                      [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CCE_SLOSS
    inc_counter("egblas");
    return egblas_cce_sloss(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::cce_loss");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision CCE loss
 */
#ifdef EGBLAS_HAS_CCE_DLOSS
static constexpr bool has_cce_dloss = true;
#else
static constexpr bool has_cce_dloss = false;
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
inline double cce_loss([[maybe_unused]] size_t n,
                       [[maybe_unused]] double alpha,
                       [[maybe_unused]] double* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] double* B,
                       [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CCE_DLOSS
    inc_counter("egblas");
    return egblas_cce_dloss(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::cce_loss");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has single-precision CCE error
 */
#ifdef EGBLAS_HAS_CCE_SERROR
static constexpr bool has_cce_serror = true;
#else
static constexpr bool has_cce_serror = false;
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
inline float cce_error(
    [[maybe_unused]] size_t n, [[maybe_unused]] size_t m, [[maybe_unused]] float alpha, [[maybe_unused]] float* A, [[maybe_unused]] float* B) {
#ifdef EGBLAS_HAS_CCE_SERROR
    inc_counter("egblas");
    return egblas_cce_serror(n, m, alpha, A, B);
#else
    cpp_unreachable("Invalid call to egblas::cce_error");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision CCE error
 */
#ifdef EGBLAS_HAS_CCE_DERROR
static constexpr bool has_cce_derror = true;
#else
static constexpr bool has_cce_derror = false;
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
inline double cce_error(
    [[maybe_unused]] size_t n, [[maybe_unused]] size_t m, [[maybe_unused]] double alpha, [[maybe_unused]] double* A, [[maybe_unused]] double* B) {
#ifdef EGBLAS_HAS_CCE_DERROR
    inc_counter("egblas");
    return egblas_cce_derror(n, m, alpha, A, B);
#else
    cpp_unreachable("Invalid call to egblas::cce_error");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has single-precision CCE loss
 */
#ifdef EGBLAS_HAS_SCCE
static constexpr bool has_scce = true;
#else
static constexpr bool has_scce = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas cce operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline std::pair<float, float> cce([[maybe_unused]] size_t n,
                                   [[maybe_unused]] size_t m,
                                   [[maybe_unused]] float alpha,
                                   [[maybe_unused]] float beta,
                                   [[maybe_unused]] float* A,
                                   [[maybe_unused]] float* B) {
#ifdef EGBLAS_HAS_SCCE
    inc_counter("egblas");
    return egblas_scce(n, m, alpha, beta, A, B);
#else
    cpp_unreachable("Invalid call to egblas::cce");

    return std::make_pair(0.0f, 0.0f);
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision CCE loss
 */
#ifdef EGBLAS_HAS_DCCE
static constexpr bool has_dcce = true;
#else
static constexpr bool has_dcce = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas cce operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline std::pair<double, double> cce([[maybe_unused]] size_t n,
                                     [[maybe_unused]] size_t m,
                                     [[maybe_unused]] double alpha,
                                     [[maybe_unused]] double beta,
                                     [[maybe_unused]] double* A,
                                     [[maybe_unused]] double* B) {
#ifdef EGBLAS_HAS_DCCE
    inc_counter("egblas");
    return egblas_dcce(n, m, alpha, beta, A, B);
#else
    cpp_unreachable("Invalid call to egblas::cce");

    return std::make_pair(0.0, 0.0);
#endif
}

} //end of namespace etl::impl::egblas
