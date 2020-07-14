//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the mse operations.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision MSE loss
 */
#ifdef EGBLAS_HAS_MSE_SLOSS
static constexpr bool has_mse_sloss = true;
#else
static constexpr bool has_mse_sloss = false;
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
inline float mse_loss([[maybe_unused]] size_t n,
                      [[maybe_unused]] float alpha,
                      [[maybe_unused]] const float* A,
                      [[maybe_unused]] size_t lda,
                      [[maybe_unused]] const float* B,
                      [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_MSE_SLOSS
    inc_counter("egblas");
    return egblas_mse_sloss(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::mse_loss");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision MSE loss
 */
#ifdef EGBLAS_HAS_MSE_DLOSS
static constexpr bool has_mse_dloss = true;
#else
static constexpr bool has_mse_dloss = false;
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
inline double mse_loss([[maybe_unused]] size_t n,
                       [[maybe_unused]] double alpha,
                       [[maybe_unused]] const double* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const double* B,
                       [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_MSE_DLOSS
    inc_counter("egblas");
    return egblas_mse_dloss(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::mse_loss");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has single-precision MSE error
 */
#ifdef EGBLAS_HAS_MSE_SERROR
static constexpr bool has_mse_serror = true;
#else
static constexpr bool has_mse_serror = false;
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
inline float mse_error([[maybe_unused]] size_t n,
                       [[maybe_unused]] float alpha,
                       [[maybe_unused]] const float* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const float* B,
                       [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_MSE_SERROR
    inc_counter("egblas");
    return egblas_mse_serror(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::mse_error");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision MSE error
 */
#ifdef EGBLAS_HAS_MSE_DERROR
static constexpr bool has_mse_derror = true;
#else
static constexpr bool has_mse_derror = false;
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
inline double mse_error([[maybe_unused]] size_t n,
                        [[maybe_unused]] double alpha,
                        [[maybe_unused]] const double* A,
                        [[maybe_unused]] size_t lda,
                        [[maybe_unused]] const double* B,
                        [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_MSE_DERROR
    inc_counter("egblas");
    return egblas_mse_derror(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::mse_error");

    return 0.0;
#endif
}

/*!
 * \brief Indicates if EGBLAS has single-precision MSE
 */
#ifdef EGBLAS_HAS_MSE_SLOSS
static constexpr bool has_smse = true;
#else
static constexpr bool has_smse = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas mse operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline std::pair<float, float> mse([[maybe_unused]] size_t n,
                                   [[maybe_unused]] float alpha,
                                   [[maybe_unused]] float beta,
                                   [[maybe_unused]] const float* A,
                                   [[maybe_unused]] size_t lda,
                                   [[maybe_unused]] const float* B,
                                   [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SMSE
    inc_counter("egblas");
    return egblas_smse(n, alpha, beta, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::mse");

    return std::make_pair(0.0f, 0.0f);
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision MSE
 */
#ifdef EGBLAS_HAS_MSE_DLOSS
static constexpr bool has_dmse = true;
#else
static constexpr bool has_dmse = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas mse operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline std::pair<double, double> mse([[maybe_unused]] size_t n,
                                     [[maybe_unused]] double alpha,
                                     [[maybe_unused]] double beta,
                                     [[maybe_unused]] const double* A,
                                     [[maybe_unused]] size_t lda,
                                     [[maybe_unused]] const double* B,
                                     [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SMSE
    inc_counter("egblas");
    return egblas_dmse(n, alpha, beta, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::mse");

    return std::make_pair(0.0, 0.0);
#endif
}

} //end of namespace etl::impl::egblas
