//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for conv operations.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SCONV1_VALID
static constexpr bool has_sconv1_valid = true;
#else
static constexpr bool has_sconv1_valid = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas conv1_valid operation
 * \param n The size of the input vector
 * \param k The size of the kernel vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the input vector
 * \param lda The leading dimension of the input vectror
 * \param B The memory of the kernel vector
 * \param ldb The leading dimension of the kernel vector
 * \param C The memory of the output vector
 * \param ldc The leading dimension of the output vector
 */
inline void conv1_valid([[maybe_unused]] size_t       n,
                        [[maybe_unused]] size_t       k,
                        [[maybe_unused]] float        alpha,
                        [[maybe_unused]] const float* A,
                        [[maybe_unused]] size_t       lda,
                        [[maybe_unused]] const float* B,
                        [[maybe_unused]] size_t       ldb,
                        [[maybe_unused]] float*       C,
                        [[maybe_unused]] size_t       ldc) {
#ifdef EGBLAS_HAS_SCONV1_VALID
    inc_counter("egblas");
    egblas_sconv1_valid(n, k, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::conv1_valid");
#endif
}

#ifdef EGBLAS_HAS_DCONV1_VALID
static constexpr bool has_dconv1_valid = true;
#else
static constexpr bool has_dconv1_valid = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas conv1_valid operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void conv1_valid([[maybe_unused]] size_t        n,
                        [[maybe_unused]] size_t        k,
                        [[maybe_unused]] double        alpha,
                        [[maybe_unused]] const double* A,
                        [[maybe_unused]] size_t        lda,
                        [[maybe_unused]] const double* B,
                        [[maybe_unused]] size_t        ldb,
                        [[maybe_unused]] double*       C,
                        [[maybe_unused]] size_t        ldc) {
#ifdef EGBLAS_HAS_DCONV1_VALID
    inc_counter("egblas");
    egblas_dconv1_valid(n, k, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::conv1_valid");
#endif
}

#ifdef EGBLAS_HAS_SCONV1_SAME
static constexpr bool has_sconv1_same = true;
#else
static constexpr bool has_sconv1_same = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas conv1_same operation
 * \param n The size of the input vector
 * \param k The size of the kernel vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the input vector
 * \param lda The leading dimension of the input vectror
 * \param B The memory of the kernel vector
 * \param ldb The leading dimension of the kernel vector
 * \param C The memory of the output vector
 * \param ldc The leading dimension of the output vector
 */
inline void conv1_same([[maybe_unused]] size_t       n,
                       [[maybe_unused]] size_t       k,
                       [[maybe_unused]] float        alpha,
                       [[maybe_unused]] const float* A,
                       [[maybe_unused]] size_t       lda,
                       [[maybe_unused]] const float* B,
                       [[maybe_unused]] size_t       ldb,
                       [[maybe_unused]] float*       C,
                       [[maybe_unused]] size_t       ldc) {
#ifdef EGBLAS_HAS_SCONV1_SAME
    inc_counter("egblas");
    egblas_sconv1_same(n, k, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::conv1_same");
#endif
}

#ifdef EGBLAS_HAS_DCONV1_SAME
static constexpr bool has_dconv1_same = true;
#else
static constexpr bool has_dconv1_same = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas conv1_same operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void conv1_same([[maybe_unused]] size_t        n,
                       [[maybe_unused]] size_t        k,
                       [[maybe_unused]] double        alpha,
                       [[maybe_unused]] const double* A,
                       [[maybe_unused]] size_t        lda,
                       [[maybe_unused]] const double* B,
                       [[maybe_unused]] size_t        ldb,
                       [[maybe_unused]] double*       C,
                       [[maybe_unused]] size_t        ldc) {
#ifdef EGBLAS_HAS_DCONV1_SAME
    inc_counter("egblas");
    egblas_dconv1_same(n, k, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::conv1_same");
#endif
}

} //end of namespace etl::impl::egblas
