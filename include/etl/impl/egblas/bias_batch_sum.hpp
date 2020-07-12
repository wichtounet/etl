//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the bias_batch_sum operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SBIAS_BATCH_SUM
static constexpr bool has_sbias_batch_sum = true;
#else
static constexpr bool has_sbias_batch_sum = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bias_batch_sum operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void bias_batch_sum([[maybe_unused]] size_t b,
                           [[maybe_unused]] size_t n,
                           [[maybe_unused]] float* A,
                           [[maybe_unused]] size_t lda,
                           [[maybe_unused]] float* B,
                           [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SBIAS_BATCH_SUM
    inc_counter("egblas");
    egblas_sbias_batch_sum(b, n, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_sum");
#endif
}

#ifdef EGBLAS_HAS_DBIAS_BATCH_SUM
static constexpr bool has_dbias_batch_sum = true;
#else
static constexpr bool has_dbias_batch_sum = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas bias_batch_sum operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void bias_batch_sum([[maybe_unused]] size_t  b,
                           [[maybe_unused]] size_t  n,
                           [[maybe_unused]] double* A,
                           [[maybe_unused]] size_t  lda,
                           [[maybe_unused]] double* B,
                           [[maybe_unused]] size_t  ldb) {
#ifdef EGBLAS_HAS_DBIAS_BATCH_SUM
    inc_counter("egblas");
    egblas_dbias_batch_sum(b, n, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_sum");
#endif
}

#ifdef EGBLAS_HAS_SBIAS_BATCH_MEAN
static constexpr bool has_sbias_batch_mean = true;
#else
static constexpr bool has_sbias_batch_mean = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bias_batch_mean operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void bias_batch_mean([[maybe_unused]] size_t b,
                            [[maybe_unused]] size_t n,
                            [[maybe_unused]] float* A,
                            [[maybe_unused]] size_t lda,
                            [[maybe_unused]] float* B,
                            [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SBIAS_BATCH_MEAN
    inc_counter("egblas");
    egblas_sbias_batch_mean(b, n, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_mean");
#endif
}

#ifdef EGBLAS_HAS_DBIAS_BATCH_MEAN
static constexpr bool has_dbias_batch_mean = true;
#else
static constexpr bool has_dbias_batch_mean = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas bias_batch_mean operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void bias_batch_mean([[maybe_unused]] size_t  b,
                            [[maybe_unused]] size_t  n,
                            [[maybe_unused]] double* A,
                            [[maybe_unused]] size_t  lda,
                            [[maybe_unused]] double* B,
                            [[maybe_unused]] size_t  ldb) {
#ifdef EGBLAS_HAS_DBIAS_BATCH_MEAN
    inc_counter("egblas");
    egblas_dbias_batch_mean(b, n, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_mean");
#endif
}

#ifdef EGBLAS_HAS_SBIAS_BATCH_VAR
static constexpr bool has_sbias_batch_var = true;
#else
static constexpr bool has_sbias_batch_var = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bias_batch_var operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void bias_batch_var([[maybe_unused]] size_t b,
                           [[maybe_unused]] size_t n,
                           [[maybe_unused]] float* A,
                           [[maybe_unused]] size_t lda,
                           [[maybe_unused]] float* B,
                           [[maybe_unused]] size_t ldb,
                           [[maybe_unused]] float* Y,
                           [[maybe_unused]] size_t ldy) {
#ifdef EGBLAS_HAS_SBIAS_BATCH_VAR
    inc_counter("egblas");
    egblas_sbias_batch_var(b, n, A, lda, B, ldb, Y, ldy);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_var");
#endif
}

#ifdef EGBLAS_HAS_DBIAS_BATCH_VAR
static constexpr bool has_dbias_batch_var = true;
#else
static constexpr bool has_dbias_batch_var = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas bias_batch_var operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void bias_batch_var([[maybe_unused]] size_t  b,
                           [[maybe_unused]] size_t  n,
                           [[maybe_unused]] double* A,
                           [[maybe_unused]] size_t  lda,
                           [[maybe_unused]] double* B,
                           [[maybe_unused]] size_t  ldb,
                           [[maybe_unused]] double* Y,
                           [[maybe_unused]] size_t  ldy) {
#ifdef EGBLAS_HAS_DBIAS_BATCH_VAR
    inc_counter("egblas");
    egblas_dbias_batch_var(b, n, A, lda, B, ldb, Y, ldy);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_var");
#endif
}

#ifdef EGBLAS_HAS_SBIAS_BATCH_SUM4
static constexpr bool has_sbias_batch_sum4 = true;
#else
static constexpr bool has_sbias_batch_sum4 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bias_batch_sum operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param w The inner first dimension of the input matrix
 * \param h The inner first dimension of the input matrix
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void bias_batch_sum4([[maybe_unused]] size_t b,
                           [[maybe_unused]] size_t n,
                           [[maybe_unused]] size_t w,
                           [[maybe_unused]] size_t h,
                           [[maybe_unused]] float* A,
                           [[maybe_unused]] float* B) {
#ifdef EGBLAS_HAS_SBIAS_BATCH_SUM4
    inc_counter("egblas");
    egblas_sbias_batch_sum4(b, n, w, h, A, B);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_sum");
#endif
}

#ifdef EGBLAS_HAS_SBIAS_BATCH_MEAN4
static constexpr bool has_sbias_batch_mean4 = true;
#else
static constexpr bool has_sbias_batch_mean4 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bias_batch_mean operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param w The inner first dimension of the input matrix
 * \param h The inner first dimension of the input matrix
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void bias_batch_mean4([[maybe_unused]] size_t b,
                           [[maybe_unused]] size_t n,
                           [[maybe_unused]] size_t w,
                           [[maybe_unused]] size_t h,
                           [[maybe_unused]] float* A,
                           [[maybe_unused]] float* B) {
#ifdef EGBLAS_HAS_SBIAS_BATCH_MEAN4
    inc_counter("egblas");
    egblas_sbias_batch_mean4(b, n, w, h, A, B);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_mean");
#endif
}

#ifdef EGBLAS_HAS_DBIAS_BATCH_SUM4
static constexpr bool has_dbias_batch_sum4 = true;
#else
static constexpr bool has_dbias_batch_sum4 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bias_batch_sum operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param w The inner first dimension of the input matrix
 * \param h The inner first dimension of the input matrix
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void bias_batch_sum4([[maybe_unused]] size_t b,
                           [[maybe_unused]] size_t n,
                           [[maybe_unused]] size_t w,
                           [[maybe_unused]] size_t h,
                           [[maybe_unused]] double* A,
                           [[maybe_unused]] double* B) {
#ifdef EGBLAS_HAS_DBIAS_BATCH_SUM4
    inc_counter("egblas");
    egblas_dbias_batch_sum4(b, n, w, h, A, B);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_sum");
#endif
}

#ifdef EGBLAS_HAS_DBIAS_BATCH_MEAN4
static constexpr bool has_dbias_batch_mean4 = true;
#else
static constexpr bool has_dbias_batch_mean4 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bias_batch_mean operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param w The inner first dimension of the input matrix
 * \param h The inner first dimension of the input matrix
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void bias_batch_mean4([[maybe_unused]] size_t b,
                           [[maybe_unused]] size_t n,
                           [[maybe_unused]] size_t w,
                           [[maybe_unused]] size_t h,
                           [[maybe_unused]] double* A,
                           [[maybe_unused]] double* B) {
#ifdef EGBLAS_HAS_DBIAS_BATCH_MEAN4
    inc_counter("egblas");
    egblas_dbias_batch_mean4(b, n, w, h, A, B);
#else
    cpp_unreachable("Invalid call to egblas::bias_batch_mean");
#endif
}

} //end of namespace etl::impl::egblas
