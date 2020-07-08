//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the batch_k_scale operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SBATCH_K_SCALE2
static constexpr bool has_sbatch_k_scale2 = true;
#else
static constexpr bool has_sbatch_k_scale2 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas batch_k_scale operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void batch_k_scale([[maybe_unused]] size_t       b,
                          [[maybe_unused]] size_t       k,
                          [[maybe_unused]] const float* A,
                          [[maybe_unused]] const float* gamma,
                          [[maybe_unused]] float*       B) {
#ifdef EGBLAS_HAS_SBATCH_K_SCALE2
    inc_counter("egblas");
    egblas_sbatch_k_scale2(b, k, A, gamma, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_scale");
#endif
}

#ifdef EGBLAS_HAS_DBATCH_K_SCALE2
static constexpr bool has_dbatch_k_scale2 = true;
#else
static constexpr bool has_dbatch_k_scale2 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas batch_k_scale operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void batch_k_scale([[maybe_unused]] size_t        b,
                          [[maybe_unused]] size_t        k,
                          [[maybe_unused]] const double* A,
                          [[maybe_unused]] const double* gamma,
                          [[maybe_unused]] double*       B) {
#ifdef EGBLAS_HAS_DBATCH_K_SCALE2
    inc_counter("egblas");
    egblas_dbatch_k_scale2(b, k, A, gamma, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_scale");
#endif
}

#ifdef EGBLAS_HAS_SBATCH_K_SCALE4
static constexpr bool has_sbatch_k_scale4 = true;
#else
static constexpr bool has_sbatch_k_scale4 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas batch_k_scale operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void batch_k_scale([[maybe_unused]] size_t       b,
                          [[maybe_unused]] size_t       k,
                          [[maybe_unused]] size_t       m,
                          [[maybe_unused]] size_t       n,
                          [[maybe_unused]] const float* A,
                          [[maybe_unused]] const float* gamma,
                          [[maybe_unused]] float*       B) {
#ifdef EGBLAS_HAS_SBATCH_K_SCALE4
    inc_counter("egblas");
    egblas_sbatch_k_scale4(b, k, m, n, A, gamma, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_scale");
#endif
}

#ifdef EGBLAS_HAS_DBATCH_K_SCALE4
static constexpr bool has_dbatch_k_scale4 = true;
#else
static constexpr bool has_dbatch_k_scale4 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas batch_k_scale operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void batch_k_scale([[maybe_unused]] size_t        b,
                          [[maybe_unused]] size_t        k,
                          [[maybe_unused]] size_t        m,
                          [[maybe_unused]] size_t        n,
                          [[maybe_unused]] const double* A,
                          [[maybe_unused]] const double* gamma,
                          [[maybe_unused]] double*       B) {
#ifdef EGBLAS_HAS_DBATCH_K_SCALE4
    inc_counter("egblas");
    egblas_dbatch_k_scale4(b, k, m, n, A, gamma, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_scale");
#endif
}

// batch_k_scale_plus

#ifdef EGBLAS_HAS_SBATCH_K_SCALE_PLUS2
static constexpr bool has_sbatch_k_scale_plus2 = true;
#else
static constexpr bool has_sbatch_k_scale_plus2 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas batch_k_scale_plus operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void batch_k_scale_plus([[maybe_unused]] size_t       b,
                               [[maybe_unused]] size_t       k,
                               [[maybe_unused]] const float* A,
                               [[maybe_unused]] const float* gamma,
                               [[maybe_unused]] const float* beta,
                               [[maybe_unused]] float*       B) {
#ifdef EGBLAS_HAS_SBATCH_K_SCALE_PLUS2
    inc_counter("egblas");
    egblas_sbatch_k_scale_plus2(b, k, A, gamma, beta, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_scale_plus");
#endif
}

#ifdef EGBLAS_HAS_DBATCH_K_SCALE_PLUS2
static constexpr bool has_dbatch_k_scale_plus2 = true;
#else
static constexpr bool has_dbatch_k_scale_plus2 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas batch_k_scale_plus operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void batch_k_scale_plus([[maybe_unused]] size_t        b,
                               [[maybe_unused]] size_t        k,
                               [[maybe_unused]] const double* A,
                               [[maybe_unused]] const double* gamma,
                               [[maybe_unused]] const double* beta,
                               [[maybe_unused]] double*       B) {
#ifdef EGBLAS_HAS_DBATCH_K_SCALE_PLUS2
    inc_counter("egblas");
    egblas_dbatch_k_scale_plus2(b, k, A, gamma, beta, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_scale_plus");
#endif
}

#ifdef EGBLAS_HAS_SBATCH_K_SCALE_PLUS4
static constexpr bool has_sbatch_k_scale_plus4 = true;
#else
static constexpr bool has_sbatch_k_scale_plus4 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas batch_k_scale_plus operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void batch_k_scale_plus([[maybe_unused]] size_t       b,
                               [[maybe_unused]] size_t       k,
                               [[maybe_unused]] size_t       m,
                               [[maybe_unused]] size_t       n,
                               [[maybe_unused]] const float* A,
                               [[maybe_unused]] const float* gamma,
                               [[maybe_unused]] const float* beta,
                               [[maybe_unused]] float*       B) {
#ifdef EGBLAS_HAS_SBATCH_K_SCALE_PLUS4
    inc_counter("egblas");
    egblas_sbatch_k_scale_plus4(b, k, m, n, A, gamma, beta, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_scale_plus");
#endif
}

#ifdef EGBLAS_HAS_DBATCH_K_SCALE_PLUS4
static constexpr bool has_dbatch_k_scale_plus4 = true;
#else
static constexpr bool has_dbatch_k_scale_plus4 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas batch_k_scale_plus operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void batch_k_scale_plus([[maybe_unused]] size_t        b,
                               [[maybe_unused]] size_t        k,
                               [[maybe_unused]] size_t        m,
                               [[maybe_unused]] size_t        n,
                               [[maybe_unused]] const double* A,
                               [[maybe_unused]] const double* gamma,
                               [[maybe_unused]] const double* beta,
                               [[maybe_unused]] double*       B) {
#ifdef EGBLAS_HAS_DBATCH_K_SCALE_PLUS4
    inc_counter("egblas");
    egblas_dbatch_k_scale_plus4(b, k, m, n, A, gamma, beta, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_scale_plus");
#endif
}

#ifdef EGBLAS_HAS_SBATCH_K_MINUS_SCALE2
static constexpr bool has_sbatch_k_minus_scale2 = true;
#else
static constexpr bool has_sbatch_k_minus_scale2 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas batch_k_minus_scale operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void batch_k_minus_scale([[maybe_unused]] size_t       b,
                                [[maybe_unused]] size_t       k,
                                [[maybe_unused]] const float* A,
                                [[maybe_unused]] const float* gamma,
                                [[maybe_unused]] const float* beta,
                                [[maybe_unused]] float*       B) {
#ifdef EGBLAS_HAS_SBATCH_K_MINUS_SCALE2
    inc_counter("egblas");
    egblas_sbatch_k_minus_scale2(b, k, A, gamma, beta, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_minus_scale");
#endif
}

#ifdef EGBLAS_HAS_DBATCH_K_MINUS_SCALE2
static constexpr bool has_dbatch_k_minus_scale2 = true;
#else
static constexpr bool has_dbatch_k_minus_scale2 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas batch_k_minus_scale operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 */
inline void batch_k_minus_scale([[maybe_unused]] size_t        b,
                                [[maybe_unused]] size_t        k,
                                [[maybe_unused]] const double* A,
                                [[maybe_unused]] const double* gamma,
                                [[maybe_unused]] const double* beta,
                                [[maybe_unused]] double*       B) {
#ifdef EGBLAS_HAS_DBATCH_K_MINUS_SCALE2
    inc_counter("egblas");
    egblas_dbatch_k_minus_scale2(b, k, A, gamma, beta, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_minus_scale");
#endif
}

#ifdef EGBLAS_HAS_SBATCH_K_MINUS_SCALE4
static constexpr bool has_sbatch_k_minus_scale4 = true;
#else
static constexpr bool has_sbatch_k_minus_scale4 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas batch_k_minus_scale operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void batch_k_minus_scale([[maybe_unused]] size_t       b,
                                [[maybe_unused]] size_t       k,
                                [[maybe_unused]] size_t       m,
                                [[maybe_unused]] size_t       n,
                                [[maybe_unused]] const float* A,
                                [[maybe_unused]] const float* gamma,
                                [[maybe_unused]] const float* beta,
                                [[maybe_unused]] float*       B) {
#ifdef EGBLAS_HAS_SBATCH_K_MINUS_SCALE4
    inc_counter("egblas");
    egblas_sbatch_k_minus_scale4(b, k, m, n, A, gamma, beta, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_minus_scale");
#endif
}

#ifdef EGBLAS_HAS_DBATCH_K_MINUS_SCALE4
static constexpr bool has_dbatch_k_minus_scale4 = true;
#else
static constexpr bool has_dbatch_k_minus_scale4 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas batch_k_minus_scale operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void batch_k_minus_scale([[maybe_unused]] size_t        b,
                                [[maybe_unused]] size_t        k,
                                [[maybe_unused]] size_t        m,
                                [[maybe_unused]] size_t        n,
                                [[maybe_unused]] const double* A,
                                [[maybe_unused]] const double* gamma,
                                [[maybe_unused]] const double* beta,
                                [[maybe_unused]] double*       B) {
#ifdef EGBLAS_HAS_DBATCH_K_MINUS_SCALE4
    inc_counter("egblas");
    egblas_dbatch_k_minus_scale4(b, k, m, n, A, gamma, beta, B);
#else
    cpp_unreachable("Invalid call to egblas::batch_k_minus_scale");
#endif
}

} //end of namespace etl::impl::egblas
