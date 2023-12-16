//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the equal operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision equal.
 */
#ifdef EGBLAS_HAS_SEQUAL
static constexpr bool has_sequal = true;
#else
static constexpr bool has_sequal = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void equal([[maybe_unused]] size_t n,
                  [[maybe_unused]] const float* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] const float* B,
                  [[maybe_unused]] size_t ldb,
                  [[maybe_unused]] bool* C,
                  [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SEQUAL
    inc_counter("egblas");
    egblas_sequal(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::equal");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision equal.
 */
#ifdef EGBLAS_HAS_DEQUAL
static constexpr bool has_dequal = true;
#else
static constexpr bool has_dequal = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void equal([[maybe_unused]] size_t n,
                  [[maybe_unused]] const double* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] const double* B,
                  [[maybe_unused]] size_t ldb,
                  [[maybe_unused]] bool* C,
                  [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DEQUAL
    inc_counter("egblas");
    egblas_dequal(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::equal");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision equal.
 */
#ifdef EGBLAS_HAS_CEQUAL
static constexpr bool has_cequal = true;
#else
static constexpr bool has_cequal = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void equal([[maybe_unused]] size_t n,
                  [[maybe_unused]] const std::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] const std::complex<float>* B,
                  [[maybe_unused]] size_t ldb,
                  [[maybe_unused]] bool* C,
                  [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CEQUAL
    inc_counter("egblas");
    egblas_cequal(n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::equal");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void equal([[maybe_unused]] size_t n,
                  [[maybe_unused]] const etl::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] const etl::complex<float>* B,
                  [[maybe_unused]] size_t ldb,
                  [[maybe_unused]] bool* C,
                  [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CEQUAL
    inc_counter("egblas");
    egblas_cequal(n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::equal");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision equal.
 */
#ifdef EGBLAS_HAS_ZEQUAL
static constexpr bool has_zequal = true;
#else
static constexpr bool has_zequal = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void equal([[maybe_unused]] size_t n,
                  [[maybe_unused]] const std::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] const std::complex<double>* B,
                  [[maybe_unused]] size_t ldb,
                  [[maybe_unused]] bool* C,
                  [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZEQUAL
    inc_counter("egblas");
    egblas_zequal(n, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::equal");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void equal([[maybe_unused]] size_t n,
                  [[maybe_unused]] const etl::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] const etl::complex<double>* B,
                  [[maybe_unused]] size_t ldb,
                  [[maybe_unused]] bool* C,
                  [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZEQUAL
    inc_counter("egblas");
    egblas_zequal(n, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::equal");
#endif
}

} //end of namespace etl::impl::egblas
