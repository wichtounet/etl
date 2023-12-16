//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the less operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision less.
 */
#ifdef EGBLAS_HAS_SLESS
static constexpr bool has_sless = true;
#else
static constexpr bool has_sless = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas less operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less([[maybe_unused]] size_t n,
                 [[maybe_unused]] const float* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const float* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] bool* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SLESS
    inc_counter("egblas");
    egblas_sless(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision less.
 */
#ifdef EGBLAS_HAS_DLESS
static constexpr bool has_dless = true;
#else
static constexpr bool has_dless = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas less operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less([[maybe_unused]] size_t n,
                 [[maybe_unused]] const double* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const double* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] bool* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DLESS
    inc_counter("egblas");
    egblas_dless(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision less.
 */
#ifdef EGBLAS_HAS_CLESS
static constexpr bool has_cless = true;
#else
static constexpr bool has_cless = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas less operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less([[maybe_unused]] size_t n,
                 [[maybe_unused]] const std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const std::complex<float>* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] bool* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CLESS
    inc_counter("egblas");
    egblas_cless(n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas less operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less([[maybe_unused]] size_t n,
                 [[maybe_unused]] const etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] bool* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CLESS
    inc_counter("egblas");
    egblas_cless(n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision less.
 */
#ifdef EGBLAS_HAS_ZLESS
static constexpr bool has_zless = true;
#else
static constexpr bool has_zless = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas less operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less([[maybe_unused]] size_t n,
                 [[maybe_unused]] const std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const std::complex<double>* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] bool* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZLESS
    inc_counter("egblas");
    egblas_zless(n, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas less operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less([[maybe_unused]] size_t n,
                 [[maybe_unused]] const etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] bool* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZLESS
    inc_counter("egblas");
    egblas_zless(n, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less");
#endif
}

} //end of namespace etl::impl::egblas
