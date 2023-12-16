//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the clip operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision clip.
 */
#ifdef EGBLAS_HAS_SCLIP
static constexpr bool has_sclip = true;
#else
static constexpr bool has_sclip = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas clip operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip([[maybe_unused]] size_t n, [[maybe_unused]] const float alpha, [[maybe_unused]] const float* A, [[maybe_unused]] size_t lda, [[maybe_unused]] const float* B, [[maybe_unused]] size_t ldb, [[maybe_unused]] float* C, [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SCLIP
    inc_counter("egblas");
    egblas_sclip(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::clip");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision clip.
 */
#ifdef EGBLAS_HAS_DCLIP
static constexpr bool has_dclip = true;
#else
static constexpr bool has_dclip = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas clip operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip([[maybe_unused]] size_t n, [[maybe_unused]] double alpha, [[maybe_unused]] const double* A, [[maybe_unused]] size_t lda, [[maybe_unused]] const double* B, [[maybe_unused]] size_t ldb, [[maybe_unused]] double* C, [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DCLIP
    inc_counter("egblas");
    egblas_dclip(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::clip");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision clip.
 */
#ifdef EGBLAS_HAS_CCLIP
static constexpr bool has_cclip = true;
#else
static constexpr bool has_cclip = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas clip operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip([[maybe_unused]] size_t n,
                 [[maybe_unused]] const std::complex<float> alpha,
                 [[maybe_unused]] const std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const std::complex<float>* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] std::complex<float>* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CCLIP
    inc_counter("egblas");
    egblas_cclip(n, complex_cast(alpha), reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb,
                 reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::clip");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas clip operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip([[maybe_unused]] size_t n,
                 [[maybe_unused]] const etl::complex<float> alpha,
                 [[maybe_unused]] const etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] etl::complex<float>* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CCLIP
    inc_counter("egblas");
    egblas_cclip(n, complex_cast(alpha), reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb,
                 reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::clip");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision clip.
 */
#ifdef EGBLAS_HAS_ZCLIP
static constexpr bool has_zclip = true;
#else
static constexpr bool has_zclip = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas clip operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip([[maybe_unused]] size_t n,
                 [[maybe_unused]] const std::complex<double> alpha,
                 [[maybe_unused]] const std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const std::complex<double>* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] std::complex<double>* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZCLIP
    inc_counter("egblas");
    egblas_zclip(n, complex_cast(alpha), reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                 reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::clip");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas clip operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip([[maybe_unused]] size_t n,
                 [[maybe_unused]] const etl::complex<double> alpha,
                 [[maybe_unused]] const etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] const etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb,
                 [[maybe_unused]] etl::complex<double>* C,
                 [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZCLIP
    inc_counter("egblas");
    egblas_zclip(n, complex_cast(alpha), reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                 reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::clip");
#endif
}

} //end of namespace etl::impl::egblas
