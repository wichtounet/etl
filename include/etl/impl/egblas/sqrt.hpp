//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the sqrt operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision square root.
 */
#ifdef EGBLAS_HAS_SSQRT
static constexpr bool has_ssqrt = true;
#else
static constexpr bool has_ssqrt = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas sqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sqrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] float* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SSQRT
    inc_counter("egblas");
    egblas_ssqrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision square root.
 */
#ifdef EGBLAS_HAS_DSQRT
static constexpr bool has_dsqrt = true;
#else
static constexpr bool has_dsqrt = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas sqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sqrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] double* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DSQRT
    inc_counter("egblas");
    egblas_dsqrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision square
 * root.
 */
#ifdef EGBLAS_HAS_CSQRT
static constexpr bool has_csqrt = true;
#else
static constexpr bool has_csqrt = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas sqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sqrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<float> alpha,
                 [[maybe_unused]] std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSQRT
    inc_counter("egblas");
    egblas_csqrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sqrt");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas sqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sqrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<float> alpha,
                 [[maybe_unused]] etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSQRT
    inc_counter("egblas");
    egblas_csqrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision square
 * root.
 */
#ifdef EGBLAS_HAS_ZSQRT
static constexpr bool has_zsqrt = true;
#else
static constexpr bool has_zsqrt = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas sqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sqrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<double> alpha,
                 [[maybe_unused]] std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSQRT
    inc_counter("egblas");
    egblas_zsqrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sqrt");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas sqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sqrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<double> alpha,
                 [[maybe_unused]] etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSQRT
    inc_counter("egblas");
    egblas_zsqrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sqrt");
#endif
}

} //end of namespace etl::impl::egblas
