//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the sign operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision sign.
 */
#ifdef EGBLAS_HAS_SSIGN
static constexpr bool has_ssign = true;
#else
static constexpr bool has_ssign = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas sign operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sign([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] float* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SSIGN
    inc_counter("egblas");
    egblas_ssign(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sign");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision sign.
 */
#ifdef EGBLAS_HAS_DSIGN
static constexpr bool has_dsign = true;
#else
static constexpr bool has_dsign = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas sign operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sign([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] double* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DSIGN
    inc_counter("egblas");
    egblas_dsign(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sign");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision sign.
 */
#ifdef EGBLAS_HAS_CSIGN
static constexpr bool has_csign = true;
#else
static constexpr bool has_csign = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas sign operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sign([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<float> alpha,
                 [[maybe_unused]] std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSIGN
    inc_counter("egblas");
    egblas_csign(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sign");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas sign operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sign([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<float> alpha,
                 [[maybe_unused]] etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSIGN
    inc_counter("egblas");
    egblas_csign(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sign");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision sign.
 */
#ifdef EGBLAS_HAS_ZSIGN
static constexpr bool has_zsign = true;
#else
static constexpr bool has_zsign = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas sign operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sign([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<double> alpha,
                 [[maybe_unused]] std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSIGN
    inc_counter("egblas");
    egblas_zsign(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sign");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas sign operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sign([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<double> alpha,
                 [[maybe_unused]] etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSIGN
    inc_counter("egblas");
    egblas_zsign(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sign");
#endif
}

} //end of namespace etl::impl::egblas
