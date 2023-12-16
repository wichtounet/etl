//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the cbrt operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision cubic root.
 */
#ifdef EGBLAS_HAS_SCBRT
static constexpr bool has_scbrt = true;
#else
static constexpr bool has_scbrt = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas cbrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cbrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] float* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SCBRT
    inc_counter("egblas");
    egblas_scbrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::cbrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision cubic root.
 */
#ifdef EGBLAS_HAS_DCBRT
static constexpr bool has_dcbrt = true;
#else
static constexpr bool has_dcbrt = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas cbrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cbrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] double* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DCBRT
    inc_counter("egblas");
    egblas_dcbrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::cbrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision cubic root.
 */
#ifdef EGBLAS_HAS_CCBRT
static constexpr bool has_ccbrt = true;
#else
static constexpr bool has_ccbrt = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas cbrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cbrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<float> alpha,
                 [[maybe_unused]] std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CCBRT
    inc_counter("egblas");
    egblas_ccbrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::cbrt");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas cbrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cbrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<float> alpha,
                 [[maybe_unused]] etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CCBRT
    inc_counter("egblas");
    egblas_ccbrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::cbrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision cubic root.
 */
#ifdef EGBLAS_HAS_ZCBRT
static constexpr bool has_zcbrt = true;
#else
static constexpr bool has_zcbrt = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas cbrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cbrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<double> alpha,
                 [[maybe_unused]] std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZCBRT
    inc_counter("egblas");
    egblas_zcbrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::cbrt");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas cbrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cbrt([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<double> alpha,
                 [[maybe_unused]] etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZCBRT
    inc_counter("egblas");
    egblas_zcbrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::cbrt");
#endif
}

} //end of namespace etl::impl::egblas
