//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
inline void cbrt(size_t n, float alpha, float* A, size_t lda, float* B, size_t ldb) {
#ifdef EGBLAS_HAS_SCBRT
    inc_counter("egblas");
    egblas_scbrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void cbrt(size_t n, double alpha, double* A, size_t lda, double* B, size_t ldb) {
#ifdef EGBLAS_HAS_DCBRT
    inc_counter("egblas");
    egblas_dcbrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void cbrt(size_t n, std::complex<float> alpha, std::complex<float>* A, size_t lda, std::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CCBRT
    inc_counter("egblas");
    egblas_ccbrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void cbrt(size_t n, etl::complex<float> alpha, etl::complex<float>* A, size_t lda, etl::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CCBRT
    inc_counter("egblas");
    egblas_ccbrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void cbrt(size_t n, std::complex<double> alpha, std::complex<double>* A, size_t lda, std::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZCBRT
    inc_counter("egblas");
    egblas_zcbrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void cbrt(size_t n, etl::complex<double> alpha, etl::complex<double>* A, size_t lda, etl::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZCBRT
    inc_counter("egblas");
    egblas_zcbrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::cbrt");
#endif
}

} //end of namespace etl::impl::egblas
