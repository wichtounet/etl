//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the axdy_3 operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAXDY_3
static constexpr bool has_saxdy_3 = true;
#else
static constexpr bool has_saxdy_3 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas axdy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy_3(size_t n, float alpha, float* A, size_t lda, float* B, size_t ldb, float* C, size_t ldc) {
#ifdef EGBLAS_HAS_SAXDY_3
    inc_counter("egblas");
    egblas_saxdy_3(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::axdy_3");
#endif
}

#ifdef EGBLAS_HAS_DAXDY_3
static constexpr bool has_daxdy_3 = true;
#else
static constexpr bool has_daxdy_3 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas axdy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy_3(size_t n, double alpha, double* A, size_t lda, double* B, size_t ldb, double* C, size_t ldc) {
#ifdef EGBLAS_HAS_DAXDY_3
    inc_counter("egblas");
    egblas_daxdy_3(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::axdy_3");
#endif
}

#ifdef EGBLAS_HAS_CAXDY_3
static constexpr bool has_caxdy_3 = true;
#else
static constexpr bool has_caxdy_3 = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas axdy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy_3(
    size_t n, std::complex<float> alpha, std::complex<float>* A, size_t lda, std::complex<float>* B, size_t ldb, std::complex<float>* C, size_t ldc) {
#ifdef EGBLAS_HAS_CAXDY_3
    inc_counter("egblas");
    egblas_caxdy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::axdy_3");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas axdy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy_3(
    size_t n, etl::complex<float> alpha, etl::complex<float>* A, size_t lda, etl::complex<float>* B, size_t ldb, etl::complex<float>* C, size_t ldc) {
#ifdef EGBLAS_HAS_CAXDY_3
    inc_counter("egblas");
    egblas_caxdy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::axdy_3");
#endif
}

#ifdef EGBLAS_HAS_ZAXDY_3
static constexpr bool has_zaxdy_3 = true;
#else
static constexpr bool has_zaxdy_3 = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas axdy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy_3(
    size_t n, std::complex<double> alpha, std::complex<double>* A, size_t lda, std::complex<double>* B, size_t ldb, std::complex<double>* C, size_t ldc) {
#ifdef EGBLAS_HAS_ZAXDY_3
    inc_counter("egblas");
    egblas_zaxdy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
                   reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::axdy_3");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas axdy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy_3(
    size_t n, etl::complex<double> alpha, etl::complex<double>* A, size_t lda, etl::complex<double>* B, size_t ldb, etl::complex<double>* C, size_t ldc) {
#ifdef EGBLAS_HAS_ZAXDY_3
    inc_counter("egblas");
    egblas_zaxdy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
                   reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::axdy_3");
#endif
}

} //end of namespace etl::impl::egblas
