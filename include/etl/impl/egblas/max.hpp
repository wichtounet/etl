//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the max operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision max.
 */
#ifdef EGBLAS_HAS_SMAX
static constexpr bool has_smax = true;
#else
static constexpr bool has_smax = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(size_t n, float alpha, float* A, size_t lda, float* B, size_t ldb) {
#ifdef EGBLAS_HAS_SMAX
    inc_counter("egblas");
    egblas_smax(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision max.
 */
#ifdef EGBLAS_HAS_DMAX
static constexpr bool has_dmax = true;
#else
static constexpr bool has_dmax = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(size_t n, double alpha, double* A, size_t lda, double* B, size_t ldb) {
#ifdef EGBLAS_HAS_DMAX
    inc_counter("egblas");
    egblas_dmax(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision max.
 */
#ifdef EGBLAS_HAS_CMAX
static constexpr bool has_cmax = true;
#else
static constexpr bool has_cmax = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(size_t n, std::complex<float> alpha, std::complex<float>* A, size_t lda, std::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CMAX
    inc_counter("egblas");
    egblas_cmax(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(size_t n, etl::complex<float> alpha, etl::complex<float>* A, size_t lda, etl::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CMAX
    inc_counter("egblas");
    egblas_cmax(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision max.
 */
#ifdef EGBLAS_HAS_ZMAX
static constexpr bool has_zmax = true;
#else
static constexpr bool has_zmax = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(size_t n, std::complex<double> alpha, std::complex<double>* A, size_t lda, std::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZMAX
    inc_counter("egblas");
    egblas_zmax(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(size_t n, etl::complex<double> alpha, etl::complex<double>* A, size_t lda, etl::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZMAX
    inc_counter("egblas");
    egblas_zmax(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Indicates if EGBLAS has single-precision max.
 */
#ifdef EGBLAS_HAS_SMAX3
static constexpr bool has_smax3 = true;
#else
static constexpr bool has_smax3 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(size_t n, float alpha, float* A, size_t lda, float* B, size_t ldb, float* C, size_t ldc) {
#ifdef EGBLAS_HAS_SMAX3
    inc_counter("egblas");
    egblas_smax(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision max.
 */
#ifdef EGBLAS_HAS_DMAX3
static constexpr bool has_dmax3 = true;
#else
static constexpr bool has_dmax3 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(size_t n, double alpha, double* A, size_t lda, double* B, size_t ldb, double* C, size_t ldc) {
#ifdef EGBLAS_HAS_DMAX3
    inc_counter("egblas");
    egblas_dmax(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision max.
 */
#ifdef EGBLAS_HAS_CMAX3
static constexpr bool has_cmax3 = true;
#else
static constexpr bool has_cmax3 = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(
    size_t n, std::complex<float> alpha, std::complex<float>* A, size_t lda, std::complex<float>* B, size_t ldb, std::complex<float>* C, size_t ldc) {
#ifdef EGBLAS_HAS_CMAX3
    inc_counter("egblas");
    egblas_cmax(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(
    size_t n, etl::complex<float> alpha, etl::complex<float>* A, size_t lda, etl::complex<float>* B, size_t ldb, etl::complex<float>* C, size_t ldc) {
#ifdef EGBLAS_HAS_CMAX3
    inc_counter("egblas");
    egblas_cmax(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision max.
 */
#ifdef EGBLAS_HAS_ZMAX3
static constexpr bool has_zmax3 = true;
#else
static constexpr bool has_zmax3 = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(
    size_t n, std::complex<double> alpha, std::complex<double>* A, size_t lda, std::complex<double>* B, size_t ldb, std::complex<double>* C, size_t ldc) {
#ifdef EGBLAS_HAS_ZMAX3
    inc_counter("egblas");
    egblas_zmax(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
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

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas max operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void max(
    size_t n, etl::complex<double> alpha, etl::complex<double>* A, size_t lda, etl::complex<double>* B, size_t ldb, etl::complex<double>* C, size_t ldc) {
#ifdef EGBLAS_HAS_ZMAX3
    inc_counter("egblas");
    egblas_zmax(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
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

    cpp_unreachable("Invalid call to egblas::max");
#endif
}

} //end of namespace etl::impl::egblas
