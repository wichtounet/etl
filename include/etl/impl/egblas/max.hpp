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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] float alpha,
                [[maybe_unused]] float* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] float* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SMAX
    inc_counter("egblas");
    egblas_smax(n, alpha, A, lda, B, ldb);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] double alpha,
                [[maybe_unused]] double* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] double* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DMAX
    inc_counter("egblas");
    egblas_dmax(n, alpha, A, lda, B, ldb);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<float> alpha,
                [[maybe_unused]] std::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CMAX
    inc_counter("egblas");
    egblas_cmax(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<float> alpha,
                [[maybe_unused]] etl::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CMAX
    inc_counter("egblas");
    egblas_cmax(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<double> alpha,
                [[maybe_unused]] std::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZMAX
    inc_counter("egblas");
    egblas_zmax(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<double> alpha,
                [[maybe_unused]] etl::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZMAX
    inc_counter("egblas");
    egblas_zmax(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] float alpha,
                [[maybe_unused]] float* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] float* B,
                [[maybe_unused]] size_t ldb,
                [[maybe_unused]] float* C,
                [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SMAX3
    inc_counter("egblas");
    egblas_smax(n, alpha, A, lda, B, ldb, C, ldc);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] double alpha,
                [[maybe_unused]] double* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] double* B,
                [[maybe_unused]] size_t ldb,
                [[maybe_unused]] double* C,
                [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DMAX3
    inc_counter("egblas");
    egblas_dmax(n, alpha, A, lda, B, ldb, C, ldc);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<float> alpha,
                [[maybe_unused]] std::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<float>* B,
                [[maybe_unused]] size_t ldb,
                [[maybe_unused]] std::complex<float>* C,
                [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CMAX3
    inc_counter("egblas");
    egblas_cmax(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<float> alpha,
                [[maybe_unused]] etl::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<float>* B,
                [[maybe_unused]] size_t ldb,
                [[maybe_unused]] etl::complex<float>* C,
                [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CMAX3
    inc_counter("egblas");
    egblas_cmax(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<double> alpha,
                [[maybe_unused]] std::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<double>* B,
                [[maybe_unused]] size_t ldb,
                [[maybe_unused]] std::complex<double>* C,
                [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZMAX3
    inc_counter("egblas");
    egblas_zmax(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
                reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
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
inline void max([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<double> alpha,
                [[maybe_unused]] etl::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<double>* B,
                [[maybe_unused]] size_t ldb,
                [[maybe_unused]] etl::complex<double>* C,
                [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZMAX3
    inc_counter("egblas");
    egblas_zmax(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
                reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::max");
#endif
}

} //end of namespace etl::impl::egblas
