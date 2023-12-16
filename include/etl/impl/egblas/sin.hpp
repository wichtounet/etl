//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the sin operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision sinus.
 */
#ifdef EGBLAS_HAS_SSIN
static constexpr bool has_ssin = true;
#else
static constexpr bool has_ssin = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas sin operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sin([[maybe_unused]] size_t n,
                [[maybe_unused]] float alpha,
                [[maybe_unused]] float* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] float* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SSIN
    inc_counter("egblas");
    egblas_ssin(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sin");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision sinus.
 */
#ifdef EGBLAS_HAS_DSIN
static constexpr bool has_dsin = true;
#else
static constexpr bool has_dsin = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas sin operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sin([[maybe_unused]] size_t n,
                [[maybe_unused]] double alpha,
                [[maybe_unused]] double* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] double* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DSIN
    inc_counter("egblas");
    egblas_dsin(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sin");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision sinus.
 */
#ifdef EGBLAS_HAS_CSIN
static constexpr bool has_csin = true;
#else
static constexpr bool has_csin = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas sin operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sin([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<float> alpha,
                [[maybe_unused]] std::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSIN
    inc_counter("egblas");
    egblas_csin(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sin");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas sin operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sin([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<float> alpha,
                [[maybe_unused]] etl::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSIN
    inc_counter("egblas");
    egblas_csin(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sin");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision sinus.
 */
#ifdef EGBLAS_HAS_ZSIN
static constexpr bool has_zsin = true;
#else
static constexpr bool has_zsin = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas sin operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sin([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<double> alpha,
                [[maybe_unused]] std::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSIN
    inc_counter("egblas");
    egblas_zsin(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sin");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas sin operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sin([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<double> alpha,
                [[maybe_unused]] etl::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSIN
    inc_counter("egblas");
    egblas_zsin(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sin");
#endif
}

} //end of namespace etl::impl::egblas
