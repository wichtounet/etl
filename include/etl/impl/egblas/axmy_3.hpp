//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the axmy_3 operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAXMY_3
static constexpr bool has_saxmy_3 = true;
#else
static constexpr bool has_saxmy_3 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas axmy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axmy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] float alpha,
                   [[maybe_unused]] float* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] float* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] float* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SAXMY_3
    inc_counter("egblas");
    egblas_saxmy_3(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::axmy_3");
#endif
}

#ifdef EGBLAS_HAS_DAXMY_3
static constexpr bool has_daxmy_3 = true;
#else
static constexpr bool has_daxmy_3 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas axmy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axmy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] double alpha,
                   [[maybe_unused]] double* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] double* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] double* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DAXMY_3
    inc_counter("egblas");
    egblas_daxmy_3(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::axmy_3");
#endif
}

#ifdef EGBLAS_HAS_CAXMY_3
static constexpr bool has_caxmy_3 = true;
#else
static constexpr bool has_caxmy_3 = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas axmy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axmy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] std::complex<float> alpha,
                   [[maybe_unused]] std::complex<float>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] std::complex<float>* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] std::complex<float>* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CAXMY_3
    inc_counter("egblas");
    egblas_caxmy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axmy_3");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas axmy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axmy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] etl::complex<float> alpha,
                   [[maybe_unused]] etl::complex<float>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] etl::complex<float>* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] etl::complex<float>* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CAXMY_3
    inc_counter("egblas");
    egblas_caxmy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axmy_3");
#endif
}

#ifdef EGBLAS_HAS_ZAXMY_3
static constexpr bool has_zaxmy_3 = true;
#else
static constexpr bool has_zaxmy_3 = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas axmy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axmy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] std::complex<double> alpha,
                   [[maybe_unused]] std::complex<double>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] std::complex<double>* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] std::complex<double>* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZAXMY_3
    inc_counter("egblas");
    egblas_zaxmy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
                   reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axmy_3");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas axmy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axmy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] etl::complex<double> alpha,
                   [[maybe_unused]] etl::complex<double>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] etl::complex<double>* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] etl::complex<double>* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZAXMY_3
    inc_counter("egblas");
    egblas_zaxmy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
                   reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axmy_3");
#endif
}

} //end of namespace etl::impl::egblas
