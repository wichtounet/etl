//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the axpy_3 operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAXPY_3
static constexpr bool has_saxpy_3 = true;
#else
static constexpr bool has_saxpy_3 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas axpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] float alpha,
                   [[maybe_unused]] float* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] float* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] float* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SAXPY_3
    etl::inc_counter("egblas");
    egblas_saxpy_3(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::axpy_3");
#endif
}

#ifdef EGBLAS_HAS_DAXPY_3
static constexpr bool has_daxpy_3 = true;
#else
static constexpr bool has_daxpy_3 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas axpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] double alpha,
                   [[maybe_unused]] double* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] double* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] double* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DAXPY_3
    etl::inc_counter("egblas");
    egblas_daxpy_3(n, alpha, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::axpy_3");
#endif
}

#ifdef EGBLAS_HAS_CAXPY_3
static constexpr bool has_caxpy_3 = true;
#else
static constexpr bool has_caxpy_3 = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas axpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] std::complex<float> alpha,
                   [[maybe_unused]] std::complex<float>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] std::complex<float>* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] std::complex<float>* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CAXPY_3
    etl::inc_counter("egblas");
    egblas_caxpy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axpy_3");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas axpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] etl::complex<float> alpha,
                   [[maybe_unused]] etl::complex<float>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] etl::complex<float>* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] etl::complex<float>* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CAXPY_3
    etl::inc_counter("egblas");
    egblas_caxpy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb, reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axpy_3");
#endif
}

#ifdef EGBLAS_HAS_ZAXPY_3
static constexpr bool has_zaxpy_3 = true;
#else
static constexpr bool has_zaxpy_3 = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas axpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] std::complex<double> alpha,
                   [[maybe_unused]] std::complex<double>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] std::complex<double>* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] std::complex<double>* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZAXPY_3
    etl::inc_counter("egblas");
    egblas_zaxpy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
                   reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axpy_3");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas axpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy_3([[maybe_unused]] size_t n,
                   [[maybe_unused]] etl::complex<double> alpha,
                   [[maybe_unused]] etl::complex<double>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] etl::complex<double>* B,
                   [[maybe_unused]] size_t ldb,
                   [[maybe_unused]] etl::complex<double>* C,
                   [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZAXPY_3
    etl::inc_counter("egblas");
    egblas_zaxpy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb,
                   reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axpy_3");
#endif
}

} //end of namespace etl::impl::egblas
