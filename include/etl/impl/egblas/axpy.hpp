//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the axpy operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAXPY
static constexpr bool has_saxpy = true;
#else
static constexpr bool has_saxpy = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas axpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] const float* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SAXPY
    inc_counter("egblas");
    egblas_saxpy(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpy");
#endif
}

#ifdef EGBLAS_HAS_DAXPY
static constexpr bool has_daxpy = true;
#else
static constexpr bool has_daxpy = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas axpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] const double* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DAXPY
    inc_counter("egblas");
    egblas_daxpy(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpy");
#endif
}

#ifdef EGBLAS_HAS_CAXPY
static constexpr bool has_caxpy = true;
#else
static constexpr bool has_caxpy = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas axpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<float> alpha,
                 [[maybe_unused]] const std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CAXPY
    inc_counter("egblas");
    egblas_caxpy(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpy");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas axpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<float> alpha,
                 [[maybe_unused]] const etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CAXPY
    inc_counter("egblas");
    egblas_caxpy(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpy");
#endif
}

#ifdef EGBLAS_HAS_ZAXPY
static constexpr bool has_zaxpy = true;
#else
static constexpr bool has_zaxpy = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas axpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<double> alpha,
                 [[maybe_unused]] const std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZAXPY
    inc_counter("egblas");
    egblas_zaxpy(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpy");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas axpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpy([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<double> alpha,
                 [[maybe_unused]] const etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZAXPY
    inc_counter("egblas");
    egblas_zaxpy(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpy");
#endif
}

} //end of namespace etl::impl::egblas
