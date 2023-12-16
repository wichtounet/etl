//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the tanh operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has tanhgle-precision hyperbolic tangent.
 */
#ifdef EGBLAS_HAS_STANH
static constexpr bool has_stanh = true;
#else
static constexpr bool has_stanh = false;
#endif

/*!
 * \brief Wrappers for tanhgle-precision egblas tanh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void tanh([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] float* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_STANH
    inc_counter("egblas");
    egblas_stanh(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::tanh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision hyperbolic tangent.
 */
#ifdef EGBLAS_HAS_DTANH
static constexpr bool has_dtanh = true;
#else
static constexpr bool has_dtanh = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas tanh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void tanh([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] double* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DTANH
    inc_counter("egblas");
    egblas_dtanh(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::tanh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex tanhgle-precision hyperbolic tangent.
 */
#ifdef EGBLAS_HAS_CTANH
static constexpr bool has_ctanh = true;
#else
static constexpr bool has_ctanh = false;
#endif

/*!
 * \brief Wrappers for complex tanhgle-precision egblas tanh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void tanh([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<float> alpha,
                 [[maybe_unused]] std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CTANH
    inc_counter("egblas");
    egblas_ctanh(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::tanh");
#endif
}

/*!
 * \brief Wrappers for complex tanhgle-precision egblas tanh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void tanh([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<float> alpha,
                 [[maybe_unused]] etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CTANH
    inc_counter("egblas");
    egblas_ctanh(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::tanh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision hyperbolic tangent.
 */
#ifdef EGBLAS_HAS_ZTANH
static constexpr bool has_ztanh = true;
#else
static constexpr bool has_ztanh = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas tanh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void tanh([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<double> alpha,
                 [[maybe_unused]] std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZTANH
    inc_counter("egblas");
    egblas_ztanh(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::tanh");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas tanh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void tanh([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<double> alpha,
                 [[maybe_unused]] etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZTANH
    inc_counter("egblas");
    egblas_ztanh(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::tanh");
#endif
}

} //end of namespace etl::impl::egblas
