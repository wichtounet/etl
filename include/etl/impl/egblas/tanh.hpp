//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
inline void tanh(size_t n, float alpha, float* A, size_t lda, float* B, size_t ldb) {
#ifdef EGBLAS_HAS_STANH
    inc_counter("egblas");
    egblas_stanh(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void tanh(size_t n, double alpha, double* A, size_t lda, double* B, size_t ldb) {
#ifdef EGBLAS_HAS_DTANH
    inc_counter("egblas");
    egblas_dtanh(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void tanh(size_t n, std::complex<float> alpha, std::complex<float>* A, size_t lda, std::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CTANH
    inc_counter("egblas");
    egblas_ctanh(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void tanh(size_t n, etl::complex<float> alpha, etl::complex<float>* A, size_t lda, etl::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CTANH
    inc_counter("egblas");
    egblas_ctanh(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void tanh(size_t n, std::complex<double> alpha, std::complex<double>* A, size_t lda, std::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZTANH
    inc_counter("egblas");
    egblas_ztanh(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void tanh(size_t n, etl::complex<double> alpha, etl::complex<double>* A, size_t lda, etl::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZTANH
    inc_counter("egblas");
    egblas_ztanh(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::tanh");
#endif
}

} //end of namespace etl::impl::egblas
