//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the exp operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision exponentiation.
 */
#ifdef EGBLAS_HAS_SEXP
static constexpr bool has_sexp = true;
#else
static constexpr bool has_sexp = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas exp operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void exp(size_t n, float alpha, float* A, size_t lda, float* B, size_t ldb) {
#ifdef EGBLAS_HAS_SEXP
    inc_counter("egblas");
    egblas_sexp(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::exp");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision exponentiation.
 */
#ifdef EGBLAS_HAS_DEXP
static constexpr bool has_dexp = true;
#else
static constexpr bool has_dexp = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas exp operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void exp(size_t n, double alpha, double* A, size_t lda, double* B, size_t ldb) {
#ifdef EGBLAS_HAS_DEXP
    inc_counter("egblas");
    egblas_dexp(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::exp");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision exponentiation.
 */
#ifdef EGBLAS_HAS_CEXP
static constexpr bool has_cexp = true;
#else
static constexpr bool has_cexp = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas exp operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void exp(size_t n, std::complex<float> alpha, std::complex<float>* A, size_t lda, std::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CEXP
    inc_counter("egblas");
    egblas_cexp(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::exp");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas exp operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void exp(size_t n, etl::complex<float> alpha, etl::complex<float>* A, size_t lda, etl::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CEXP
    inc_counter("egblas");
    egblas_cexp(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::exp");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision exponentiation.
 */
#ifdef EGBLAS_HAS_ZEXP
static constexpr bool has_zexp = true;
#else
static constexpr bool has_zexp = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas exp operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void exp(size_t n, std::complex<double> alpha, std::complex<double>* A, size_t lda, std::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZEXP
    inc_counter("egblas");
    egblas_zexp(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::exp");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas exp operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void exp(size_t n, etl::complex<double> alpha, etl::complex<double>* A, size_t lda, etl::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZEXP
    inc_counter("egblas");
    egblas_zexp(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::exp");
#endif
}

} //end of namespace etl::impl::egblas
