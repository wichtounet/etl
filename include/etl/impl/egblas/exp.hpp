//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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
inline void exp([[maybe_unused]] size_t n,
                [[maybe_unused]] float alpha,
                [[maybe_unused]] float* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] float* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SEXP
    inc_counter("egblas");
    egblas_sexp(n, alpha, A, lda, B, ldb);
#else
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
inline void exp([[maybe_unused]] size_t n,
                [[maybe_unused]] double alpha,
                [[maybe_unused]] double* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] double* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DEXP
    inc_counter("egblas");
    egblas_dexp(n, alpha, A, lda, B, ldb);
#else
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
inline void exp([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<float> alpha,
                [[maybe_unused]] std::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CEXP
    inc_counter("egblas");
    egblas_cexp(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
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
inline void exp([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<float> alpha,
                [[maybe_unused]] etl::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CEXP
    inc_counter("egblas");
    egblas_cexp(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
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
inline void exp([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<double> alpha,
                [[maybe_unused]] std::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZEXP
    inc_counter("egblas");
    egblas_zexp(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
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
inline void exp([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<double> alpha,
                [[maybe_unused]] etl::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZEXP
    inc_counter("egblas");
    egblas_zexp(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::exp");
#endif
}

} //end of namespace etl::impl::egblas
