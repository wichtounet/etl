//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the softplus operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision softplus.
 */
#ifdef EGBLAS_HAS_SSOFTPLUS
static constexpr bool has_ssoftplus = true;
#else
static constexpr bool has_ssoftplus = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas softplus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void softplus([[maybe_unused]] size_t n,
                     [[maybe_unused]] float alpha,
                     [[maybe_unused]] float* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] float* B,
                     [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SSOFTPLUS
    inc_counter("egblas");
    egblas_ssoftplus(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::softplus");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision softplus.
 */
#ifdef EGBLAS_HAS_DSOFTPLUS
static constexpr bool has_dsoftplus = true;
#else
static constexpr bool has_dsoftplus = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas softplus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void softplus([[maybe_unused]] size_t n,
                     [[maybe_unused]] double alpha,
                     [[maybe_unused]] double* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] double* B,
                     [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DSOFTPLUS
    inc_counter("egblas");
    egblas_dsoftplus(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::softplus");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision softplus.
 */
#ifdef EGBLAS_HAS_CSOFTPLUS
static constexpr bool has_csoftplus = true;
#else
static constexpr bool has_csoftplus = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas softplus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void softplus([[maybe_unused]] size_t n,
                     [[maybe_unused]] std::complex<float> alpha,
                     [[maybe_unused]] std::complex<float>* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] std::complex<float>* B,
                     [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSOFTPLUS
    inc_counter("egblas");
    egblas_csoftplus(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::softplus");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas softplus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void softplus([[maybe_unused]] size_t n,
                     [[maybe_unused]] etl::complex<float> alpha,
                     [[maybe_unused]] etl::complex<float>* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] etl::complex<float>* B,
                     [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSOFTPLUS
    inc_counter("egblas");
    egblas_csoftplus(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::softplus");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision softplus.
 */
#ifdef EGBLAS_HAS_ZSOFTPLUS
static constexpr bool has_zsoftplus = true;
#else
static constexpr bool has_zsoftplus = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas softplus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void softplus([[maybe_unused]] size_t n,
                     [[maybe_unused]] std::complex<double> alpha,
                     [[maybe_unused]] std::complex<double>* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] std::complex<double>* B,
                     [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSOFTPLUS
    inc_counter("egblas");
    egblas_zsoftplus(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::softplus");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas softplus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void softplus([[maybe_unused]] size_t n,
                     [[maybe_unused]] etl::complex<double> alpha,
                     [[maybe_unused]] etl::complex<double>* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] etl::complex<double>* B,
                     [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSOFTPLUS
    inc_counter("egblas");
    egblas_zsoftplus(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::softplus");
#endif
}

} //end of namespace etl::impl::egblas
