//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the cos operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision cosine.
 */
#ifdef EGBLAS_HAS_SCOS
static constexpr bool has_scos = true;
#else
static constexpr bool has_scos = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas cos operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cos([[maybe_unused]] size_t n,
                [[maybe_unused]] float alpha,
                [[maybe_unused]] float* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] float* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SCOS
    inc_counter("egblas");
    egblas_scos(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::cos");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision cosine.
 */
#ifdef EGBLAS_HAS_DCOS
static constexpr bool has_dcos = true;
#else
static constexpr bool has_dcos = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas cos operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cos([[maybe_unused]] size_t n,
                [[maybe_unused]] double alpha,
                [[maybe_unused]] double* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] double* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DCOS
    inc_counter("egblas");
    egblas_dcos(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::cos");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision cosine.
 */
#ifdef EGBLAS_HAS_CCOS
static constexpr bool has_ccos = true;
#else
static constexpr bool has_ccos = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas cos operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cos([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<float> alpha,
                [[maybe_unused]] std::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CCOS
    inc_counter("egblas");
    egblas_ccos(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::cos");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas cos operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cos([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<float> alpha,
                [[maybe_unused]] etl::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CCOS
    inc_counter("egblas");
    egblas_ccos(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::cos");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision cosine.
 */
#ifdef EGBLAS_HAS_ZCOS
static constexpr bool has_zcos = true;
#else
static constexpr bool has_zcos = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas cos operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cos([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<double> alpha,
                [[maybe_unused]] std::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZCOS
    inc_counter("egblas");
    egblas_zcos(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::cos");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas cos operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cos([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<double> alpha,
                [[maybe_unused]] etl::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZCOS
    inc_counter("egblas");
    egblas_zcos(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::cos");
#endif
}

} //end of namespace etl::impl::egblas
