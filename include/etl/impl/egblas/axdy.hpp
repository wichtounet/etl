//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the axdy operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAXDY
static constexpr bool has_saxdy = true;
#else
static constexpr bool has_saxdy = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas axdy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy([[maybe_unused]] size_t n, [[maybe_unused]] float alpha, [[maybe_unused]] float* A, [[maybe_unused]] size_t lda, [[maybe_unused]] float* B, [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SAXDY
    inc_counter("egblas");
    egblas_saxdy(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdy");
#endif
}

#ifdef EGBLAS_HAS_DAXDY
static constexpr bool has_daxdy = true;
#else
static constexpr bool has_daxdy = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas axdy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy([[maybe_unused]] size_t n, [[maybe_unused]] double alpha, [[maybe_unused]] double* A, [[maybe_unused]] size_t lda, [[maybe_unused]] double* B, [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DAXDY
    inc_counter("egblas");
    egblas_daxdy(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdy");
#endif
}

#ifdef EGBLAS_HAS_CAXDY
static constexpr bool has_caxdy = true;
#else
static constexpr bool has_caxdy = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas axdy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy([[maybe_unused]] size_t n, [[maybe_unused]] std::complex<float> alpha, [[maybe_unused]] std::complex<float>* A, [[maybe_unused]] size_t lda, [[maybe_unused]] std::complex<float>* B, [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CAXDY
    inc_counter("egblas");
    egblas_caxdy(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdy");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas axdy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy([[maybe_unused]] size_t n, [[maybe_unused]] etl::complex<float> alpha, [[maybe_unused]] etl::complex<float>* A, [[maybe_unused]] size_t lda, [[maybe_unused]] etl::complex<float>* B, [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CAXDY
    inc_counter("egblas");
    egblas_caxdy(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdy");
#endif
}

#ifdef EGBLAS_HAS_ZAXDY
static constexpr bool has_zaxdy = true;
#else
static constexpr bool has_zaxdy = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas axdy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy([[maybe_unused]] size_t n, [[maybe_unused]] std::complex<double> alpha, [[maybe_unused]] std::complex<double>* A, [[maybe_unused]] size_t lda, [[maybe_unused]] std::complex<double>* B, [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZAXDY
    inc_counter("egblas");
    egblas_zaxdy(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdy");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas axdy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdy([[maybe_unused]] size_t n, [[maybe_unused]] etl::complex<double> alpha, [[maybe_unused]] etl::complex<double>* A, [[maybe_unused]] size_t lda, [[maybe_unused]] etl::complex<double>* B, [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZAXDY
    inc_counter("egblas");
    egblas_zaxdy(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdy");
#endif
}

} //end of namespace etl::impl::egblas
