//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the log operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision logarithm.
 */
#ifdef EGBLAS_HAS_SLOG
static constexpr bool has_slog = true;
#else
static constexpr bool has_slog = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log([[maybe_unused]] size_t n,
                [[maybe_unused]] float alpha,
                [[maybe_unused]] float* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] float* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SLOG
    inc_counter("egblas");
    egblas_slog(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::log");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision logarithm.
 */
#ifdef EGBLAS_HAS_DLOG
static constexpr bool has_dlog = true;
#else
static constexpr bool has_dlog = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log([[maybe_unused]] size_t n,
                [[maybe_unused]] double alpha,
                [[maybe_unused]] double* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] double* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DLOG
    inc_counter("egblas");
    egblas_dlog(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::log");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision logarithm.
 */
#ifdef EGBLAS_HAS_CLOG
static constexpr bool has_clog = true;
#else
static constexpr bool has_clog = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<float> alpha,
                [[maybe_unused]] std::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CLOG
    inc_counter("egblas");
    egblas_clog(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::log");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<float> alpha,
                [[maybe_unused]] etl::complex<float>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<float>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CLOG
    inc_counter("egblas");
    egblas_clog(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::log");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision logarithm.
 */
#ifdef EGBLAS_HAS_ZLOG
static constexpr bool has_zlog = true;
#else
static constexpr bool has_zlog = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log([[maybe_unused]] size_t n,
                [[maybe_unused]] std::complex<double> alpha,
                [[maybe_unused]] std::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] std::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZLOG
    inc_counter("egblas");
    egblas_zlog(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::log");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas log operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log([[maybe_unused]] size_t n,
                [[maybe_unused]] etl::complex<double> alpha,
                [[maybe_unused]] etl::complex<double>* A,
                [[maybe_unused]] size_t lda,
                [[maybe_unused]] etl::complex<double>* B,
                [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZLOG
    inc_counter("egblas");
    egblas_zlog(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::log");
#endif
}

} //end of namespace etl::impl::egblas
