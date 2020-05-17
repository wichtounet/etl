//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
#ifdef EGBLAS_HAS_SLOG10
static constexpr bool has_slog10 = true;
#else
static constexpr bool has_slog10 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas log10 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log10([[maybe_unused]] size_t n,
                  [[maybe_unused]] float alpha,
                  [[maybe_unused]] float* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] float* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SLOG10
    inc_counter("egblas");
    egblas_slog10(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::log10");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision logarithm.
 */
#ifdef EGBLAS_HAS_DLOG10
static constexpr bool has_dlog10 = true;
#else
static constexpr bool has_dlog10 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas log10 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log10([[maybe_unused]] size_t n,
                  [[maybe_unused]] double alpha,
                  [[maybe_unused]] double* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] double* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DLOG10
    inc_counter("egblas");
    egblas_dlog10(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::log10");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision logarithm.
 */
#ifdef EGBLAS_HAS_CLOG10
static constexpr bool has_clog10 = true;
#else
static constexpr bool has_clog10 = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas log10 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log10([[maybe_unused]] size_t n,
                  [[maybe_unused]] std::complex<float> alpha,
                  [[maybe_unused]] std::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<float>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CLOG10
    inc_counter("egblas");
    egblas_clog10(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::log10");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas log10 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log10([[maybe_unused]] size_t n,
                  [[maybe_unused]] etl::complex<float> alpha,
                  [[maybe_unused]] etl::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] etl::complex<float>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CLOG10
    inc_counter("egblas");
    egblas_clog10(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::log10");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision logarithm.
 */
#ifdef EGBLAS_HAS_ZLOG10
static constexpr bool has_zlog10 = true;
#else
static constexpr bool has_zlog10 = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas log10 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log10([[maybe_unused]] size_t n,
                  [[maybe_unused]] std::complex<double> alpha,
                  [[maybe_unused]] std::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<double>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZLOG10
    inc_counter("egblas");
    egblas_zlog10(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::log10");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas log10 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void log10([[maybe_unused]] size_t n,
                  [[maybe_unused]] etl::complex<double> alpha,
                  [[maybe_unused]] etl::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] etl::complex<double>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZLOG10
    inc_counter("egblas");
    egblas_zlog10(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::log10");
#endif
}

} //end of namespace etl::impl::egblas
