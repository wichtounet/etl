//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the invsqrt operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision inverse square root.
 */
#ifdef EGBLAS_HAS_SINVSQRT
static constexpr bool has_sinvsqrt = true;
#else
static constexpr bool has_sinvsqrt = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt([[maybe_unused]] size_t n,
                    [[maybe_unused]] float alpha,
                    [[maybe_unused]] float* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] float* B,
                    [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SINVSQRT
    inc_counter("egblas");
    egblas_sinvsqrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision inverse square root.
 */
#ifdef EGBLAS_HAS_DINVSQRT
static constexpr bool has_dinvsqrt = true;
#else
static constexpr bool has_dinvsqrt = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt([[maybe_unused]] size_t n,
                    [[maybe_unused]] double alpha,
                    [[maybe_unused]] double* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] double* B,
                    [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DINVSQRT
    inc_counter("egblas");
    egblas_dinvsqrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision inverse square root.
 */
#ifdef EGBLAS_HAS_CINVSQRT
static constexpr bool has_cinvsqrt = true;
#else
static constexpr bool has_cinvsqrt = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt([[maybe_unused]] size_t n,
                    [[maybe_unused]] std::complex<float> alpha,
                    [[maybe_unused]] std::complex<float>* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] std::complex<float>* B,
                    [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CINVSQRT
    inc_counter("egblas");
    egblas_cinvsqrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt([[maybe_unused]] size_t n,
                    [[maybe_unused]] etl::complex<float> alpha,
                    [[maybe_unused]] etl::complex<float>* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] etl::complex<float>* B,
                    [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CINVSQRT
    inc_counter("egblas");
    egblas_cinvsqrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision inverse square root.
 */
#ifdef EGBLAS_HAS_ZINVSQRT
static constexpr bool has_zinvsqrt = true;
#else
static constexpr bool has_zinvsqrt = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt([[maybe_unused]] size_t n,
                    [[maybe_unused]] std::complex<double> alpha,
                    [[maybe_unused]] std::complex<double>* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] std::complex<double>* B,
                    [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZINVSQRT
    inc_counter("egblas");
    egblas_zinvsqrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt([[maybe_unused]] size_t n,
                    [[maybe_unused]] etl::complex<double> alpha,
                    [[maybe_unused]] etl::complex<double>* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] etl::complex<double>* B,
                    [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZINVSQRT
    inc_counter("egblas");
    egblas_zinvsqrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

} //end of namespace etl::impl::egblas
