//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the floor operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision floor.
 */
#ifdef EGBLAS_HAS_SFLOOR
static constexpr bool has_sfloor = true;
#else
static constexpr bool has_sfloor = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas floor operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void floor([[maybe_unused]] size_t n,
                  [[maybe_unused]] float alpha,
                  [[maybe_unused]] float* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] float* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SFLOOR
    inc_counter("egblas");
    egblas_sfloor(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::floor");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision floor.
 */
#ifdef EGBLAS_HAS_DFLOOR
static constexpr bool has_dfloor = true;
#else
static constexpr bool has_dfloor = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas floor operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void floor([[maybe_unused]] size_t n,
                  [[maybe_unused]] double alpha,
                  [[maybe_unused]] double* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] double* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DFLOOR
    inc_counter("egblas");
    egblas_dfloor(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::floor");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision floor.
 */
#ifdef EGBLAS_HAS_CFLOOR
static constexpr bool has_cfloor = true;
#else
static constexpr bool has_cfloor = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas floor operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void floor([[maybe_unused]] size_t n,
                  [[maybe_unused]] std::complex<float> alpha,
                  [[maybe_unused]] std::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<float>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CFLOOR
    inc_counter("egblas");
    egblas_cfloor(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::floor");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas floor operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void floor([[maybe_unused]] size_t n,
                  [[maybe_unused]] etl::complex<float> alpha,
                  [[maybe_unused]] etl::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] etl::complex<float>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CFLOOR
    inc_counter("egblas");
    egblas_cfloor(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::floor");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision floor.
 */
#ifdef EGBLAS_HAS_ZFLOOR
static constexpr bool has_zfloor = true;
#else
static constexpr bool has_zfloor = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas floor operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void floor([[maybe_unused]] size_t n,
                  [[maybe_unused]] std::complex<double> alpha,
                  [[maybe_unused]] std::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<double>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZFLOOR
    inc_counter("egblas");
    egblas_zfloor(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::floor");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas floor operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void floor([[maybe_unused]] size_t n,
                  [[maybe_unused]] etl::complex<double> alpha,
                  [[maybe_unused]] etl::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] etl::complex<double>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZFLOOR
    inc_counter("egblas");
    egblas_zfloor(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::floor");
#endif
}

} //end of namespace etl::impl::egblas
