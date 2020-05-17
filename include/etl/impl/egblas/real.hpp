//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the real operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has complex single-precision real.
 */
#ifdef EGBLAS_HAS_CREAL
static constexpr bool has_creal = true;
#else
static constexpr bool has_creal = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas real operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void real([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CREAL
    inc_counter("egblas");
    egblas_creal(n, alpha, reinterpret_cast<cuComplex*>(A), lda, (B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::real");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas real operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void real([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CREAL
    inc_counter("egblas");
    egblas_creal(n, alpha, reinterpret_cast<cuComplex*>(A), lda, (B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::real");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision real.
 */
#ifdef EGBLAS_HAS_ZREAL
static constexpr bool has_zreal = true;
#else
static constexpr bool has_zreal = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas real operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void real([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZREAL
    inc_counter("egblas");
    egblas_zreal(n, alpha, reinterpret_cast<cuDoubleComplex*>(A), lda, (B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::real");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas real operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void real([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZREAL
    inc_counter("egblas");
    egblas_zreal(n, alpha, reinterpret_cast<cuDoubleComplex*>(A), lda, (B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::real");
#endif
}

} //end of namespace etl::impl::egblas
