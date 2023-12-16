//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the imag operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has complex single-precision imag.
 */
#ifdef EGBLAS_HAS_Cimag
static constexpr bool has_cimag = true;
#else
static constexpr bool has_cimag = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas imag operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void imag([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_Cimag
    inc_counter("egblas");
    egblas_cimag(n, alpha, reinterpret_cast<cuComplex*>(A), lda, (B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::imag");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas imag operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void imag([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_Cimag
    inc_counter("egblas");
    egblas_cimag(n, alpha, reinterpret_cast<cuComplex*>(A), lda, (B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::imag");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision imag.
 */
#ifdef EGBLAS_HAS_Zimag
static constexpr bool has_zimag = true;
#else
static constexpr bool has_zimag = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas imag operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void imag([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_Zimag
    inc_counter("egblas");
    egblas_zimag(n, alpha, reinterpret_cast<cuDoubleComplex*>(A), lda, (B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::imag");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas imag operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void imag([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_Zimag
    inc_counter("egblas");
    egblas_zimag(n, alpha, reinterpret_cast<cuDoubleComplex*>(A), lda, (B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::imag");
#endif
}

} //end of namespace etl::impl::egblas
