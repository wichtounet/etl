//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the ceil operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision ceil.
 */
#ifdef EGBLAS_HAS_SCEIL
static constexpr bool has_sceil = true;
#else
static constexpr bool has_sceil = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas ceil operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void ceil([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] float* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SCEIL
    inc_counter("egblas");
    egblas_sceil(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::ceil");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision ceil.
 */
#ifdef EGBLAS_HAS_DCEIL
static constexpr bool has_dceil = true;
#else
static constexpr bool has_dceil = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas ceil operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void ceil([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] double* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DCEIL
    inc_counter("egblas");
    egblas_dceil(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::ceil");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision ceil.
 */
#ifdef EGBLAS_HAS_CCEIL
static constexpr bool has_cceil = true;
#else
static constexpr bool has_cceil = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas ceil operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void ceil([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<float> alpha,
                 [[maybe_unused]] std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CCEIL
    inc_counter("egblas");
    egblas_cceil(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::ceil");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas ceil operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void ceil([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<float> alpha,
                 [[maybe_unused]] etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CCEIL
    inc_counter("egblas");
    egblas_cceil(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::ceil");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision ceil.
 */
#ifdef EGBLAS_HAS_ZCEIL
static constexpr bool has_zceil = true;
#else
static constexpr bool has_zceil = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas ceil operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void ceil([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<double> alpha,
                 [[maybe_unused]] std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZCEIL
    inc_counter("egblas");
    egblas_zceil(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::ceil");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas ceil operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void ceil([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<double> alpha,
                 [[maybe_unused]] etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZCEIL
    inc_counter("egblas");
    egblas_zceil(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::ceil");
#endif
}

} //end of namespace etl::impl::egblas
