//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the sinh operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision hyperbolic sinus.
 */
#ifdef EGBLAS_HAS_SSINH
static constexpr bool has_ssinh = true;
#else
static constexpr bool has_ssinh = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas sinh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sinh([[maybe_unused]] size_t n,
                 [[maybe_unused]] float alpha,
                 [[maybe_unused]] float* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] float* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SSINH
    inc_counter("egblas");
    egblas_ssinh(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sinh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision hyperbolic sinus.
 */
#ifdef EGBLAS_HAS_DSINH
static constexpr bool has_dsinh = true;
#else
static constexpr bool has_dsinh = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas sinh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sinh([[maybe_unused]] size_t n,
                 [[maybe_unused]] double alpha,
                 [[maybe_unused]] double* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] double* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DSINH
    inc_counter("egblas");
    egblas_dsinh(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sinh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision hyperbolic sinus.
 */
#ifdef EGBLAS_HAS_CSINH
static constexpr bool has_csinh = true;
#else
static constexpr bool has_csinh = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas sinh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sinh([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<float> alpha,
                 [[maybe_unused]] std::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSINH
    inc_counter("egblas");
    egblas_csinh(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sinh");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas sinh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sinh([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<float> alpha,
                 [[maybe_unused]] etl::complex<float>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<float>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CSINH
    inc_counter("egblas");
    egblas_csinh(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sinh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision hyperbolic sinus.
 */
#ifdef EGBLAS_HAS_ZSINH
static constexpr bool has_zsinh = true;
#else
static constexpr bool has_zsinh = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas sinh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sinh([[maybe_unused]] size_t n,
                 [[maybe_unused]] std::complex<double> alpha,
                 [[maybe_unused]] std::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] std::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSINH
    inc_counter("egblas");
    egblas_zsinh(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sinh");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas sinh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sinh([[maybe_unused]] size_t n,
                 [[maybe_unused]] etl::complex<double> alpha,
                 [[maybe_unused]] etl::complex<double>* A,
                 [[maybe_unused]] size_t lda,
                 [[maybe_unused]] etl::complex<double>* B,
                 [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZSINH
    inc_counter("egblas");
    egblas_zsinh(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::sinh");
#endif
}

} //end of namespace etl::impl::egblas
