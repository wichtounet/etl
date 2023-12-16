//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the axdbpy operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAXDBPY_3
static constexpr bool has_saxdbpy_3 = true;
#else
static constexpr bool has_saxdbpy_3 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas axdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy_3([[maybe_unused]] size_t n,
                     [[maybe_unused]] float alpha,
                     [[maybe_unused]] float* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] float beta,
                     [[maybe_unused]] float* B,
                     [[maybe_unused]] size_t ldb,
                     [[maybe_unused]] float* C,
                     [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SAXDBPY_3
    inc_counter("egblas");
    egblas_saxdbpy_3(n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy_3");
#endif
}

#ifdef EGBLAS_HAS_DAXDBPY_3
static constexpr bool has_daxdbpy_3 = true;
#else
static constexpr bool has_daxdbpy_3 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas axdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy_3([[maybe_unused]] size_t n,
                     [[maybe_unused]] double alpha,
                     [[maybe_unused]] double* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] double beta,
                     [[maybe_unused]] double* B,
                     [[maybe_unused]] size_t ldb,
                     [[maybe_unused]] double* C,
                     [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DAXDBPY_3
    inc_counter("egblas");
    egblas_daxdbpy_3(n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy_3");
#endif
}

#ifdef EGBLAS_HAS_CAXDBPY_3
static constexpr bool has_caxdbpy_3 = true;
#else
static constexpr bool has_caxdbpy_3 = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas axdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy_3([[maybe_unused]] size_t n,
                     [[maybe_unused]] std::complex<float> alpha,
                     [[maybe_unused]] std::complex<float>* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] std::complex<float> beta,
                     [[maybe_unused]] std::complex<float>* B,
                     [[maybe_unused]] size_t ldb,
                     [[maybe_unused]] std::complex<float>* C,
                     [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CAXDBPY_3
    inc_counter("egblas");
    egblas_caxdbpy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb,
                     reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy_3");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas axdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy_3([[maybe_unused]] size_t n,
                     [[maybe_unused]] etl::complex<float> alpha,
                     [[maybe_unused]] etl::complex<float>* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] etl::complex<float> beta,
                     [[maybe_unused]] etl::complex<float>* B,
                     [[maybe_unused]] size_t ldb,
                     [[maybe_unused]] etl::complex<float>* C,
                     [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CAXDBPY_3
    inc_counter("egblas");
    egblas_caxdbpy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb,
                     reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy");
#endif
}

#ifdef EGBLAS_HAS_ZAXDBPY_3
static constexpr bool has_zaxdbpy_3 = true;
#else
static constexpr bool has_zaxdbpy_3 = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas axdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy_3([[maybe_unused]] size_t n,
                     [[maybe_unused]] std::complex<double> alpha,
                     [[maybe_unused]] std::complex<double>* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] std::complex<double> beta,
                     [[maybe_unused]] std::complex<double>* B,
                     [[maybe_unused]] size_t ldb,
                     [[maybe_unused]] std::complex<double>* C,
                     [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZAXDBPY_3
    inc_counter("egblas");
    egblas_zaxdbpy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb,
                     reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy_3");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas axdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy_3([[maybe_unused]] size_t n,
                     [[maybe_unused]] etl::complex<double> alpha,
                     [[maybe_unused]] etl::complex<double>* A,
                     [[maybe_unused]] size_t lda,
                     [[maybe_unused]] etl::complex<double> beta,
                     [[maybe_unused]] etl::complex<double>* B,
                     [[maybe_unused]] size_t ldb,
                     [[maybe_unused]] etl::complex<double>* C,
                     [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZAXDBPY_3
    inc_counter("egblas");
    egblas_zaxdbpy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb,
                     reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy_3");
#endif
}

} //end of namespace etl::impl::egblas
