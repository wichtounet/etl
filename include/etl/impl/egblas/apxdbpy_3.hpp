//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the apxdbpy operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAPXDBPY_3
static constexpr bool has_sapxdbpy_3 = true;
#else
static constexpr bool has_sapxdbpy_3 = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas apxdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy_3([[maybe_unused]] size_t n, [[maybe_unused]] float alpha, [[maybe_unused]] float* A, [[maybe_unused]] size_t lda, [[maybe_unused]] float beta, [[maybe_unused]] float* B, [[maybe_unused]] size_t ldb, [[maybe_unused]] float* C, [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SAPXDBPY_3
    inc_counter("egblas");
    egblas_sapxdbpy_3(n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::apxdbpy_3");
#endif
}

#ifdef EGBLAS_HAS_DAPXDBPY_3
static constexpr bool has_dapxdbpy_3 = true;
#else
static constexpr bool has_dapxdbpy_3 = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas apxdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy_3([[maybe_unused]] size_t n, [[maybe_unused]] double alpha, [[maybe_unused]] double* A, [[maybe_unused]] size_t lda, [[maybe_unused]] double beta, [[maybe_unused]] double* B, [[maybe_unused]] size_t ldb, [[maybe_unused]] double* C, [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DAPXDBPY_3
    inc_counter("egblas");
    egblas_dapxdbpy_3(n, alpha, A, lda, beta, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::apxdbpy_3");
#endif
}

#ifdef EGBLAS_HAS_CAPXDBPY_3
static constexpr bool has_capxdbpy_3 = true;
#else
static constexpr bool has_capxdbpy_3 = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas apxdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy_3([[maybe_unused]] size_t n,
                      [[maybe_unused]] std::complex<float> alpha,
                      [[maybe_unused]] std::complex<float>* A,
                      [[maybe_unused]] size_t lda,
                      [[maybe_unused]] std::complex<float> beta,
                      [[maybe_unused]] std::complex<float>* B,
                      [[maybe_unused]] size_t ldb,
                      [[maybe_unused]] std::complex<float>* C,
                      [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CAPXDBPY_3
    inc_counter("egblas");
    egblas_capxdbpy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb,
                      reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::apxdbpy_3");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas apxdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy_3([[maybe_unused]] size_t n,
                      [[maybe_unused]] etl::complex<float> alpha,
                      [[maybe_unused]] etl::complex<float>* A,
                      [[maybe_unused]] size_t lda,
                      [[maybe_unused]] etl::complex<float> beta,
                      [[maybe_unused]] etl::complex<float>* B,
                      [[maybe_unused]] size_t ldb,
                      [[maybe_unused]] etl::complex<float>* C,
                      [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CAPXDBPY_3
    inc_counter("egblas");
    egblas_capxdbpy_3(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb,
                      reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::apxdbpy");
#endif
}

#ifdef EGBLAS_HAS_ZAPXDBPY_3
static constexpr bool has_zapxdbpy_3 = true;
#else
static constexpr bool has_zapxdbpy_3 = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas apxdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy_3([[maybe_unused]] size_t n,
                      [[maybe_unused]] std::complex<double> alpha,
                      [[maybe_unused]] std::complex<double>* A,
                      [[maybe_unused]] size_t lda,
                      [[maybe_unused]] std::complex<double> beta,
                      [[maybe_unused]] std::complex<double>* B,
                      [[maybe_unused]] size_t ldb,
                      [[maybe_unused]] std::complex<double>* C,
                      [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZAPXDBPY_3
    inc_counter("egblas");
    egblas_zapxdbpy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb,
                      reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::apxdbpy_3");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas apxdbpy_3 operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy_3([[maybe_unused]] size_t n,
                      [[maybe_unused]] etl::complex<double> alpha,
                      [[maybe_unused]] etl::complex<double>* A,
                      [[maybe_unused]] size_t lda,
                      [[maybe_unused]] etl::complex<double> beta,
                      [[maybe_unused]] etl::complex<double>* B,
                      [[maybe_unused]] size_t ldb,
                      [[maybe_unused]] etl::complex<double>* C,
                      [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZAPXDBPY_3
    inc_counter("egblas");
    egblas_zapxdbpy_3(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb,
                      reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unreachable("Invalid call to egblas::apxdbpy_3");
#endif
}

} //end of namespace etl::impl::egblas
