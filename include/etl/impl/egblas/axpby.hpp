//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the axpby operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAXPBY
static constexpr bool has_saxpby = true;
#else
static constexpr bool has_saxpby = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas axpby operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpby([[maybe_unused]] size_t n,
                  [[maybe_unused]] float alpha,
                  [[maybe_unused]] const float* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] float beta,
                  [[maybe_unused]] float* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SAXPBY
    inc_counter("egblas");
    egblas_saxpby(n, alpha, A, lda, beta, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpby");
#endif
}

#ifdef EGBLAS_HAS_DAXPBY
static constexpr bool has_daxpby = true;
#else
static constexpr bool has_daxpby = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas axpby operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpby([[maybe_unused]] size_t n,
                  [[maybe_unused]] double alpha,
                  [[maybe_unused]] const double* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] double beta,
                  [[maybe_unused]] double* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DAXPBY
    inc_counter("egblas");
    egblas_daxpby(n, alpha, A, lda, beta, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpby");
#endif
}

#ifdef EGBLAS_HAS_CAXPBY
static constexpr bool has_caxpby = true;
#else
static constexpr bool has_caxpby = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas axpby operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpby([[maybe_unused]] size_t n,
                  [[maybe_unused]] std::complex<float> alpha,
                  [[maybe_unused]] const std::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<float> beta,
                  [[maybe_unused]] std::complex<float>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CAXPBY
    inc_counter("egblas");
    egblas_caxpby(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpby");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas axpby operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpby([[maybe_unused]] size_t n,
                  [[maybe_unused]] etl::complex<float> alpha,
                  [[maybe_unused]] const etl::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<float> beta,
                  [[maybe_unused]] etl::complex<float>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CAXPBY
    inc_counter("egblas");
    egblas_caxpby(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpby");
#endif
}

#ifdef EGBLAS_HAS_ZAXPBY
static constexpr bool has_zaxpby = true;
#else
static constexpr bool has_zaxpby = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas axpby operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpby([[maybe_unused]] size_t n,
                  [[maybe_unused]] std::complex<double> alpha,
                  [[maybe_unused]] const std::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<double> beta,
                  [[maybe_unused]] std::complex<double>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZAXPBY
    inc_counter("egblas");
    egblas_zaxpby(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpby");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas axpby operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axpby([[maybe_unused]] size_t n,
                  [[maybe_unused]] etl::complex<double> alpha,
                  [[maybe_unused]] const etl::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<double> beta,
                  [[maybe_unused]] etl::complex<double>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZAXPBY
    inc_counter("egblas");
    egblas_zaxpby(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axpby");
#endif
}

} //end of namespace etl::impl::egblas
