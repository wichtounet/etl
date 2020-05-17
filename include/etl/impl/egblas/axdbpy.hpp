//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

#ifdef EGBLAS_HAS_SAXDBPY
static constexpr bool has_saxdbpy = true;
#else
static constexpr bool has_saxdbpy = false;
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
inline void axdbpy([[maybe_unused]] size_t n,
                   [[maybe_unused]] float alpha,
                   [[maybe_unused]] float* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] float beta,
                   [[maybe_unused]] float* B,
                   [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SAXDBPY
    inc_counter("egblas");
    egblas_saxdbpy(n, alpha, A, lda, beta, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy");
#endif
}

#ifdef EGBLAS_HAS_DAXDBPY
static constexpr bool has_daxdbpy = true;
#else
static constexpr bool has_daxdbpy = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas axdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy([[maybe_unused]] size_t n,
                   [[maybe_unused]] double alpha,
                   [[maybe_unused]] double* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] double beta,
                   [[maybe_unused]] double* B,
                   [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DAXDBPY
    inc_counter("egblas");
    egblas_daxdbpy(n, alpha, A, lda, beta, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy");
#endif
}

#ifdef EGBLAS_HAS_CAXDBPY
static constexpr bool has_caxdbpy = true;
#else
static constexpr bool has_caxdbpy = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas axdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy([[maybe_unused]] size_t n,
                   [[maybe_unused]] std::complex<float> alpha,
                   [[maybe_unused]] std::complex<float>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] std::complex<float> beta,
                   [[maybe_unused]] std::complex<float>* B,
                   [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CAXDBPY
    inc_counter("egblas");
    egblas_caxdbpy(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas axdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy([[maybe_unused]] size_t n,
                   [[maybe_unused]] etl::complex<float> alpha,
                   [[maybe_unused]] etl::complex<float>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] std::complex<float> beta,
                   [[maybe_unused]] etl::complex<float>* B,
                   [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CAXDBPY
    inc_counter("egblas");
    egblas_caxdbpy(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy");
#endif
}

#ifdef EGBLAS_HAS_ZAXDBPY
static constexpr bool has_zaxdbpy = true;
#else
static constexpr bool has_zaxdbpy = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas axdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy([[maybe_unused]] size_t n,
                   [[maybe_unused]] std::complex<double> alpha,
                   [[maybe_unused]] std::complex<double>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] std::complex<double> beta,
                   [[maybe_unused]] std::complex<double>* B,
                   [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZAXDBPY
    inc_counter("egblas");
    egblas_zaxdbpy(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas axdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void axdbpy([[maybe_unused]] size_t n,
                   [[maybe_unused]] etl::complex<double> alpha,
                   [[maybe_unused]] etl::complex<double>* A,
                   [[maybe_unused]] size_t lda,
                   [[maybe_unused]] std::complex<double> beta,
                   [[maybe_unused]] etl::complex<double>* B,
                   [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZAXDBPY
    inc_counter("egblas");
    egblas_zaxdbpy(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::axdbpy");
#endif
}

} //end of namespace etl::impl::egblas
