//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the minus operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision minusarithm.
 */
#ifdef EGBLAS_HAS_SMINUS
static constexpr bool has_sminus = true;
#else
static constexpr bool has_sminus = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas minus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void minus([[maybe_unused]] size_t n,
                  [[maybe_unused]] float alpha,
                  [[maybe_unused]] float* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] float* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SMINUS
    inc_counter("egblas");
    egblas_sminus(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::minus");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision minusarithm.
 */
#ifdef EGBLAS_HAS_DMINUS
static constexpr bool has_dminus = true;
#else
static constexpr bool has_dminus = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas minus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void minus([[maybe_unused]] size_t n,
                  [[maybe_unused]] double alpha,
                  [[maybe_unused]] double* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] double* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DMINUS
    inc_counter("egblas");
    egblas_dminus(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::minus");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision minusarithm.
 */
#ifdef EGBLAS_HAS_CMINUS
static constexpr bool has_cminus = true;
#else
static constexpr bool has_cminus = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas minus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void minus([[maybe_unused]] size_t n,
                  [[maybe_unused]] std::complex<float> alpha,
                  [[maybe_unused]] std::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<float>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CMINUS
    inc_counter("egblas");
    egblas_cminus(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::minus");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas minus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void minus([[maybe_unused]] size_t n,
                  [[maybe_unused]] etl::complex<float> alpha,
                  [[maybe_unused]] etl::complex<float>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] etl::complex<float>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_CMINUS
    inc_counter("egblas");
    egblas_cminus(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::minus");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision minusarithm.
 */
#ifdef EGBLAS_HAS_ZMINUS
static constexpr bool has_zminus = true;
#else
static constexpr bool has_zminus = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas minus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void minus([[maybe_unused]] size_t n,
                  [[maybe_unused]] std::complex<double> alpha,
                  [[maybe_unused]] std::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] std::complex<double>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZMINUS
    inc_counter("egblas");
    egblas_zminus(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::minus");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas minus operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void minus([[maybe_unused]] size_t n,
                  [[maybe_unused]] etl::complex<double> alpha,
                  [[maybe_unused]] etl::complex<double>* A,
                  [[maybe_unused]] size_t lda,
                  [[maybe_unused]] etl::complex<double>* B,
                  [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_ZMINUS
    inc_counter("egblas");
    egblas_zminus(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unreachable("Invalid call to egblas::minus");
#endif
}

} //end of namespace etl::impl::egblas
