//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the greater operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision greater.
 */
#ifdef EGBLAS_HAS_SGREATER
static constexpr bool has_sgreater = true;
#else
static constexpr bool has_sgreater = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas greater operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void greater([[maybe_unused]] size_t n,
                    [[maybe_unused]] const float* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] const float* B,
                    [[maybe_unused]] size_t ldb,
                    [[maybe_unused]] bool* C,
                    [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SGREATER
    inc_counter("egblas");
    egblas_sgreater(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::greater");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision greater.
 */
#ifdef EGBLAS_HAS_DGREATER
static constexpr bool has_dgreater = true;
#else
static constexpr bool has_dgreater = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas greater operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void greater([[maybe_unused]] size_t n,
                    [[maybe_unused]] const double* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] const double* B,
                    [[maybe_unused]] size_t ldb,
                    [[maybe_unused]] bool* C,
                    [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DGREATER
    inc_counter("egblas");
    egblas_dgreater(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::greater");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision greater.
 */
#ifdef EGBLAS_HAS_CGREATER
static constexpr bool has_cgreater = true;
#else
static constexpr bool has_cgreater = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas greater operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void greater([[maybe_unused]] size_t n,
                    [[maybe_unused]] const std::complex<float>* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] const std::complex<float>* B,
                    [[maybe_unused]] size_t ldb,
                    [[maybe_unused]] bool* C,
                    [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CGREATER
    inc_counter("egblas");
    egblas_cgreater(n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::greater");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas greater operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void greater([[maybe_unused]] size_t n,
                    [[maybe_unused]] const etl::complex<float>* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] const etl::complex<float>* B,
                    [[maybe_unused]] size_t ldb,
                    [[maybe_unused]] bool* C,
                    [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CGREATER
    inc_counter("egblas");
    egblas_cgreater(n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::greater");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision greater.
 */
#ifdef EGBLAS_HAS_ZGREATER
static constexpr bool has_zgreater = true;
#else
static constexpr bool has_zgreater = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas greater operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void greater([[maybe_unused]] size_t n,
                    [[maybe_unused]] const std::complex<double>* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] const std::complex<double>* B,
                    [[maybe_unused]] size_t ldb,
                    [[maybe_unused]] bool* C,
                    [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZGREATER
    inc_counter("egblas");
    egblas_zgreater(n, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::greater");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas greater operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void greater([[maybe_unused]] size_t n,
                    [[maybe_unused]] const etl::complex<double>* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] const etl::complex<double>* B,
                    [[maybe_unused]] size_t ldb,
                    [[maybe_unused]] bool* C,
                    [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZGREATER
    inc_counter("egblas");
    egblas_zgreater(n, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::greater");
#endif
}

} //end of namespace etl::impl::egblas
