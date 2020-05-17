//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the less_equal operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision less_equal.
 */
#ifdef EGBLAS_HAS_SLESS_EQUAL
static constexpr bool has_sless_equal = true;
#else
static constexpr bool has_sless_equal = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas less_equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less_equal([[maybe_unused]] size_t n,
                       [[maybe_unused]] const float* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const float* B,
                       [[maybe_unused]] size_t ldb,
                       [[maybe_unused]] bool* C,
                       [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_SLESS_EQUAL
    inc_counter("egblas");
    egblas_sless_equal(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less_equal");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision less_equal.
 */
#ifdef EGBLAS_HAS_DLESS_EQUAL
static constexpr bool has_dless_equal = true;
#else
static constexpr bool has_dless_equal = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas less_equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less_equal([[maybe_unused]] size_t n,
                       [[maybe_unused]] const double* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const double* B,
                       [[maybe_unused]] size_t ldb,
                       [[maybe_unused]] bool* C,
                       [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_DLESS_EQUAL
    inc_counter("egblas");
    egblas_dless_equal(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less_equal");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision less_equal.
 */
#ifdef EGBLAS_HAS_CLESS_EQUAL
static constexpr bool has_cless_equal = true;
#else
static constexpr bool has_cless_equal = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas less_equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less_equal([[maybe_unused]] size_t n,
                       [[maybe_unused]] const std::complex<float>* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const std::complex<float>* B,
                       [[maybe_unused]] size_t ldb,
                       [[maybe_unused]] bool* C,
                       [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CLESS_EQUAL
    inc_counter("egblas");
    egblas_cless_equal(n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less_equal");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas less_equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less_equal([[maybe_unused]] size_t n,
                       [[maybe_unused]] const etl::complex<float>* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const etl::complex<float>* B,
                       [[maybe_unused]] size_t ldb,
                       [[maybe_unused]] bool* C,
                       [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_CLESS_EQUAL
    inc_counter("egblas");
    egblas_cless_equal(n, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less_equal");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision less_equal.
 */
#ifdef EGBLAS_HAS_ZLESS_EQUAL
static constexpr bool has_zless_equal = true;
#else
static constexpr bool has_zless_equal = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas less_equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less_equal([[maybe_unused]] size_t n,
                       [[maybe_unused]] const std::complex<double>* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const std::complex<double>* B,
                       [[maybe_unused]] size_t ldb,
                       [[maybe_unused]] bool* C,
                       [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZLESS_EQUAL
    inc_counter("egblas");
    egblas_zless_equal(n, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less_equal");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas less_equal operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void less_equal([[maybe_unused]] size_t n,
                       [[maybe_unused]] const etl::complex<double>* A,
                       [[maybe_unused]] size_t lda,
                       [[maybe_unused]] const etl::complex<double>* B,
                       [[maybe_unused]] size_t ldb,
                       [[maybe_unused]] bool* C,
                       [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_ZLESS_EQUAL
    inc_counter("egblas");
    egblas_zless_equal(n, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::less_equal");
#endif
}

} //end of namespace etl::impl::egblas
