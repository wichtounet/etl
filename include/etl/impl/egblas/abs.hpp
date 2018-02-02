//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the abs operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision absarithm.
 */
#ifdef EGBLAS_HAS_SABS
static constexpr bool has_sabs = true;
#else
static constexpr bool has_sabs = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas abs operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void abs(size_t n, float alpha, float* A , size_t lda, float* B , size_t ldb){
#ifdef EGBLAS_HAS_SABS
    inc_counter("egblas");
    egblas_sabs(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::abs");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision absarithm.
 */
#ifdef EGBLAS_HAS_DABS
static constexpr bool has_dabs = true;
#else
static constexpr bool has_dabs = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas abs operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void abs(size_t n, double alpha, double* A , size_t lda, double* B , size_t ldb){
#ifdef EGBLAS_HAS_DABS
    inc_counter("egblas");
    egblas_dabs(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::abs");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision absarithm.
 */
#ifdef EGBLAS_HAS_CABS
static constexpr bool has_cabs = true;
#else
static constexpr bool has_cabs = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas abs operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void abs(size_t n, float alpha, std::complex<float>* A , size_t lda, float* B , size_t ldb){
#ifdef EGBLAS_HAS_CABS
    inc_counter("egblas");
    egblas_cabs(n, alpha, reinterpret_cast<cuComplex*>(A), lda, (B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::abs");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas abs operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void abs(size_t n, float alpha, etl::complex<float>* A , size_t lda, float* B , size_t ldb){
#ifdef EGBLAS_HAS_CABS
    inc_counter("egblas");
    egblas_cabs(n, alpha, reinterpret_cast<cuComplex*>(A), lda, (B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::abs");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision absarithm.
 */
#ifdef EGBLAS_HAS_ZABS
static constexpr bool has_zabs = true;
#else
static constexpr bool has_zabs = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas abs operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void abs(size_t n, double alpha, std::complex<double>* A , size_t lda, double* B , size_t ldb){
#ifdef EGBLAS_HAS_ZABS
    inc_counter("egblas");
    egblas_zabs(n, alpha, reinterpret_cast<cuDoubleComplex*>(A), lda, (B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::abs");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas abs operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void abs(size_t n, double alpha, etl::complex<double>* A , size_t lda, double* B , size_t ldb){
#ifdef EGBLAS_HAS_ZABS
    inc_counter("egblas");
    egblas_zabs(n, alpha, reinterpret_cast<cuDoubleComplex*>(A), lda, (B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::abs");
#endif
}

} //end of namespace etl::impl::egblas
