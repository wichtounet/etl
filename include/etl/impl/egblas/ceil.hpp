//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace etl {

namespace impl {

namespace egblas {

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
inline void ceil(size_t n, float alpha, float* A , size_t lda, float* B , size_t ldb){
#ifdef EGBLAS_HAS_SCEIL
    inc_counter("egblas");
    egblas_sceil(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void ceil(size_t n, double alpha, double* A , size_t lda, double* B , size_t ldb){
#ifdef EGBLAS_HAS_DCEIL
    inc_counter("egblas");
    egblas_dceil(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void ceil(size_t n, std::complex<float> alpha, std::complex<float>* A , size_t lda, std::complex<float>* B , size_t ldb){
#ifdef EGBLAS_HAS_CCEIL
    inc_counter("egblas");
    egblas_cceil(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void ceil(size_t n, etl::complex<float> alpha, etl::complex<float>* A , size_t lda, etl::complex<float>* B , size_t ldb){
#ifdef EGBLAS_HAS_CCEIL
    inc_counter("egblas");
    egblas_cceil(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void ceil(size_t n, std::complex<double> alpha, std::complex<double>* A , size_t lda, std::complex<double>* B , size_t ldb){
#ifdef EGBLAS_HAS_ZCEIL
    inc_counter("egblas");
    egblas_zceil(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void ceil(size_t n, etl::complex<double> alpha, etl::complex<double>* A , size_t lda, etl::complex<double>* B , size_t ldb){
#ifdef EGBLAS_HAS_ZCEIL
    inc_counter("egblas");
    egblas_zceil(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::ceil");
#endif
}

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
