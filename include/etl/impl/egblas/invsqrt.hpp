//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the invsqrt operation.
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
 * \brief Indicates if EGBLAS has single-precision inverse square root.
 */
#ifdef EGBLAS_HAS_SINVSQRT
static constexpr bool has_sinvsqrt = true;
#else
static constexpr bool has_sinvsqrt = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt(size_t n, float alpha, float* A , size_t lda, float* B , size_t ldb){
#ifdef EGBLAS_HAS_SINVSQRT
    egblas_sinvsqrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision inverse square root.
 */
#ifdef EGBLAS_HAS_DINVSQRT
static constexpr bool has_dinvsqrt = true;
#else
static constexpr bool has_dinvsqrt = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt(size_t n, double alpha, double* A , size_t lda, double* B , size_t ldb){
#ifdef EGBLAS_HAS_DINVSQRT
    egblas_dinvsqrt(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision inverse square root.
 */
#ifdef EGBLAS_HAS_CINVSQRT
static constexpr bool has_cinvsqrt = true;
#else
static constexpr bool has_cinvsqrt = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt(size_t n, std::complex<float> alpha, std::complex<float>* A , size_t lda, std::complex<float>* B , size_t ldb){
#ifdef EGBLAS_HAS_CINVSQRT
    egblas_cinvsqrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt(size_t n, etl::complex<float> alpha, etl::complex<float>* A , size_t lda, etl::complex<float>* B , size_t ldb){
#ifdef EGBLAS_HAS_CINVSQRT
    egblas_cinvsqrt(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision inverse square root.
 */
#ifdef EGBLAS_HAS_ZINVSQRT
static constexpr bool has_zinvsqrt = true;
#else
static constexpr bool has_zinvsqrt = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt(size_t n, std::complex<double> alpha, std::complex<double>* A , size_t lda, std::complex<double>* B , size_t ldb){
#ifdef EGBLAS_HAS_ZINVSQRT
    egblas_zinvsqrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas invsqrt operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void invsqrt(size_t n, etl::complex<double> alpha, etl::complex<double>* A , size_t lda, etl::complex<double>* B , size_t ldb){
#ifdef EGBLAS_HAS_ZINVSQRT
    egblas_zinvsqrt(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::invsqrt");
#endif
}

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
