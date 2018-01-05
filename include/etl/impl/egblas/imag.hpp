//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the imag operation.
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
 * \brief Indicates if EGBLAS has complex single-precision imag.
 */
#ifdef EGBLAS_HAS_Cimag
static constexpr bool has_cimag = true;
#else
static constexpr bool has_cimag = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas imag operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void imag(size_t n, float alpha, std::complex<float>* A , size_t lda, float* B , size_t ldb){
#ifdef EGBLAS_HAS_Cimag
    inc_counter("egblas");
    egblas_cimag(n, alpha, reinterpret_cast<cuComplex*>(A), lda, (B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::imag");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas imag operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void imag(size_t n, float alpha, etl::complex<float>* A , size_t lda, float* B , size_t ldb){
#ifdef EGBLAS_HAS_Cimag
    inc_counter("egblas");
    egblas_cimag(n, alpha, reinterpret_cast<cuComplex*>(A), lda, (B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::imag");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision imag.
 */
#ifdef EGBLAS_HAS_Zimag
static constexpr bool has_zimag = true;
#else
static constexpr bool has_zimag = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas imag operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void imag(size_t n, double alpha, std::complex<double>* A , size_t lda, double* B , size_t ldb){
#ifdef EGBLAS_HAS_Zimag
    inc_counter("egblas");
    egblas_zimag(n, alpha, reinterpret_cast<cuDoubleComplex*>(A), lda, (B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::imag");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas imag operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void imag(size_t n, double alpha, etl::complex<double>* A , size_t lda, double* B , size_t ldb){
#ifdef EGBLAS_HAS_Zimag
    inc_counter("egblas");
    egblas_zimag(n, alpha, reinterpret_cast<cuDoubleComplex*>(A), lda, (B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::imag");
#endif
}

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
