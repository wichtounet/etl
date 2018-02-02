//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief CUBLAS wrappers for the axpy operation.
 */

#pragma once

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"

#endif

namespace etl::impl::cublas {

#ifdef ETL_CUBLAS_MODE

/*!
 * \brief Wrappers for single-precision cublas axpy operation
 * \param handle The CUBLAS handle
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cublas_axpy(cublasHandle_t handle, size_t n, const float* alpha, const float* A , size_t lda, float* B , size_t ldb){
    inc_counter("cublas");
    cublas_check(cublasSaxpy(handle, n, alpha, A, lda, B, ldb));
}

/*!
 * \brief Wrappers for double-precision cublas axpy operation
 * \param handle The CUBLAS handle
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cublas_axpy(cublasHandle_t handle, size_t n, const double* alpha, const double* A , size_t lda, double* B , size_t ldb){
    inc_counter("cublas");
    cublas_check(cublasDaxpy(handle, n, alpha, A, lda, B, ldb));
}

/*!
 * \brief Wrappers for complex single-precision cublas axpy operation
 * \param handle The CUBLAS handle
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cublas_axpy(cublasHandle_t handle, size_t n, const std::complex<float>* alpha, const std::complex<float>* A , size_t lda, std::complex<float>* B , size_t ldb){
    inc_counter("cublas");
    cublas_check(cublasCaxpy(handle, n, reinterpret_cast<const cuComplex*>(alpha), reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb));
}

/*!
 * \brief Wrappers for complex double-precision cublas axpy operation
 * \param handle The CUBLAS handle
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cublas_axpy(cublasHandle_t handle, size_t n, const std::complex<double>* alpha, const std::complex<double>* A , size_t lda, std::complex<double>* B , size_t ldb){
    inc_counter("cublas");
    cublas_check(cublasZaxpy(handle, n, reinterpret_cast<const cuDoubleComplex*>(alpha), reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb));
}

/*!
 * \brief Wrappers for complex single-precision cublas axpy operation
 * \param handle The CUBLAS handle
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cublas_axpy(cublasHandle_t handle, size_t n, const etl::complex<float>* alpha, const etl::complex<float>* A , size_t lda, etl::complex<float>* B , size_t ldb){
    inc_counter("cublas");
    cublas_check(cublasCaxpy(handle, n, reinterpret_cast<const cuComplex*>(alpha), reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb));
}

/*!
 * \brief Wrappers for complex double-precision cublas axpy operation
 * \param handle The CUBLAS handle
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cublas_axpy(cublasHandle_t handle, size_t n, const etl::complex<double>* alpha, const etl::complex<double>* A , size_t lda, etl::complex<double>* B , size_t ldb){
    inc_counter("cublas");
    cublas_check(cublasZaxpy(handle, n, reinterpret_cast<const cuDoubleComplex*>(alpha), reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb));
}

#endif

} //end of namespace etl::impl::cublas
