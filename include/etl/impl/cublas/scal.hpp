//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief CUBLAS wrappers for the scal operation.
 */

#pragma once

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"

#endif

namespace etl::impl::cublas {

#ifdef ETL_CUBLAS_MODE

/*!
 * \brief Wrappers for single-precision cublas scal operation
 * \param handle The CUBLAS handle
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void cublas_scal(cublasHandle_t handle, size_t n, const float* alpha, float* A, size_t lda) {
    inc_counter("cublas");
    cublas_check(cublasSscal(handle, n, alpha, A, lda));
}

/*!
 * \brief Wrappers for single-precision cublas scal operation
 * \param handle The CUBLAS handle
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void cublas_scal(cublasHandle_t handle, size_t n, const double* alpha, double* A, size_t lda) {
    inc_counter("cublas");
    cublas_check(cublasDscal(handle, n, alpha, A, lda));
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
inline void cublas_scal(cublasHandle_t handle, size_t n, const std::complex<float>* alpha, std::complex<float>* A, size_t lda) {
    inc_counter("cublas");
    cublas_check(cublasCscal(handle, n, reinterpret_cast<const cuComplex*>(alpha), reinterpret_cast<cuComplex*>(A), lda));
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
inline void cublas_scal(cublasHandle_t handle, size_t n, const std::complex<double>* alpha, std::complex<double>* A, size_t lda) {
    inc_counter("cublas");
    cublas_check(cublasZscal(handle, n, reinterpret_cast<const cuDoubleComplex*>(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda));
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
inline void cublas_scal(cublasHandle_t handle, size_t n, const etl::complex<float>* alpha, etl::complex<float>* A, size_t lda) {
    inc_counter("cublas");
    cublas_check(cublasCscal(handle, n, reinterpret_cast<const cuComplex*>(alpha), reinterpret_cast<cuComplex*>(A), lda));
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
inline void cublas_scal(cublasHandle_t handle, size_t n, const etl::complex<double>* alpha, etl::complex<double>* A, size_t lda) {
    inc_counter("cublas");
    cublas_check(cublasZscal(handle, n, reinterpret_cast<const cuDoubleComplex*>(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda));
}

#endif

} //end of namespace etl::impl::cublas
