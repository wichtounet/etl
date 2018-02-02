//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the apxdbpy operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SAPXDBPY
static constexpr bool has_sapxdbpy = true;
#else
static constexpr bool has_sapxdbpy = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas apxdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy(size_t n, float alpha, float* A, size_t lda, float beta, float* B, size_t ldb) {
#ifdef EGBLAS_HAS_SAPXDBPY
    inc_counter("egblas");
    egblas_sapxdbpy(n, alpha, A, lda, beta, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(beta);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::apxdbpy");
#endif
}

#ifdef EGBLAS_HAS_DAPXDBPY
static constexpr bool has_dapxdbpy = true;
#else
static constexpr bool has_dapxdbpy = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas apxdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy(size_t n, double alpha, double* A , size_t lda, double beta, double* B , size_t ldb){
#ifdef EGBLAS_HAS_DAPXDBPY
    inc_counter("egblas");
    egblas_dapxdbpy(n, alpha, A, lda, beta, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(beta);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::apxdbpy");
#endif
}

#ifdef EGBLAS_HAS_CAPXDBPY
static constexpr bool has_capxdbpy = true;
#else
static constexpr bool has_capxdbpy = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas apxdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy(size_t n, std::complex<float> alpha, std::complex<float>* A , size_t lda, std::complex<float> beta, std::complex<float>* B , size_t ldb){
#ifdef EGBLAS_HAS_CAPXDBPY
    inc_counter("egblas");
    egblas_capxdbpy(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(beta);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::apxdbpy");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas apxdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy(size_t n, etl::complex<float> alpha, etl::complex<float>* A , size_t lda, std::complex<float> beta, etl::complex<float>* B , size_t ldb){
#ifdef EGBLAS_HAS_CAPXDBPY
    inc_counter("egblas");
    egblas_capxdbpy(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(beta);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::apxdbpy");
#endif
}


#ifdef EGBLAS_HAS_ZAPXDBPY
static constexpr bool has_zapxdbpy = true;
#else
static constexpr bool has_zapxdbpy = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas apxdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy(size_t n, std::complex<double> alpha, std::complex<double>* A , size_t lda, std::complex<double> beta, std::complex<double>* B , size_t ldb){
#ifdef EGBLAS_HAS_ZAPXDBPY
    inc_counter("egblas");
    egblas_zapxdbpy(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(beta);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::apxdbpy");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas apxdbpy operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void apxdbpy(size_t n, etl::complex<double> alpha, etl::complex<double>* A , size_t lda, std::complex<double> beta, etl::complex<double>* B , size_t ldb){
#ifdef EGBLAS_HAS_ZAPXDBPY
    inc_counter("egblas");
    egblas_zapxdbpy(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, complex_cast(beta), reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(beta);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::apxdbpy");
#endif
}

} //end of namespace etl::impl::egblas
