//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the cosh operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has coshgle-precision hyperbolic cosine.
 */
#ifdef EGBLAS_HAS_SCOSH
static constexpr bool has_scosh = true;
#else
static constexpr bool has_scosh = false;
#endif

/*!
 * \brief Wrappers for coshgle-precision egblas cosh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cosh(size_t n, float alpha, float* A, size_t lda, float* B, size_t ldb) {
#ifdef EGBLAS_HAS_SCOSH
    inc_counter("egblas");
    egblas_scosh(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::cosh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision hyperbolic cosine.
 */
#ifdef EGBLAS_HAS_DCOSH
static constexpr bool has_dcosh = true;
#else
static constexpr bool has_dcosh = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas cosh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cosh(size_t n, double alpha, double* A, size_t lda, double* B, size_t ldb) {
#ifdef EGBLAS_HAS_DCOSH
    inc_counter("egblas");
    egblas_dcosh(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::cosh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex coshgle-precision hyperbolic cosine.
 */
#ifdef EGBLAS_HAS_CCOSH
static constexpr bool has_ccosh = true;
#else
static constexpr bool has_ccosh = false;
#endif

/*!
 * \brief Wrappers for complex coshgle-precision egblas cosh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cosh(size_t n, std::complex<float> alpha, std::complex<float>* A, size_t lda, std::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CCOSH
    inc_counter("egblas");
    egblas_ccosh(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::cosh");
#endif
}

/*!
 * \brief Wrappers for complex coshgle-precision egblas cosh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cosh(size_t n, etl::complex<float> alpha, etl::complex<float>* A, size_t lda, etl::complex<float>* B, size_t ldb) {
#ifdef EGBLAS_HAS_CCOSH
    inc_counter("egblas");
    egblas_ccosh(n, complex_cast(alpha), reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::cosh");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision hyperbolic cosine.
 */
#ifdef EGBLAS_HAS_ZCOSH
static constexpr bool has_zcosh = true;
#else
static constexpr bool has_zcosh = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas cosh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cosh(size_t n, std::complex<double> alpha, std::complex<double>* A, size_t lda, std::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZCOSH
    inc_counter("egblas");
    egblas_zcosh(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::cosh");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas cosh operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void cosh(size_t n, etl::complex<double> alpha, etl::complex<double>* A, size_t lda, etl::complex<double>* B, size_t ldb) {
#ifdef EGBLAS_HAS_ZCOSH
    inc_counter("egblas");
    egblas_zcosh(n, complex_cast(alpha), reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(B), ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::cosh");
#endif
}

} //end of namespace etl::impl::egblas
