//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the clip_value operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/util/complex_cast.hpp"
#include "etl/util/safe_cast.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision clip_value.
 */
#ifdef EGBLAS_HAS_SCLIP_VALUE
static constexpr bool has_sclip_value = true;
#else
static constexpr bool has_sclip_value = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas clip_value operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip_value(size_t n, const float alpha, float A, float B, float* C, size_t ldc) {
#ifdef EGBLAS_HAS_SCLIP_VALUE
    inc_counter("egblas");
    egblas_sclip_value(n, alpha, A, B, C, ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(B);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::clip_value");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision clip_value.
 */
#ifdef EGBLAS_HAS_DCLIP_VALUE
static constexpr bool has_dclip_value = true;
#else
static constexpr bool has_dclip_value = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas clip_value operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip_value(size_t n, double alpha, double A, double B, double* C, size_t ldc) {
#ifdef EGBLAS_HAS_DCLIP_VALUE
    inc_counter("egblas");
    egblas_dclip_value(n, alpha, A, B, C, ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(B);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::clip_value");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex single-precision clip_value.
 */
#ifdef EGBLAS_HAS_CCLIP_VALUE
static constexpr bool has_cclip_value = true;
#else
static constexpr bool has_cclip_value = false;
#endif

/*!
 * \brief Wrappers for complex single-precision egblas clip_value operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip_value(size_t n, const std::complex<float> alpha, std::complex<float> A, std::complex<float> B, std::complex<float>* C, size_t ldc) {
#ifdef EGBLAS_HAS_CCLIP_VALUE
    inc_counter("egblas");
    egblas_cclip_value(n,
        complex_cast(alpha), complex_cast(A),
        complex_cast(B),
        reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(B);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::clip_value");
#endif
}

/*!
 * \brief Wrappers for complex single-precision egblas clip_value operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip_value(size_t n, const etl::complex<float> alpha, etl::complex<float> A, etl::complex<float> B, etl::complex<float>* C, size_t ldc) {
#ifdef EGBLAS_HAS_CCLIP_VALUE
    inc_counter("egblas");
    egblas_cclip_value(n,
        complex_cast(alpha), complex_cast(A),
        complex_cast(B),
        reinterpret_cast<cuComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(B);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::clip_value");
#endif
}

/*!
 * \brief Indicates if EGBLAS has complex double-precision clip_value.
 */
#ifdef EGBLAS_HAS_ZCLIP_VALUE
static constexpr bool has_zclip_value = true;
#else
static constexpr bool has_zclip_value = false;
#endif

/*!
 * \brief Wrappers for complex double-precision egblas clip_value operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip_value(size_t n, const std::complex<double> alpha, std::complex<double> A, std::complex<double> B, std::complex<double>* C, size_t ldc) {
#ifdef EGBLAS_HAS_ZCLIP_VALUE
    inc_counter("egblas");
    egblas_zclip_value(n,
        complex_cast(alpha), complex_cast(A),
        complex_cast(B),
        reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(B);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::clip_value");
#endif
}

/*!
 * \brief Wrappers for complex double-precision egblas clip_value operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param B The memory of the vector b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void clip_value(size_t n, const etl::complex<double> alpha, etl::complex<double> A, etl::complex<double> B, etl::complex<double>* C, size_t ldc) {
#ifdef EGBLAS_HAS_ZCLIP_VALUE
    inc_counter("egblas");
    egblas_zclip_value(n,
        complex_cast(alpha), complex_cast(A),
        complex_cast(B),
        reinterpret_cast<cuDoubleComplex*>(C), ldc);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(B);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::clip_value");
#endif
}

} //end of namespace etl::impl::egblas
