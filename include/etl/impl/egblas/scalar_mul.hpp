//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the scalar_mul operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl {

namespace impl {

namespace egblas {

#ifdef EGBLAS_HAS_SCALAR_SMUL

static constexpr bool has_scalar_smul = true;

/*!
 * \brief Muls the scalar beta to each element of the single-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
inline void scalar_mul(float* x, size_t n, size_t s, const float beta){
    inc_counter("egblas");
    egblas_scalar_smul(x, n, s, beta);
}

#else

static constexpr bool has_scalar_smul = false;

#endif

#ifdef EGBLAS_HAS_SCALAR_DMUL

static constexpr bool has_scalar_dmul = true;

/*!
 * \brief Muls the scalar beta to each element of the double-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
inline void scalar_mul(double* x, size_t n, size_t s, const double beta){
    inc_counter("egblas");
    egblas_scalar_dmul(x, n, s, beta);
}

#else

static constexpr bool has_scalar_dmul = false;

#endif

#ifdef EGBLAS_HAS_SCALAR_CMUL

static constexpr bool has_scalar_cmul = true;

/*!
 * \brief Muls the scalar beta to each element of the complex single-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
inline void scalar_mul(etl::complex<float>* x, size_t n, size_t s, const etl::complex<float> beta){
    inc_counter("egblas");
    egblas_scalar_cmul(reinterpret_cast<cuComplex*>(x), n, s, complex_cast(beta));
}

/*!
 * \brief Muls the scalar beta to each element of the complex single-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
inline void scalar_mul(std::complex<float>* x, size_t n, size_t s, const std::complex<float> beta){
    inc_counter("egblas");
    egblas_scalar_cmul(reinterpret_cast<cuComplex*>(x), n, s, complex_cast(beta));
}

#else

static constexpr bool has_scalar_cmul = false;

#endif

#ifdef EGBLAS_HAS_SCALAR_ZMUL

static constexpr bool has_scalar_zmul = true;

/*!
 * \brief Muls the scalar beta to each element of the complex single-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
inline void scalar_mul(etl::complex<double>* x, size_t n, size_t s, const etl::complex<double> beta){
    inc_counter("egblas");
    egblas_scalar_zmul(reinterpret_cast<cuDoubleComplex*>(x), n, s, complex_cast(beta));
}

/*!
 * \brief Muls the scalar beta to each element of the complex single-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
inline void scalar_mul(std::complex<double>* x, size_t n, size_t s, const std::complex<double> beta){
    inc_counter("egblas");
    egblas_scalar_zmul(reinterpret_cast<cuDoubleComplex*>(x), n, s, complex_cast(beta));
}

#else

static constexpr bool has_scalar_zmul = false;

#endif

#ifndef ETL_EGBLAS_MODE

/*!
 * \brief Muls the scalar beta to each element of the single-precision vector x
 * \param x The vector to mul the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to mul
 */
template<typename T>
inline void scalar_mul(T* x, size_t n, size_t s, const T beta){
    cpp_unused(x);
    cpp_unused(n);
    cpp_unused(s);
    cpp_unused(beta);

    cpp_unreachable("Invalid call to egblas::scalar_mul");
}

#endif

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
