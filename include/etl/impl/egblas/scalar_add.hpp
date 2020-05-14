//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the scalar_add operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SCALAR_SADD

static constexpr bool has_scalar_sadd = true;

/*!
 * \brief Adds the scalar beta to each element of the single-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
inline void scalar_add(float* x, size_t n, size_t s, const float beta) {
    inc_counter("egblas");
    egblas_scalar_sadd(x, n, s, beta);
}

#else

static constexpr bool has_scalar_sadd = false;

#endif

#ifdef EGBLAS_HAS_SCALAR_DADD

static constexpr bool has_scalar_dadd = true;

/*!
 * \brief Adds the scalar beta to each element of the double-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
inline void scalar_add(double* x, size_t n, size_t s, const double beta) {
    inc_counter("egblas");
    egblas_scalar_dadd(x, n, s, beta);
}

#else

static constexpr bool has_scalar_dadd = false;

#endif

#ifdef EGBLAS_HAS_SCALAR_CADD

static constexpr bool has_scalar_cadd = true;

/*!
 * \brief Adds the scalar beta to each element of the complex single-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
inline void scalar_add(etl::complex<float>* x, size_t n, size_t s, const etl::complex<float> beta) {
    inc_counter("egblas");
    egblas_scalar_cadd(reinterpret_cast<cuComplex*>(x), n, s, complex_cast(beta));
}

/*!
 * \brief Adds the scalar beta to each element of the complex single-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
inline void scalar_add(std::complex<float>* x, size_t n, size_t s, const std::complex<float> beta) {
    inc_counter("egblas");
    egblas_scalar_cadd(reinterpret_cast<cuComplex*>(x), n, s, complex_cast(beta));
}

#else

static constexpr bool has_scalar_cadd = false;

#endif

#ifdef EGBLAS_HAS_SCALAR_ZADD

static constexpr bool has_scalar_zadd = true;

/*!
 * \brief Adds the scalar beta to each element of the complex single-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
inline void scalar_add(etl::complex<double>* x, size_t n, size_t s, const etl::complex<double> beta) {
    inc_counter("egblas");
    egblas_scalar_zadd(reinterpret_cast<cuDoubleComplex*>(x), n, s, complex_cast(beta));
}

/*!
 * \brief Adds the scalar beta to each element of the complex single-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
inline void scalar_add(std::complex<double>* x, size_t n, size_t s, const std::complex<double> beta) {
    inc_counter("egblas");
    egblas_scalar_zadd(reinterpret_cast<cuDoubleComplex*>(x), n, s, complex_cast(beta));
}

#else

static constexpr bool has_scalar_zadd = false;

#endif

#ifndef ETL_EGBLAS_MODE

/*!
 * \brief Adds the scalar beta to each element of the single-precision vector x
 * \param x The vector to add the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to add
 */
template <typename T>
inline void scalar_add(T* x, size_t n, size_t s, const T beta) {
    cpp_unused(x);
    cpp_unused(n);
    cpp_unused(s);
    cpp_unused(beta);

    cpp_unreachable("Invalid call to egblas::scalar_add");
}

#endif

} //end of namespace etl::impl::egblas
