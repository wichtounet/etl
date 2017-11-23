//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the scalar_set operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl {

namespace impl {

namespace egblas {

#ifdef EGBLAS_HAS_SCALAR_SSET

static constexpr bool has_scalar_sset = true;

/*!
 * \brief sets the scalar beta to each element of the single-precision vector x
 * \param x The vector to set the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to set
 */
inline void scalar_set(float* x, size_t n, size_t s, const float beta){
    egblas_scalar_sset(x, n, s, beta);
}

#else

static constexpr bool has_scalar_sset = false;

#endif

#ifdef EGBLAS_HAS_SCALAR_DSET

static constexpr bool has_scalar_dset = true;

/*!
 * \brief sets the scalar beta to each element of the double-precision vector x
 * \param x The vector to set the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to set
 */
inline void scalar_set(double* x, size_t n, size_t s, const double beta){
    egblas_scalar_dset(x, n, s, beta);
}

#else

static constexpr bool has_scalar_dset = false;

#endif

#ifndef ETL_EGBLAS_MODE

/*!
 * \brief sets the scalar beta to each element of the single-precision vector x
 * \param x The vector to set the scalar to (GPU pointer)
 * \param n The size of the vector
 * \param s The stride of the vector
 * \param beta The scalar to set
 */
template<typename T>
inline void scalar_set(T* x, size_t n, size_t s, const T beta){
    cpp_unused(x);
    cpp_unused(n);
    cpp_unused(s);
    cpp_unused(beta);

    cpp_unreachable("Invalid call to egblas::scalar_set");
}

#endif

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
