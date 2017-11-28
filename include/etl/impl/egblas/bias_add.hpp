//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the bias_add_2d operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/util/counters.hpp"

#include <egblas.hpp>

#endif

namespace etl {

namespace impl {

namespace egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision absarithm.
 */
#ifdef EGBLAS_HAS_SBIAS_ADD_2D
static constexpr bool has_sbias_add_2d = true;
#else
static constexpr bool has_sbias_add_2d = false;
#endif

/*!
 * \brief Add the 1D bias to to the batched 1D input
 * \param m The first dimension of x and y
 * \param n The second dimension of x and y
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param b The vector x (GPU memory)
 * \param incb The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
inline void bias_add_2d(size_t m, size_t n, const float* x, size_t incx, const float* b, size_t incb, float* y, size_t incy){
#ifdef EGBLAS_HAS_SBIAS_ADD_2D
    inc_counter("egblas");
    egblas_sbias_add_2d(m, n, x, incx, b, incb, y, incy);
#else
    cpp_unused(m);
    cpp_unused(n);
    cpp_unused(x);
    cpp_unused(incx);
    cpp_unused(b);
    cpp_unused(incb);
    cpp_unused(y);
    cpp_unused(incy);

    cpp_unreachable("Invalid call to egblas::bias_add_2d");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision absarithm.
 */
#ifdef EGBLAS_HAS_DBIAS_ADD_2D
static constexpr bool has_dbias_add_2d = true;
#else
static constexpr bool has_dbias_add_2d = false;
#endif

/*!
 * \brief Add the 1D bias to to the batched 1D input
 * \param m The first dimension of x and y
 * \param n The second dimension of x and y
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param b The vector x (GPU memory)
 * \param incb The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
inline void bias_add_2d(size_t m, size_t n, const double* x, size_t incx, const double* b, size_t incb, double* y, size_t incy){
#ifdef EGBLAS_HAS_DBIAS_ADD_2D
    inc_counter("egblas");
    egblas_dbias_add_2d(m, n, x, incx, b, incb, y, incy);
#else
    cpp_unused(m);
    cpp_unused(n);
    cpp_unused(x);
    cpp_unused(incx);
    cpp_unused(b);
    cpp_unused(incb);
    cpp_unused(y);
    cpp_unused(incy);

    cpp_unreachable("Invalid call to egblas::bias_add_2d");
#endif
}

/*!
 * \brief Indicates if EGBLAS has single-precision absarithm.
 */
#ifdef EGBLAS_HAS_SBIAS_ADD_4D
static constexpr bool has_sbias_add_4d = true;
#else
static constexpr bool has_sbias_add_4d = false;
#endif

/*!
 * \brief Add the 1D bias to to the batched 1D input
 * \param m The first dimension of x and y
 * \param n The second dimension of x and y
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param b The vector x (GPU memory)
 * \param incb The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
inline void bias_add_4d(size_t m, size_t n, size_t o, size_t p, const float* x, size_t incx, const float* b, size_t incb, float* y, size_t incy){
#ifdef EGBLAS_HAS_SBIAS_ADD_4D
    inc_counter("egblas");
    egblas_sbias_add_4d(m, n, o, p, x, incx, b, incb, y, incy);
#else
    cpp_unused(m);
    cpp_unused(n);
    cpp_unused(o);
    cpp_unused(p);
    cpp_unused(x);
    cpp_unused(incx);
    cpp_unused(b);
    cpp_unused(incb);
    cpp_unused(y);
    cpp_unused(incy);

    cpp_unreachable("Invalid call to egblas::bias_add_4d");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision absarithm.
 */
#ifdef EGBLAS_HAS_DBIAS_ADD_4D
static constexpr bool has_dbias_add_4d = true;
#else
static constexpr bool has_dbias_add_4d = false;
#endif

/*!
 * \brief Add the 1D bias to to the batched 1D input
 * \param m The first dimension of x and y
 * \param n The second dimension of x and y
 * \param x The vector x (GPU memory)
 * \param incx The stride of x
 * \param b The vector x (GPU memory)
 * \param incb The stride of x
 * \param y The vector y (GPU memory)
 * \param incy The stride of y
 */
inline void bias_add_4d(size_t m, size_t n, size_t o, size_t p, const double* x, size_t incx, const double* b, size_t incb, double* y, size_t incy){
#ifdef EGBLAS_HAS_DBIAS_ADD_4D
    inc_counter("egblas");
    egblas_dbias_add_4d(m, n, o, p, x, incx, b, incb, y, incy);
#else
    cpp_unused(m);
    cpp_unused(n);
    cpp_unused(o);
    cpp_unused(p);
    cpp_unused(x);
    cpp_unused(incx);
    cpp_unused(b);
    cpp_unused(incb);
    cpp_unused(y);
    cpp_unused(incy);

    cpp_unreachable("Invalid call to egblas::bias_add_4d");
#endif
}

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
