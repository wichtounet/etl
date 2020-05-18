//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

namespace etl::impl::egblas {

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
inline void bias_add_2d([[maybe_unused]] size_t m,
                        [[maybe_unused]] size_t n,
                        [[maybe_unused]] const float* x,
                        [[maybe_unused]] size_t incx,
                        [[maybe_unused]] const float* b,
                        [[maybe_unused]] size_t incb,
                        [[maybe_unused]] float* y,
                        [[maybe_unused]] size_t incy) {
#ifdef EGBLAS_HAS_SBIAS_ADD_2D
    inc_counter("egblas");
    egblas_sbias_add_2d(m, n, x, incx, b, incb, y, incy);
#else
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
inline void bias_add_2d([[maybe_unused]] size_t m,
                        [[maybe_unused]] size_t n,
                        [[maybe_unused]] const double* x,
                        [[maybe_unused]] size_t incx,
                        [[maybe_unused]] const double* b,
                        [[maybe_unused]] size_t incb,
                        [[maybe_unused]] double* y,
                        [[maybe_unused]] size_t incy) {
#ifdef EGBLAS_HAS_DBIAS_ADD_2D
    inc_counter("egblas");
    egblas_dbias_add_2d(m, n, x, incx, b, incb, y, incy);
#else
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
inline void bias_add_4d([[maybe_unused]] size_t m,
                        [[maybe_unused]] size_t n,
                        [[maybe_unused]] size_t o,
                        [[maybe_unused]] size_t p,
                        [[maybe_unused]] const float* x,
                        [[maybe_unused]] size_t incx,
                        [[maybe_unused]] const float* b,
                        [[maybe_unused]] size_t incb,
                        [[maybe_unused]] float* y,
                        [[maybe_unused]] size_t incy) {
#ifdef EGBLAS_HAS_SBIAS_ADD_4D
    inc_counter("egblas");
    egblas_sbias_add_4d(m, n, o, p, x, incx, b, incb, y, incy);
#else
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
inline void bias_add_4d([[maybe_unused]] size_t m,
                        [[maybe_unused]] size_t n,
                        [[maybe_unused]] size_t o,
                        [[maybe_unused]] size_t p,
                        [[maybe_unused]] const double* x,
                        [[maybe_unused]] size_t incx,
                        [[maybe_unused]] const double* b,
                        [[maybe_unused]] size_t incb,
                        [[maybe_unused]] double* y,
                        [[maybe_unused]] size_t incy) {
#ifdef EGBLAS_HAS_DBIAS_ADD_4D
    inc_counter("egblas");
    egblas_dbias_add_4d(m, n, o, p, x, incx, b, incb, y, incy);
#else
    cpp_unreachable("Invalid call to egblas::bias_add_4d");
#endif
}

} //end of namespace etl::impl::egblas
