//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the shuffle operations.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl {

namespace impl {

namespace egblas {

/*!
 * \brief Indicates if EGBLAS has shuffle
 */
#ifdef EGBLAS_HAS_SHUFFLE
static constexpr bool has_shuffle = true;
#else
static constexpr bool has_shuffle = false;
#endif

/*!
 * \brief Wrappers for egblas shuffle operation
 * \param n The number of elements of the vector
 * \param x The vector to shuffle (GPU memory)
 * \param incx The size of each element of the vector
 */
inline void shuffle(size_t n, void* x, size_t incx){
#ifdef EGBLAS_HAS_SHUFFLE
    egblas_shuffle(n, x, incx);
#else
    cpp_unused(n);
    cpp_unused(x);
    cpp_unused(incx);

    cpp_unreachable("Invalid call to egblas::shuffle");
#endif
}

/*!
 * \brief Indicates if EGBLAS has shuffle_seed
 */
#ifdef EGBLAS_HAS_SHUFFLE_SEED
static constexpr bool has_shuffle_seed = true;
#else
static constexpr bool has_shuffle_seed = false;
#endif

/*!
 * \brief Wrappers for egblas shuffle_seed operation
 * \param n The number of elements of the vector
 * \param x The vector to shuffle (GPU memory)
 * \param incx The size of each element of the vector
 * \param seed The seed to initialize the random generator with
 */
inline void shuffle_seed(size_t n, void* x, size_t incx, size_t seed){
#ifdef EGBLAS_HAS_SHUFFLE_SEED
    egblas_shuffle_seed(n, x, incx, seed);
#else
    cpp_unused(n);
    cpp_unused(x);
    cpp_unused(incx);
    cpp_unused(seed);

    cpp_unreachable("Invalid call to egblas::shuffle_seed");
#endif
}

/*!
 * \brief Indicates if EGBLAS has par_shuffle
 */
#ifdef EGBLAS_HAS_PAR_SHUFFLE
static constexpr bool has_par_shuffle = true;
#else
static constexpr bool has_par_shuffle = false;
#endif

/*!
 * \brief Wrappers for egblas par_shuffle operation
 * \param n The number of elements of the vector
 * \param x The first vector to shuffle (GPU memory)
 * \param incx The size of each element of the first vector
 * \param y The second vector to shuffle (GPU memory)
 * \param incy The size of each element of the second vector
 */
inline void par_shuffle(size_t n, void* x, size_t incx, void* y, size_t incy){
#ifdef EGBLAS_HAS_PAR_SHUFFLE
    egblas_par_shuffle(n, x, incx, y, incy);
#else
    cpp_unused(n);
    cpp_unused(x);
    cpp_unused(incx);
    cpp_unused(y);
    cpp_unused(incy);

    cpp_unreachable("Invalid call to egblas::par_shuffle");
#endif
}

/*!
 * \brief Indicates if EGBLAS has par_shuffle_seed
 */
#ifdef EGBLAS_HAS_PAR_SHUFFLE_SEED
static constexpr bool has_par_shuffle_seed = true;
#else
static constexpr bool has_par_shuffle_seed = false;
#endif

/*!
 * \brief Wrappers for egblas par_shuffle_seed operation
 * \param n The number of elements of the vector
 * \param x The first vector to shuffle (GPU memory)
 * \param incx The size of each element of the first vector
 * \param y The second vector to shuffle (GPU memory)
 * \param incy The size of each element of the second vector
 * \param seed The seed to initialize the random generator with
 */
inline void par_shuffle_seed(size_t n, void* x, size_t incx, void* y, size_t incy, size_t seed){
#ifdef EGBLAS_HAS_PAR_SHUFFLE_SEED
    egblas_par_shuffle_seed(n, x, incx, y, incy, seed);
#else
    cpp_unused(n);
    cpp_unused(x);
    cpp_unused(incx);
    cpp_unused(y);
    cpp_unused(incy);
    cpp_unused(seed);

    cpp_unreachable("Invalid call to egblas::par_shuffle");
#endif
}

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
