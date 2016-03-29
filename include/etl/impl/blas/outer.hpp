//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief BLAS implementation of the outer product
 */

#pragma once

#ifdef ETL_BLAS_MODE
#include "cblas.h"
#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_BLAS_MODE

/*!
 * \brief Compute the outer product of a and b and store the result in c
 * \param a The lhs expression
 * \param b The rhs expression
 * \param c The output expression
 */
template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value&& all_single_precision<A, B, C>::value)>
void outer(const A& a, const B& b, C&& c) {
    c = 0;

    cblas_sger(
        CblasRowMajor,
        etl::dim<0>(a), etl::dim<0>(b),
        1.0,
        a.memory_start(), 1,
        b.memory_start(), 1,
        c.memory_start(), etl::dim<0>(b));
}

/*!
 * \copydoc outer
 */
template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value&& all_double_precision<A, B, C>::value)>
void outer(const A& a, const B& b, C&& c) {
    c = 0;

    cblas_dger(
        CblasRowMajor,
        etl::dim<0>(a), etl::dim<0>(b),
        1.0,
        a.memory_start(), 1,
        b.memory_start(), 1,
        c.memory_start(), etl::dim<0>(b));
}

/*!
 * \copydoc outer
 */
template <typename A, typename B, typename C, cpp_enable_if(!all_dma<A, B>::value)>
void outer(const A& /*a*/, const B& /*b*/, C&& /*c*/) {
    cpp_unreachable("BLAS not enabled/available");
}

#else

/*!
 * \copydoc outer
 */
template <typename A, typename B, typename C>
void outer(const A& /*a*/, const B& /*b*/, C&& /*c*/) {
    cpp_unreachable("BLAS not enabled/available");
}

#endif

} //end of namespace blas
} //end of namespace impl
} //end of namespace etl
