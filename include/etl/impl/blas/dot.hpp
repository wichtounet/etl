//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief BLAS implementation of the "dot" reduction
 */

#pragma once

#ifdef ETL_BLAS_MODE
#include "cblas.h" //For ddot/sdot
#endif

namespace etl::impl::blas {

#ifdef ETL_BLAS_MODE

/*!
 * \brief Compute the dot product of a and b
 * \param a The lhs expression
 * \param b The rhs expression
 * \return the sum
 */
template <typename A, typename B>
value_t<A> dot(const A& a, const B& b) {
    if constexpr (all_dma<A, B>) {
        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();

        if constexpr (all_single_precision<A, B>) {
            return cblas_sdot(etl::size(a), a.memory_start(), 1, b.memory_start(), 1);
        } else {
            return cblas_ddot(etl::size(a), a.memory_start(), 1, b.memory_start(), 1);
        }
    } else {
        cpp_unreachable("BLAS not enabled/available");
        return 0.0;
    }
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \copydoc dot
 */
template <typename A, typename B>
value_t<A> dot(const A& /*a*/, const B& /*b*/) {
    cpp_unreachable("BLAS not enabled/available");
    return 0.0;
}

    //COVERAGE_EXCLUDE_END

#endif

} //end of namespace etl::impl::blas
