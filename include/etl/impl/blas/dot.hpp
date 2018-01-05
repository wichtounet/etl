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

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_BLAS_MODE

/*!
 * \brief Compute the dot product of a and b
 * \param a The lhs expression
 * \param b The rhs expression
 * \return the sum
 */
template <typename A, typename B, cpp_enable_iff(all_dma<A, B>&& all_single_precision<A, B>)>
value_t<A> dot(const A& a, const B& b) {
    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    const float* m_a = a.memory_start();
    const float* m_b = b.memory_start();

    return cblas_sdot(etl::size(a), m_a, 1, m_b, 1);
}

/*!
 * \copydoc dot
 */
template <typename A, typename B, cpp_enable_iff(all_dma<A, B>&& all_double_precision<A, B>)>
value_t<A> dot(const A& a, const B& b) {
    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    const double* m_a = a.memory_start();
    const double* m_b = b.memory_start();

    return cblas_ddot(etl::size(a), m_a, 1, m_b, 1);
}

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \copydoc dot
 */
template <typename A, typename B, cpp_enable_iff(!all_dma<A, B>)>
value_t<A> dot(const A& /*a*/, const B& /*b*/) {
    cpp_unreachable("BLAS not enabled/available");
    return 0.0;
}

#else

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

} //end of namespace blas
} //end of namespace impl
} //end of namespace etl
