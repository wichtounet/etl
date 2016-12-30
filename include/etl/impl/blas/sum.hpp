//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief BLAS implementation of the "sum" reduction
 */

#pragma once

#ifdef ETL_BLAS_MODE
#include "cblas.h" //For ddot/sdot
#endif

/*
 * Unfortunately, BLAS library are retarded and do not offer sum,
 * but only asum (absolute sum), therefore, the onyl solution is to
 * use a vector of ones and a dot product
 */

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_BLAS_MODE

/*!
 * \brief Compute the sum product of a
 * \param a The lhs expression
 * \return the sum
 */
template <typename A, cpp_enable_if(all_dma<A>::value&& all_single_precision<A>::value)>
value_t<A> sum(const A& a) {
    etl::dyn_vector<value_t<A>> ones(etl::size(a));
    ones = 1.0f;

    const float* m_a = a.memory_start();
    const float* m_b = ones.memory_start();

    return cblas_sdot(etl::size(a), m_a, 1, m_b, 1);
}

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_if(all_dma<A>::value&& all_double_precision<A>::value)>
value_t<A> sum(const A& a) {
    etl::dyn_vector<value_t<A>> ones(etl::size(a));
    ones = 1.0;

    const double* m_a = a.memory_start();
    const double* m_b = ones.memory_start();

    return cblas_ddot(etl::size(a), m_a, 1, m_b, 1);
}

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_if(!all_dma<A>::value)>
value_t<A> sum(const A& /*a*/) {
    cpp_unreachable("BLAS not enabled/available");
    return 0.0;
}

#else

/*!
 * \copydoc sum
 */
template <typename A>
value_t<A> sum(const A& /*a*/) {
    cpp_unreachable("BLAS not enabled/available");
    return 0.0;
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace blas
} //end of namespace impl
} //end of namespace etl
