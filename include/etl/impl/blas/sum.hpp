//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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
template <typename A, cpp_enable_iff(is_dma<A>&& is_single_precision<A>)>
value_t<A> sum(const A& a) {
    etl::dyn_vector<value_t<A>> ones(etl::size(a));
    ones = 1.0f;

    a.ensure_cpu_up_to_date();

    const float* m_a = a.memory_start();
    const float* m_b = ones.memory_start();

    return cblas_sdot(etl::size(a), m_a, 1, m_b, 1);
}

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_iff(is_dma<A>&& is_double_precision<A>)>
value_t<A> sum(const A& a) {
    etl::dyn_vector<value_t<A>> ones(etl::size(a));
    ones = 1.0;

    a.ensure_cpu_up_to_date();

    const double* m_a = a.memory_start();
    const double* m_b = ones.memory_start();

    return cblas_ddot(etl::size(a), m_a, 1, m_b, 1);
}

/*!
 * \brief Compute the asum of a
 * \param a The lhs expression
 * \return the asum
 */
template <typename A, cpp_enable_iff(is_dma<A>&& is_single_precision<A>)>
value_t<A> asum(const A& a) {
    a.ensure_cpu_up_to_date();
    return cblas_sasum(etl::size(a), a.memory_start(), 1);
}

/*!
 * \copydoc asum
 */
template <typename A, cpp_enable_iff(is_dma<A>&& is_double_precision<A>)>
value_t<A> asum(const A& a) {
    a.ensure_cpu_up_to_date();
    return cblas_dasum(etl::size(a), a.memory_start(), 1);
}

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \copydoc sum
 */
template <typename A, cpp_enable_iff(!is_dma<A> || !is_floating<A>)>
value_t<A> sum(const A& /*a*/) {
    cpp_unreachable("BLAS not enabled/available");
    return 0.0;
}

/*!
 * \copydoc asum
 */
template <typename A, cpp_enable_iff(!is_dma<A> || !is_floating<A>)>
value_t<A> asum(const A& /*a*/) {
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

/*!
 * \copydoc asum
 */
template <typename A>
value_t<A> asum(const A& /*a*/) {
    cpp_unreachable("BLAS not enabled/available");
    return 0.0;
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace blas
} //end of namespace impl
} //end of namespace etl
