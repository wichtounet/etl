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
template <typename A>
value_t<A> sum(const A& a) {
    if constexpr (is_dma<A>) {
        etl::dyn_vector<value_t<A>> ones(etl::size(a));
        ones = value_t<A>(1);

        a.ensure_cpu_up_to_date();

        const auto* m_a = a.memory_start();
        const auto* m_b = ones.memory_start();

        if constexpr (is_single_precision<A>) {
            return cblas_sdot(etl::size(a), m_a, 1, m_b, 1);
        } else if constexpr (is_double_precision<A>) {
            return cblas_ddot(etl::size(a), m_a, 1, m_b, 1);
        }
    } else {
        cpp_unreachable("BLAS not enabled/available");
    }
}

/*!
 * \brief Compute the asum of a
 * \param a The lhs expression
 * \return the asum
 */
template <typename A>
value_t<A> asum(const A& a) {
    if constexpr (is_dma<A>) {
        a.ensure_cpu_up_to_date();

        if constexpr (is_single_precision<A>) {
            return cblas_sasum(etl::size(a), a.memory_start(), 1);
        } else if constexpr (is_double_precision<A>) {
            return cblas_dasum(etl::size(a), a.memory_start(), 1);
        } else {
            cpp_unreachable("BLAS not enabled/available");
        }
    } else {
        cpp_unreachable("BLAS not enabled/available");
    }
}

#else

//COVERAGE_EXCLUDE_BEGIN

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
