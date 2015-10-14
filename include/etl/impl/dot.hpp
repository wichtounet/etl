//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_BLAS_MODE
#include "cblas.h" //For ddot/sdot
#endif

#include "etl/config.hpp"
#include "etl/traits_lite.hpp"

/**
 * Implementations of the dot product:
 *    1. Simple implementation using expressions
 *    2. Implementations using BLAS SDOT and DDOT
 */

namespace etl {

namespace detail {

template <typename A, typename B, typename Enable = void>
struct dot_impl {
    static auto apply(const A& a, const B& b) {
        return sum(scale(a, b));
    }
};

#ifdef ETL_BLAS_MODE

template <typename A, typename B>
struct dot_impl<A, B, std::enable_if_t<all_single_precision<A, B>::value && all_dma<A, B>::value>> {
    static float apply(const A& a, const B& b) {
        const float* m_a = a.memory_start();
        const float* m_b = b.memory_start();

        return cblas_sdot(etl::size(a), m_a, 1, m_b, 1);
    }
};

template <typename A, typename B>
struct dot_impl<A, B, std::enable_if_t<all_double_precision<A, B>::value && all_dma<A, B>::value>> {
    static double apply(const A& a, const B& b) {
        const double* m_a = a.memory_start();
        const double* m_b = b.memory_start();

        return cblas_ddot(etl::size(a), m_a, 1, m_b, 1);
    }
};

#endif

} //end of namespace detail

} //end of namespace etl
