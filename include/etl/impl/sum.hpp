//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains implementations of sum reduction
 *
 * Implementations of the sum reduction:
 *    1. Simple implementation using expressions
 */

#pragma once

#include "etl/config.hpp"
#include "etl/traits_lite.hpp"

namespace etl {

namespace detail {

template <typename E, typename Enable = void>
struct sum_impl {
    static auto apply(const E& e) {
        value_t<E> acc(0);

        for (std::size_t i = 0; i < size(e); ++i) {
            acc += e[i];
        }

        return acc;
    }
};

} //end of namespace detail

} //end of namespace etl
