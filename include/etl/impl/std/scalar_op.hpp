//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the scalar operations
 */

#pragma once

namespace etl {

namespace impl {

namespace standard {

template <typename TT>
void scalar_add(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] += rhs;
    }
}

template <typename TT>
void scalar_sub(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] -= rhs;
    }
}

template <typename TT>
void scalar_mul(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] *= rhs;
    }
}

template <typename TT>
void scalar_div(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] /= rhs;
    }
}

template <typename TT>
void scalar_mod(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] %= rhs;
    }
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
