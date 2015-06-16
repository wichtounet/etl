#pragma once
//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <algorithm>

/*
 * Use CRTP technique to inject functions that test the dimensions.
 */

namespace etl {

template<typename D>
struct dim_testable {
    using derived_t = D;

    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    bool is_square() const noexcept {
        cpp_assert(decay_traits<derived_t>::dimensions() == 2, "Only 2D matrix can be square or rectangular");
        return etl::dim<0>(as_derived()) == etl::dim<1>(as_derived());
    }

    bool is_rectangular() const noexcept {
        return !is_square();
    }

    bool is_sub_square() const noexcept {
        cpp_assert(decay_traits<derived_t>::dimensions() == 3, "Only 2D matrix can be sub square or sub rectangular");
        return etl::dim<1>(as_derived()) == etl::dim<2>(as_derived());
    }

    bool is_sub_rectangular() const noexcept {
        return !is_sub_square();
    }
};

} //end of namespace etl
