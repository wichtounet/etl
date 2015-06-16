#pragma once
//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <algorithm>

/*
 * Use CRTP technique to inject functions that test the values of
 * the expressions or the value classes.
 */

namespace etl {

template<typename D>
struct value_testable {
    using derived_t = D;

    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    bool is_finite() const noexcept {
        return std::all_of(as_derived().begin(), as_derived().end(), static_cast<bool(*)(value_t<derived_t>)>(std::isfinite));
    }

    bool is_zero() const noexcept {
        return std::all_of(as_derived().begin(), as_derived().end(), [](value_t<derived_t> v){ return v == value_t<derived_t>(0); });;
    }
};

} //end of namespace etl
