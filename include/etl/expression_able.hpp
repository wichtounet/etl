//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_EXPRESSION_ABLE_HPP
#define ETL_EXPRESSION_ABLE_HPP

#include "compat.hpp"

/*
 * Use CRTP technique to inject functions that return expressions 
 * from this. 
 */

namespace etl {

template<typename D>
struct expression_able {
    using derived_t = D;

    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    template<typename E>
    auto scale(E&& e) -> decltype(etl::scale(as_derived(), std::forward<E>(e))) {
        return etl::scale(as_derived(), std::forward<E>(e));
    }

    ETL_DEBUG_AUTO_TRICK auto fflip(){
        return etl::fflip(as_derived());
    }

    ETL_DEBUG_AUTO_TRICK auto hflip(){
        return etl::hflip(as_derived());
    }

    ETL_DEBUG_AUTO_TRICK auto vflip(){
        return etl::vflip(as_derived());
    }

    ETL_DEBUG_AUTO_TRICK auto transpose(){
        return etl::transpose(as_derived());
    }
};

} //end of namespace etl

#endif
