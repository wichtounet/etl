//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_COMPARABLE_HPP
#define ETL_COMPARABLE_HPP

/*
 * Use CRTP technique to inject comparison operators to expressions and value classes.
 */

namespace etl {

template<typename D>
struct comparable {
    using derived_t = D;

    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    template<typename E>
    bool operator==(const E& rhs){
        if(etl::size(as_derived()) != etl::size(rhs)){
            return false;
        }

        //TODO DO a deep comparison of dimensions

        return std::equal(as_derived().begin(), as_derived().end(), rhs.begin());
    }

    template<typename E>
    bool operator!=(const E& rhs){
        return !(as_derived() == rhs);
    }
};

} //end of namespace etl

#endif
