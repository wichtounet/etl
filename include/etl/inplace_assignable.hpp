//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_INPLACE_ASSIGNABLE_HPP
#define ETL_INPLACE_ASSIGNABLE_HPP

/*
 * Use CRTP technique to inject inplace operations into expressions and value classes. 
 */

namespace etl {

template<typename D>
struct inplace_assignable {
    using derived_t = D;

    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    template<typename E>
    derived_t& scale_inplace(E&& e){
        as_derived() *= e;

        return as_derived();
    }

    derived_t& fflip_inplace(){
        static_assert(etl_traits<derived_t>::dimensions() <= 2, "Impossible to fflip a matrix of D > 2");

        if(etl_traits<derived_t>::dimensions() == 2){
            std::reverse(as_derived().begin(), as_derived().end());
        }

        return as_derived();
    }
};

} //end of namespace etl

#endif
