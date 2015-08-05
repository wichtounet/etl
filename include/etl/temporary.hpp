//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/config.hpp"       //configuration of ETL
#include "etl/traits_lite.hpp"  //type traits
#include "etl/stop.hpp"

namespace etl {

template<typename E, typename Sequence>
struct build_fast_dyn_matrix_type;

template<typename E, std::size_t... I>
struct build_fast_dyn_matrix_type<E, std::index_sequence<I...>> {
    using type = fast_matrix_impl<value_t<E>, std::vector<value_t<E>>, decay_traits<E>::storage_order, decay_traits<E>::template dim<I>()...>;
};

template<typename E, cpp_enable_if(decay_traits<E>::is_fast)>
decltype(auto) force_temporary(E&& expr){
    return typename build_fast_dyn_matrix_type<E, std::make_index_sequence<decay_traits<E>::dimensions()>>::type{std::forward<E>(expr)};
}

template<typename E, cpp_disable_if(decay_traits<E>::is_fast)>
decltype(auto) force_temporary(E&& expr){
    //Sizes will be directly propagated
    return dyn_matrix_impl<value_t<E>, decay_traits<E>::storage_order, decay_traits<E>::dimensions()>{std::forward<E>(expr)};
}

template<typename E>
decltype(auto) force_temporary_dyn(E&& expr){
    //Sizes will be directly propagated
    return dyn_matrix_impl<value_t<E>, decay_traits<E>::storage_order, decay_traits<E>::dimensions()>{std::forward<E>(expr)};
}

template<typename E, cpp_enable_if(has_direct_access<E>::value || !create_temporary)>
decltype(auto) make_temporary(E&& expr){
    return std::forward<E>(expr);
}

template<typename E, cpp_enable_if(!has_direct_access<E>::value && create_temporary)>
decltype(auto) make_temporary(E&& expr){
    return force_temporary(std::forward<E>(expr));
}

} //end of namespace etl
