//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_STOP_HPP
#define ETL_STOP_HPP

#include "tmp.hpp"

namespace etl {

template<typename T, enable_if_u<
    and_u<
        is_etl_expr<T>::value,
        not_u<etl_traits<T>::is_value>::value,
        not_u<etl_traits<T>::is_fast>::value
    >::value> = detail::dummy>
auto s(const T& value){
    //Sizes will be directly propagated
    return dyn_matrix<typename T::value_type>(value);
}

template<typename T, enable_if_u<
    and_u<
        is_etl_expr<T>::value,
        not_u<etl_traits<T>::is_value>::value,
        etl_traits<T>::is_fast,
        etl_traits<T>::is_vector
    >::value> = detail::dummy>
auto s(const T& value){
    return fast_matrix<typename T::value_type, etl_traits<T>::size()>(value);
}

template<typename T, enable_if_u<
    and_u<
        is_etl_expr<T>::value,
        not_u<etl_traits<T>::is_value>::value,
        etl_traits<T>::is_fast,
        etl_traits<T>::is_matrix
    >::value> = detail::dummy>
auto s(const T& value){
    return fast_matrix<typename T::value_type, etl_traits<T>::template dim<0>(), etl_traits<T>::template dim<1>()>(value);
}

} //end of namespace etl

#endif
