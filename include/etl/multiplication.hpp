//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_MULTIPLICATION_HPP
#define ETL_MULTIPLICATION_HPP

#include <algorithm>

namespace etl {

namespace detail {

template<typename A, typename B, typename C, cpp::disable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_mmul_sizes(const A& a, const B& b, C& c){
    cpp_assert(
            dim(a,1) == dim(b,0)          //interior dimensions
        &&  dim(a,0) == dim(c,0)          //exterior dimension 1
        &&  dim(b,1) == dim(c,1),         //exterior dimension 2
        "Invalid sizes for multiplication");
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_mmul_sizes(const A&, const B&, C&){
    static_assert(
            etl_traits<A>::template dim<1>() == etl_traits<B>::template dim<0>()          //interior dimensions
        &&  etl_traits<A>::template dim<0>() == etl_traits<C>::template dim<0>()          //exterior dimension 1
        &&  etl_traits<B>::template dim<1>() == etl_traits<C>::template dim<1>(),         //exterior dimension 2
        "Invalid sizes for multiplication");
}

} //end of namespace detail

template<typename A, typename B, typename C>
static C& mmul(const A& a, const B& b, C& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(etl_traits<A>::dimensions() == 2 && etl_traits<B>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");
    detail::check_mmul_sizes(a,b,c);

    c = 0;

    for(std::size_t i = 0; i < rows(a); i++){
        for(std::size_t j = 0; j < columns(b); j++){
            for(std::size_t k = 0; k < columns(a); k++){
                c(i,j) += a(i,k) * b(k,j);
            }
        }
    }

    return c;
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast,
    etl_traits<A>::dimensions() == 1, etl_traits<B>::dimensions() == 2
> = cpp::detail::dummy>
static C& auto_vmmul(const A& a, const B& b, C& c){
    return mmul(reshape<1, etl_traits<B>::template dim<0>()>(a), b, c);
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    etl_traits<A>::is_fast, etl_traits<B>::is_fast, etl_traits<C>::is_fast,
    etl_traits<A>::dimensions() == 2, etl_traits<B>::dimensions() == 1
> = cpp::detail::dummy>
static C& auto_vmmul(const A& a, const B& b, C& c){
    return mmul(a, reshape<etl_traits<A>::template dim<1>(),1>(b), c);
}

} //end of namespace etl

#endif
