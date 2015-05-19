//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_OUTER_PRODUCT_EXPR_HPP
#define ETL_OUTER_PRODUCT_EXPR_HPP

#include <algorithm>

#include "impl/outer_product.hpp"

#include "traits_lite.hpp"
#include "temporary.hpp"

namespace etl {

template<typename T>
struct outer_product_expr {
    template<typename A, typename B, class Enable = void>
    struct result_type_builder {
        using type = dyn_matrix<value_t<A>>;
    };

    template<typename A, typename B>
    struct result_type_builder<A, B, std::enable_if_t<all_fast<A,B>::value>> {
        using type = fast_dyn_matrix<value_t<A>, decay_traits<A>::template dim<0>(), decay_traits<B>::template dim<0>()>;
    };

    template<typename A, typename B>
    using result_type = typename result_type_builder<A, B>::type;

    template<typename A, typename B, cpp_enable_if(all_fast<A,B>::value)>
    static result_type<A,B>* allocate(A&& /*a*/, B&& /*b*/){
        return new result_type<A, B>();
    }

    template<typename A, typename B, cpp_disable_if(all_fast<A,B>::value)>
    static result_type<A,B>* allocate(A&& a, B&& b){
        return new result_type<A, B>(etl::dim<0>(a), etl::dim<0>(b));
    }

    template<typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c){
        static_assert(all_etl_expr<A,B,C>::value, "Outer product only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 1 && decay_traits<B>::dimensions() == 1 && decay_traits<C>::dimensions() == 2, "Invalid dimensions for outer product");

        detail::outer_product_impl<A, B, C>::apply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }

    static std::string desc() noexcept {
        return "outer_product";
    }

    template<typename A, typename B>
    static std::size_t size(const A& a, const B& b){
        return etl::dim<0>(a) * etl::dim<0>(b);
    }

    template<typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d){
        return d == 0 ? etl::dim<0>(a) : etl::dim<0>(b);
    }

    template<typename A, typename B>
    static constexpr std::size_t size(){
        return etl_traits<A>::template dim<0>() * etl_traits<B>::template dim<0>();
    }

    template<typename A, typename B, std::size_t D>
    static constexpr std::size_t dim(){
        return D == 0 ? etl_traits<A>::template dim<0>() : etl_traits<B>::template dim<0>();
    }

    static constexpr std::size_t dimensions(){
        return 2;
    }
};

} //end of namespace etl

#endif
