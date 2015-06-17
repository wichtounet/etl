//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file mul_expression_builder.hpp
 * \brief Contains all the operators and functions to build multiplication expressions.
*/

#pragma once

#include "config.hpp"
#include "expression_helpers.hpp"

namespace etl {

template<typename A, typename B,
    cpp_enable_if(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && !is_element_wise_mul_default)>
auto operator*(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

template<typename A, typename B, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2,
    !is_element_wise_mul_default
> = cpp::detail::dummy>
auto operator*(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, vm_mul_expr> {
    return {a, b};
}

template<typename A, typename B, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1,
    !is_element_wise_mul_default
> = cpp::detail::dummy>
auto operator*(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, mv_mul_expr> {
    return {a, b};
}

template<typename A, typename B, cpp_enable_if(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2)>
auto mul(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

template<typename A, typename B, typename C, cpp_enable_if(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2)>
auto mul(A&& a, B&& b, C&& c) -> detail::forced_temporary_binary_helper<A, B, C, mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b, c};
}

template<typename A, typename B>
auto lazy_mul(A&& a, B&& b) -> detail::stable_transform_binary_helper<A, B, mm_mul_transformer> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return detail::stable_transform_binary_helper<A, B, mm_mul_transformer>{mm_mul_transformer<detail::build_type<A>, detail::build_type<B>>(a, b)};
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2
> = cpp::detail::dummy>
auto mul(A&& a, B&& b, C& c) -> detail::forced_temporary_binary_helper<A, B, C, vm_mul_expr> {
    return {a, b, c};
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1
> = cpp::detail::dummy>
auto mul(A&& a, B&& b, C& c) -> detail::forced_temporary_binary_helper<A, B, C, mv_mul_expr> {
    return {a, b, c};
}

template<typename A, typename B, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2
> = cpp::detail::dummy>
auto mul(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, vm_mul_expr> {
    return {a, b};
}

template<typename A, typename B, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1
> = cpp::detail::dummy>
auto mul(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, mv_mul_expr> {
    return {a, b};
}

template<typename A, typename B>
auto strassen_mul(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, strassen_mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

template<typename A, typename B, typename C>
auto strassen_mul(A&& a, B&& b, C&& c) -> detail::forced_temporary_binary_helper<A, B, C, strassen_mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b, c};
}

template<typename A, typename B>
auto outer(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, outer_product_expr> {
    return {std::forward<A>(a), std::forward<B>(b)};
}

template<typename A, typename B, typename C>
auto outer(A&& a, B&& b, C&& c) -> detail::forced_temporary_binary_helper<A, B, C, outer_product_expr> {
    return {std::forward<A>(a), std::forward<B>(b), std::forward<C>(c)};
}

} //end of namespace etl
