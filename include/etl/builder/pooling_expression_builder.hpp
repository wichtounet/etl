//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file pooling_expression_builder.hpp
 * \brief Contains all the pooling operators and functions to build expressions.
*/

#pragma once

#include "etl/expression_helpers.hpp"

namespace etl {

template <std::size_t C1, std::size_t C2, typename E>
auto max_pool_2d(E&& value) -> temporary_unary_expr<value_t<E>, detail::build_type<E>, max_pool_2d_expr<value_t<E>, C1, C2>, void> {
    return temporary_unary_expr<value_t<E>, detail::build_type<E>, max_pool_2d_expr<value_t<E>, C1, C2>, void>{value};
}

template <std::size_t C1, std::size_t C2, typename E>
auto avg_pool_2d(E&& value) -> temporary_unary_expr<value_t<E>, detail::build_type<E>, avg_pool_2d_expr<value_t<E>, C1, C2>, void> {
    return temporary_unary_expr<value_t<E>, detail::build_type<E>, avg_pool_2d_expr<value_t<E>, C1, C2>, void>{value};
}

template <std::size_t C1, std::size_t C2, std::size_t C3, typename E>
auto max_pool_3d(E&& value) -> temporary_unary_expr<value_t<E>, detail::build_type<E>, max_pool_3d_expr<value_t<E>, C1, C2, C3>, void> {
    return temporary_unary_expr<value_t<E>, detail::build_type<E>, max_pool_3d_expr<value_t<E>, C1, C2, C3>, void>{value};
}

template <std::size_t C1, std::size_t C2, std::size_t C3, typename E>
auto avg_pool_3d(E&& value) -> temporary_unary_expr<value_t<E>, detail::build_type<E>, avg_pool_3d_expr<value_t<E>, C1, C2, C3>, void> {
    return temporary_unary_expr<value_t<E>, detail::build_type<E>, avg_pool_3d_expr<value_t<E>, C1, C2, C3>, void>{value};
}

template <std::size_t C1, std::size_t C2, typename E, typename F>
auto avg_pool_derivative_2d(E&& /*input*/, F&& /*output*/) {
    return 1.0 / (C1 * C2);
}

template <std::size_t C1, std::size_t C2, std::size_t C3, typename E, typename F>
auto avg_pool_derivative_3d(E&& /*input*/, F&& /*output*/) {
    return 1.0 / (C1 * C2 * C3);
}

template <std::size_t C1, std::size_t C2, typename E, typename F>
auto max_pool_derivative_2d(E&& in, F&& out) -> temporary_binary_expr<value_t<E>, detail::build_type<E>, detail::build_type<F>, max_pool_derivative_2d_expr<value_t<E>, C1, C2>, void> {
    return temporary_binary_expr<value_t<E>, detail::build_type<E>, detail::build_type<F>, max_pool_derivative_2d_expr<value_t<E>, C1, C2>, void>{in, out};
}

template <std::size_t C1, std::size_t C2, std::size_t C3, typename E, typename F>
auto max_pool_derivative_3d(E&& in, F&& out) -> temporary_binary_expr<value_t<E>, detail::build_type<E>, detail::build_type<F>, max_pool_derivative_3d_expr<value_t<E>, C1, C2, C3>, void> {
    return temporary_binary_expr<value_t<E>, detail::build_type<E>, detail::build_type<F>, max_pool_derivative_3d_expr<value_t<E>, C1, C2, C3>, void>{in, out};
}

template <std::size_t C1, std::size_t C2, typename E>
auto upsample_2d(E&& value) -> temporary_unary_expr<value_t<E>, detail::build_type<E>, upsample_2d_expr<value_t<E>, C1, C2>, void> {
    return temporary_unary_expr<value_t<E>, detail::build_type<E>, upsample_2d_expr<value_t<E>, C1, C2>, void>{value};
}

template <std::size_t C1, std::size_t C2, std::size_t C3, typename E>
auto upsample_3d(E&& value) -> temporary_unary_expr<value_t<E>, detail::build_type<E>, upsample_3d_expr<value_t<E>, C1, C2, C3>, void> {
    return temporary_unary_expr<value_t<E>, detail::build_type<E>, upsample_3d_expr<value_t<E>, C1, C2, C3>, void>{value};
}

template <std::size_t C1, std::size_t C2, typename E, cpp_enable_if(is_etl_expr<E>::value)>
auto p_max_pool_h(E&& value) -> unary_expr<value_t<E>, p_max_pool_h_transformer<detail::build_type<E>, C1, C2>, transform_op> {
    validate_pmax_pooling<C1, C2>(value);
    return unary_expr<value_t<E>, p_max_pool_h_transformer<detail::build_type<E>, C1, C2>, transform_op>{p_max_pool_h_transformer<detail::build_type<E>, C1, C2>(value)};
}

template <std::size_t C1, std::size_t C2, typename E, cpp_enable_if(is_etl_expr<E>::value)>
auto p_max_pool_p(E&& value) -> unary_expr<value_t<E>, p_max_pool_p_transformer<detail::build_type<E>, C1, C2>, transform_op> {
    validate_pmax_pooling<C1, C2>(value);
    return unary_expr<value_t<E>, p_max_pool_p_transformer<detail::build_type<E>, C1, C2>, transform_op>{p_max_pool_p_transformer<detail::build_type<E>, C1, C2>(value)};
}

} //end of namespace etl
