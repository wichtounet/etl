//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains all the operators and functions to build expressions.
 */

#pragma once

//Include implementations
#include "etl/impl/dot.hpp"
#include "etl/impl/sum.hpp"
#include "etl/impl/norm.hpp"

#include "etl/builder/binary_expression_builder.hpp"
#include "etl/builder/wrapper_expression_builder.hpp"
#include "etl/builder/view_expression_builder.hpp"

namespace etl {

// Apply an unary expression on an ETL expression (vector,matrix,binary,unary)

/*!
 * \Apply Unary minus on the expression
 * \param value The expression on which to apply the operator
 * \return an expression representing the unary minus of the given expression
 */
template <typename E>
auto operator-(E&& value) -> detail::unary_helper<E, minus_unary_op> {
    return detail::unary_helper<E, minus_unary_op>{value};
}

/*!
 * \Apply Unary plus on the expression
 * \param value The expression on which to apply the operator
 * \return an expression representing the unary plus of the given expression
 */
template <typename E>
auto operator+(E&& value) -> detail::unary_helper<E, plus_unary_op> {
    return detail::unary_helper<E, plus_unary_op>{value};
}

/*!
 * \brief Apply absolute on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the absolute of each value of the given expression
 */
template <typename E>
auto abs(E&& value) -> detail::unary_helper<E, abs_unary_op> {
    static_assert(is_etl_expr<E>, "etl::abs can only be used on ETL expressions");
    return detail::unary_helper<E, abs_unary_op>{value};
}

/*!
 * \brief Create an expression with the max value of lhs or rhs
 * \param lhs The left hand side ETL expression
 * \param rhs The right hand side ETL expression
 * \return an expression representing the max values from lhs and rhs
 */
template <typename L, typename R>
auto max(L&& lhs, R&& rhs) -> detail::left_binary_helper_op_scalar<L, R, max_binary_op<detail::wrap_scalar_value_t<L>, detail::wrap_scalar_value_t<L>>> {
    static_assert(is_etl_expr<L>, "etl::max can only be used on ETL expressions");
    return {detail::wrap_scalar(lhs), detail::smart_wrap_scalar<L>(rhs)};
}

/*!
 * \brief Create an expression with the min value of lhs or rhs
 * \param lhs The left hand side ETL expression
 * \param rhs The right hand side ETL expression
 * \return an expression representing the min values from lhs and rhs
 */
template <typename L, typename R>
auto min(L&& lhs, R&& rhs) -> detail::left_binary_helper_op_scalar<L, R, min_binary_op<detail::wrap_scalar_value_t<L>, detail::wrap_scalar_value_t<L>>> {
    static_assert(is_etl_expr<L>, "etl::max can only be used on ETL expressions");
    return {detail::wrap_scalar(lhs), detail::smart_wrap_scalar<L>(rhs)};
}

/*!
 * \brief Round down each values of the ETL expression
 * \param value The ETL expression
 * \return an expression representing the values of the ETL expression rounded down.
 */
template <typename E>
auto floor(E&& value) {
    static_assert(is_etl_expr<E>, "etl::floor can only be used on ETL expressions");
    return detail::unary_helper<E, floor_unary_op>{value};
}

/*!
 * \brief Round up each values of the ETL expression
 * \param value The ETL expression
 * \return an expression representing the values of the ETL expression rounded up.
 */
template <typename E>
auto ceil(E&& value) {
    static_assert(is_etl_expr<E>, "etl::ceil can only be used on ETL expressions");
    return detail::unary_helper<E, ceil_unary_op>{value};
}

/*!
 * \brief Clip each values of the ETL expression between min and max
 * \param value The ETL expression
 * \param min The minimum
 * \param max The maximum
 * \return an expression representing the values of the ETL expression clipped between min and max
 */
template <typename E, typename T>
auto clip(E&& value, T min, T max) {
    static_assert(is_etl_expr<E>, "etl::clip can only be used on ETL expressions");
    static_assert(std::is_arithmetic_v<T>, "etl::clip can only be used with arithmetic values");
    return detail::make_stateful_unary_expr<E, clip_scalar_op<value_t<E>, value_t<E>>>(value, value_t<E>(min), value_t<E>(max));
}

/*!
 * \brief Apply pow(x, v) on each element x of the ETL expression.
 *
 * This function is not guaranteed to return the same results in
 * different operation modes (CPU, GPU, VEC). It should only be used
 * with positives x.
 *
 * \param value The ETL expression
 * \param v The power
 *
 * \return an expression representing the pow(x, v) of each value x of the given expression
 */
template <typename E, typename T>
auto pow(E&& value, T v) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, pow_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>, "etl::pow can only be used on ETL expressions");
    static_assert(std::is_arithmetic_v<T>, "etl::pow can only be used with arithmetic values");
    return {value, scalar<value_t<E>>(v)};
}

/*!
 * \brief Apply pow(x, v) on each element x of the ETL expression
 * \param value The ETL expression
 * \param v The power
 * \return an expression representing the pow(x, v) of each value x of the given expression
 */
template <typename E>
auto pow_int(E&& value, size_t v) -> detail::left_binary_helper_op<E, scalar<size_t>, integer_pow_binary_op<value_t<E>, size_t>> {
    static_assert(is_etl_expr<E>, "etl::pow can only be used on ETL expressions");
    return {value, scalar<size_t>(v)};
}

/*!
 * \brief Apply pow(x, v) on each element x of the ETL expression.
 *
 * This function does not have different precision in different
 * operation mode (GPU, VEC, ...). This is guaranteeed to work with
 * the same precision as std::pow.
 *
 * \param value The ETL expression
 * \param v The power
 *
 * \return an expression representing the pow(x, v) of each value x of the given expression
 */
template <typename E, typename T>
auto pow_precise(E&& value, T v) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, precise_pow_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>, "etl::pow_precise can only be used on ETL expressions");
    static_assert(std::is_arithmetic_v<T>, "etl::pow_precise can only be used with arithmetic values");
    return {value, scalar<value_t<E>>(v)};
}

/*!
 * \brief Creates an expression with values of 1 where the ETL expression has a value of v
 * \param value The ETL expression
 * \param v The value to test
 * \return an expression representing the values of 1 where the ETL expression has a value of v
 */
template <typename E, typename T>
auto one_if(E&& value, T v) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, one_if_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>, "etl::one_if can only be used on ETL expressions");
    static_assert(std::is_arithmetic_v<T>, "etl::one_if can only be used with arithmetic values");
    return {value, scalar<value_t<E>>(v)};
}

/*!
 * \brief Creates an expression with a value of 1 where the max value is and all zeroes other places
 * \param value The ETL expression
 * \return an expression with a value of 1 where the max value is and all zeroes other places
 */
template <typename E>
auto one_if_max(E&& value) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, one_if_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>, "etl::one_if_max can only be used on ETL expressions");
    return {value, scalar<value_t<E>>(max(value))};
}

/*!
 * \brief Extract the real part of each complex value of the given expression
 * \param value The ETL expression
 * \return an expression representing the real part of each complex of the given expression
 */
template <typename E>
auto real(E&& value) -> unary_expr<typename value_t<E>::value_type, detail::build_type<E>, real_unary_op<value_t<E>>> {
    static_assert(is_etl_expr<E>, "etl::real can only be used on ETL expressions");
    static_assert(is_complex_t<value_t<E>>, "etl::real can only be used on ETL expressions containing complex numbers");
    return unary_expr<typename value_t<E>::value_type, detail::build_type<E>, real_unary_op<value_t<E>>>{value};
}

/*!
 * \brief Extract the imag part of each complex value of the given expression
 * \param value The ETL expression
 * \return an expression representing the imag part of each complex of the given expression
 */
template <typename E>
auto imag(E&& value) -> unary_expr<typename value_t<E>::value_type, detail::build_type<E>, imag_unary_op<value_t<E>>> {
    static_assert(is_etl_expr<E>, "etl::imag can only be used on ETL expressions");
    static_assert(is_complex_t<value_t<E>>, "etl::imag can only be used on ETL expressions containing complex numbers");
    return unary_expr<typename value_t<E>::value_type, detail::build_type<E>, imag_unary_op<value_t<E>>>{value};
}

/*!
 * \brief Apply the conjugate operation on each complex value of the given expression
 * \param value The ETL expression
 * \return an expression representing the the conjugate operation of each complex of the given expression
 */
template <typename E>
auto conj(E&& value) -> unary_expr<value_t<E>, detail::build_type<E>, conj_unary_op<value_t<E>>> {
    static_assert(is_etl_expr<E>, "etl::conj can only be used on ETL expressions");
    static_assert(is_complex_t<value_t<E>>, "etl::conj can only be used on ETL expressions containing complex numbers");
    return unary_expr<value_t<E>, detail::build_type<E>, conj_unary_op<value_t<E>>>{value};
}

/*!
 * \brief Add some uniform noise (0, 1.0) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E>
auto uniform_noise(E&& value) -> detail::unary_helper<E, uniform_noise_unary_op> {
    static_assert(is_etl_expr<E>, "etl::uniform_noise can only be used on ETL expressions");
    return detail::unary_helper<E, uniform_noise_unary_op>{value};
}

/*!
 * \brief Add some uniform noise (0, 1.0) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E, typename G>
auto uniform_noise(G& g, E&& value) {
    return detail::make_stateful_unary_expr<E, uniform_noise_unary_g_op<G, value_t<E>>>(value, g);
}

/*!
 * \brief Add some normal noise (0, 1.0) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E>
auto normal_noise(E&& value) -> detail::unary_helper<E, normal_noise_unary_op> {
    static_assert(is_etl_expr<E>, "etl::normal_noise can only be used on ETL expressions");
    return detail::unary_helper<E, normal_noise_unary_op>{value};
}

/*!
 * \brief Add some normal noise (0, 1.0) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E, typename G>
auto normal_noise(G& g, E&& value) {
    return detail::make_stateful_unary_expr<E, normal_noise_unary_g_op<G, value_t<E>>>(value, g);
}

/*!
 * \brief Add some normal noise (0, sigmoid(x)) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E>
auto logistic_noise(E&& value) -> detail::unary_helper<E, logistic_noise_unary_op> {
    static_assert(is_etl_expr<E>, "etl::logistic_noise can only be used on ETL expressions");
    return detail::unary_helper<E, logistic_noise_unary_op>{value};
}

/*!
 * \brief Add some normal noise (0, sigmoid(x)) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E, typename G>
auto logistic_noise(G& g, E&& value) {
    return detail::make_stateful_unary_expr<E, logistic_noise_unary_g_op<G, value_t<E>>>(value, g);
}

/*!
 * \brief Add some normal noise (0, sigmoid(x)) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E>
auto state_logistic_noise(E&& value) {
    static_assert(is_etl_expr<E>, "etl::logistic_noise can only be used on ETL expressions");
    return detail::make_stateful_unary_expr<E, state_logistic_noise_unary_op<value_t<E>>>(value);
}

/*!
 * \brief Add some normal noise (0, sigmoid(x)) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E>
auto state_logistic_noise(E&& value, const std::shared_ptr<void*> & states) {
    static_assert(is_etl_expr<E>, "etl::logistic_noise can only be used on ETL expressions");
    return detail::make_stateful_unary_expr<E, state_logistic_noise_unary_op<value_t<E>>>(value, states);
}

/*!
 * \brief Add some normal noise (0, sigmoid(x)) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename G, typename E, cpp_enable_iff(is_etl_expr<E>)>
auto state_logistic_noise(G& g, E&& value) {
    return detail::make_stateful_unary_expr<E, state_logistic_noise_unary_g_op<G, value_t<E>>>(value, g);
}

/*!
 * \brief Add some normal noise (0, sigmoid(x)) to the given expression
 * \param value The input ETL expression
 * \return an expression representing the input expression plus noise
 */
template <typename E, typename G>
auto state_logistic_noise(G& g, E&& value, const std::shared_ptr<void*> & states) {
    static_assert(is_etl_expr<E>, "etl::logistic_noise can only be used on ETL expressions");
    return detail::make_stateful_unary_expr<E, state_logistic_noise_unary_g_op<G, value_t<E>>>(value, g, states);
}

/*!
 * \brief Add some normal noise N(0,1) to x.
 * No noise is added to values equal to zero or to given the value.
 * \param value The value to add noise to
 * \param v The value for the upper range limit
 * \return An expression representing the left value plus the noise
 */
template <typename E, typename T>
auto ranged_noise(E&& value, T v) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, ranged_noise_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>, "etl::ranged_noise can only be used on ETL expressions");
    static_assert(std::is_arithmetic_v<T>, "etl::ranged_noise can only be used with arithmetic values");
    return {value, scalar<value_t<E>>(v)};
}

// Apply a stable transformation

/*!
 * \brief Repeats the expression to the right (adds dimension after existing)
 * \param value The expression to repeat
 * \tparam D1 The first repeat
 * \tparam D The remaining repeated dimensions
 * \return an expression representing the repeated expression
 */
template <size_t D1, size_t... D, typename E>
auto rep(E&& value) -> unary_expr<value_t<E>, rep_r_transformer<detail::build_type<E>, D1, D...>, transform_op> {
    static_assert(is_etl_expr<E>, "etl::rep can only be used on ETL expressions");
    return unary_expr<value_t<E>, rep_r_transformer<detail::build_type<E>, D1, D...>, transform_op>{rep_r_transformer<detail::build_type<E>, D1, D...>(value)};
}

/*!
 * \brief Repeats the expression to the right (adds dimension after existing)
 * \param value The expression to repeat
 * \tparam D1 The first repeat
 * \tparam D The remaining repeated dimensions
 * \return an expression representing the repeated expression
 */
template <size_t D1, size_t... D, typename E>
auto rep_r(E&& value) -> unary_expr<value_t<E>, rep_r_transformer<detail::build_type<E>, D1, D...>, transform_op> {
    static_assert(is_etl_expr<E>, "etl::rep_r can only be used on ETL expressions");
    return unary_expr<value_t<E>, rep_r_transformer<detail::build_type<E>, D1, D...>, transform_op>{rep_r_transformer<detail::build_type<E>, D1, D...>(value)};
}

/*!
 * \brief Repeats the expression to the left (adds dimension before existing)
 * \param value The expression to repeat
 * \tparam D1 The first repeat
 * \tparam D The remaining repeated dimensions
 * \return an expression representing the repeated expression
 */
template <size_t D1, size_t... D, typename E>
auto rep_l(E&& value) -> unary_expr<value_t<E>, rep_l_transformer<detail::build_type<E>, D1, D...>, transform_op> {
    static_assert(is_etl_expr<E>, "etl::rep_l can only be used on ETL expressions");
    return unary_expr<value_t<E>, rep_l_transformer<detail::build_type<E>, D1, D...>, transform_op>{rep_l_transformer<detail::build_type<E>, D1, D...>(value)};
}

/*!
 * \brief Repeats the expression to the right (adds dimension after existing)
 * \param value The expression to repeat
 * \param d1 The first repeat
 * \param d The remaining repeated dimensions
 * \return an expression representing the repeated expression
 */
template <typename... D, typename E>
auto rep(E&& value, size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op> {
    static_assert(is_etl_expr<E>, "etl::rep can only be used on ETL expressions");
    return unary_expr<value_t<E>, dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op>{
        dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<size_t>(d)...}})};
}

/*!
 * \brief Repeats the expression to the right (adds dimension after existing)
 * \param value The expression to repeat
 * \param d1 The first repeat
 * \param d The remaining repeated dimensions
 * \return an expression representing the repeated expression
 */
template <typename... D, typename E>
auto rep_r(E&& value, size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op> {
    static_assert(is_etl_expr<E>, "etl::rep_r can only be used on ETL expressions");
    return unary_expr<value_t<E>, dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op>{
        dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<size_t>(d)...}})};
}

/*!
 * \brief Repeats the expression to the left (adds dimension after existing)
 * \param value The expression to repeat
 * \param d1 The first repeat
 * \param d The remaining repeated dimensions
 * \return an expression representing the repeated expression
 */
template <typename... D, typename E>
auto rep_l(E&& value, size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_l_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op> {
    static_assert(is_etl_expr<E>, "etl::rep_l can only be used on ETL expressions");
    return unary_expr<value_t<E>, dyn_rep_l_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op>{
        dyn_rep_l_transformer<detail::build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<size_t>(d)...}})};
}

/*!
 * \brief Returns the indices of the maximum values in the first axis of the
 * given matrix. If passed a vector, returns the index of the maximum element.
 *
 * \param value The matrix or vector to aggregate
 *
 * \return an expression representing the aggregated expression
 */
template <typename E>
auto argmax(E&& value) {
    static_assert(is_etl_expr<E>, "etl::argmax can only be used on ETL expressions");

    if constexpr (decay_traits<E>::dimensions() > 1) {
        return detail::make_transform_expr<E, argmax_transformer>(value);
    } else {
        return max_index(value);
    }
}

/*!
 * \brief Returns the indices of the minimum values in the first axis of the
 * given matrix. If passed a vector, returns the index of the mimimum element.
 *
 * \param value The value to aggregate
 *
 * \return an expression representing the aggregated expression
 */
template <typename E>
auto argmin(E&& value) {
    static_assert(is_etl_expr<E>, "etl::argmin can only be used on ETL expressions");

    if constexpr (decay_traits<E>::dimensions() > 1) {
        return detail::make_transform_expr<E, argmin_transformer>(value);
    } else {
        return min_index(value);
    }
}

/*!
 * \brief Aggregate (sum) a dimension from the right. This effectively removes
 * the last dimension from the expression and sums its values to the left.
 * the last dimension from the expression and sums its values to the left.
 * \param value The value to aggregate
 * \return an expression representing the aggregated expression
 */
template <typename E>
auto sum_r(E&& value) -> detail::stable_transform_helper<E, sum_r_transformer> {
    static_assert(is_etl_expr<E>, "etl::sum_r can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() > 1, "Can only use sum_r on matrix");
    return detail::make_transform_expr<E, sum_r_transformer>(value);
}

/*!
 * \brief Aggregate (sum) a dimension from the left. This effectively removes
 * the first dimension from the expression and sums its values to the right.
 * \param value The value to aggregate
 * \return an expression representing the aggregated expression
 */
template <typename E>
auto sum_l(E&& value) -> detail::stable_transform_helper<E, sum_l_transformer> {
    static_assert(is_etl_expr<E>, "etl::sum_l can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() > 1, "Can only use sum_l on matrix");
    return detail::make_transform_expr<E, sum_l_transformer>(value);
}

/*!
 * \brief Aggregate (average) a dimension from the right. This effectively removes
 * the last dimension from the expression and averages its values to the left.
 * \param value The value to aggregate
 * \return an expression representing the aggregated expression
 */
template <typename E>
auto mean_r(E&& value) -> detail::stable_transform_helper<E, mean_r_transformer> {
    static_assert(is_etl_expr<E>, "etl::mean_r can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() > 1, "Can only use mean_r on matrix");
    return detail::make_transform_expr<E, mean_r_transformer>(value);
}

/*!
 * \brief Aggregate (average) a dimension from the left. This effectively removes
 * the first dimension from the expression and averages its values to the right.
 * \param value The value to aggregate
 * \return an expression representing the aggregated expression
 */
template <typename E>
auto mean_l(E&& value) -> detail::stable_transform_helper<E, mean_l_transformer> {
    static_assert(is_etl_expr<E>, "etl::mean_l can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() > 1, "Can only use mean_l on matrix");
    return detail::make_transform_expr<E, mean_l_transformer>(value);
}

/*!
 * \brief Return, for each original position, 1.0 if the value is the max of the
 * sub matrix, 0.0 otherwise.
 *
 * \param value The matrix to explore
 *
 * \return an expression representing the 1-if-max view for each sub view of
 * the input matrix
 */
template <typename E>
auto one_if_max_sub(const E& value) -> detail::stable_transform_helper<E, one_if_max_sub_transformer> {
    static_assert(is_etl_expr<E>, "etl::one_if_max_sub can only be used on ETL expressions");
    static_assert(is_2d<E>, "Can only use one_if_max_sub 2D matrix");
    return detail::make_transform_expr<E, one_if_max_sub_transformer>(value);
}

/*!
 * \brief Returns the horizontal flipping of the given expression.
 * \param value The expression
 * \return The horizontal flipping of the given expression.
 */
template <typename E>
auto hflip(const E& value) -> detail::stable_transform_helper<E, hflip_transformer> {
    static_assert(is_etl_expr<E>, "etl::hflip can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return detail::make_transform_expr<E, hflip_transformer>(value);
}

/*!
 * \brief Returns the vertical flipping of the given expression.
 * \param value The expression
 * \return The vertical flipping of the given expression.
 */
template <typename E>
auto vflip(const E& value) -> detail::stable_transform_helper<E, vflip_transformer> {
    static_assert(is_etl_expr<E>, "etl::vflip can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return detail::make_transform_expr<E, vflip_transformer>(value);
}

/*!
 * \brief Returns the horizontal and vertical flipping of the given expression.
 * \param value The expression
 * \return The horizontal and vertical flipping of the given expression.
 */
template <typename E>
auto fflip(const E& value) -> detail::stable_transform_helper<E, fflip_transformer> {
    static_assert(is_etl_expr<E>, "etl::fflip can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return detail::make_transform_expr<E, fflip_transformer>(value);
}

/*!
 * \brief Returns the transpose of the given expression.
 * \param value The expression
 * \return The transpose of the given expression.
 */
template <typename E>
auto transpose(const E& value) -> transpose_expr<detail::build_type<E>> {
    static_assert(is_etl_expr<E>, "etl::transpose can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() <= 2, "Transpose not defined for matrix > 2D");
    return transpose_expr<detail::build_type<E>>{value};
}

/*!
 * \brief Returns the transpose of the given expression.
 * \param value The expression
 * \return The transpose of the given expression.
 */
template <typename E>
auto trans(const E& value) {
    return transpose(value);
}

/*!
 * \brief Returns the conjugate transpose of the given expression.
 * \param value The expression
 * \return The conjugate transpose of the given expression.
 */
template <typename E>
auto conj_transpose(const E& value) {
    return conj(transpose(value));
}

/*!
 * \brief Returns the conjugate transpose of the given expression.
 * \param value The expression
 * \return The conjugate transpose of the given expression.
 */
template <typename E>
auto ctrans(const E& value) {
    return conj(transpose(value));
}

/*!
 * \brief Returns euclidean norm of the given expression.
 * \param a The expression
 * \return The euclidean norm of the expression
 */
template <typename A>
value_t<A> norm(const A& a) {
    return detail::norm_impl::apply(a);
}

// TODO: Ideally we would like to be able to use a.dot(b) and
// a.cross(b). Unfortunately, this would require one of these options:
// * A new CRTP class, increasing the compilation time overhead
// * Adding the dot template function in each expression

/*!
 * \brief Returns the dot product of the two given expressions.
 * \param a The left expression
 * \param b The right expression
 * \return The dot product of the two expressions
 */
template <typename A, typename B>
value_t<A> dot(const A& a, const B& b) {
    validate_expression(a, b);
    return detail::dot_impl::apply(a, b);
}

/*!
 * \brief Returns the dot product of the two given expressions.
 * \param a The left expression
 * \param b The right expression
 * \return The dot product of the two expressions
 */
template <typename A, typename B>
etl::fast_vector<value_t<A>, 3> cross(const A& a, const B& b) {
    static_assert(etl::is_1d<A>, "Cross product is only valid for 1D vectors");
    static_assert(etl::is_1d<B>, "Cross product is only valid for 1D vectors");

    if constexpr (all_fast<A, B>) {
        static_assert(etl::decay_traits<A>::size() == 3, "Cross product is only valid for 1D vectors of size 3");
        static_assert(etl::decay_traits<B>::size() == 3, "Cross product is only valid for 1D vectors of size 3");
    } else {
        cpp_assert(etl::decay_traits<A>::size(a) == 3, "Cross product is only valid for 1D vectors of size 3");
        cpp_assert(etl::decay_traits<B>::size(b) == 3, "Cross product is only valid for 1D vectors of size 3");
    }

    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

/*!
 * \brief Returns the sum of all the values contained in the given expression
 * \param values The expression to reduce
 * \return The sum of the values of the expression
 */
template <typename E>
value_t<E> sum(E&& values) {
    static_assert(is_etl_expr<E>, "etl::sum can only be used on ETL expressions");

    //Reduction force evaluation
    force(values);

    return detail::sum_impl::apply(values);
}

/*!
 * \brief Returns the sum of all the absolute values contained in the given expression
 * \param values The expression to reduce
 * \return The sum of the absolute values of the expression
 */
template <typename E>
value_t<E> asum(E&& values) {
    static_assert(is_etl_expr<E>, "etl::asum can only be used on ETL expressions");

    //Reduction force evaluation
    force(values);

    return detail::asum_impl::apply(values);
}

/*!
 * \brief Returns the mean of all the values contained in the given expression
 * \param values The expression to reduce
 * \return The mean of the values of the expression
 */
template <typename E>
value_t<E> mean(E&& values) {
    static_assert(is_etl_expr<E>, "etl::mean can only be used on ETL expressions");

    return sum(values) / etl::size(values);
}

/*!
 * \brief Returns the mean of all the absolute values contained in the given expression
 * \param values The expression to reduce
 * \return The mean of the absolute values of the expression
 */
template <typename E>
value_t<E> amean(E&& values) {
    static_assert(is_etl_expr<E>, "etl::amean can only be used on ETL expressions");

    return asum(values) / etl::size(values);
}

/*!
 * \brief Returns the standard deviation of all the values contained in the given expression
 * \param values The expression to reduce
 * \return The standard deviation of the values of the expression
 */
template <typename E>
value_t<E> stddev(E&& values) {
    static_assert(is_etl_expr<E>, "etl::stddev can only be used on ETL expressions");

    auto mean = etl::mean(values);

    double std = 0.0;
    for (auto value : values) {
        std += (value - mean) * (value - mean);
    }

    return std::sqrt(std / etl::size(values));
}

/*!
 * \brief Returns the standard deviation of all the values contained in the given expression
 *
 * \param values The expression to reduce
 * \param mean The mean of the exprssion
 *
 * \return The standard deviation of the values of the expression
 */
template <typename E>
value_t<E> stddev(E&& values, value_t<E> mean) {
    static_assert(is_etl_expr<E>, "etl::stddev can only be used on ETL expressions");

    double std = 0.0;
    for (auto value : values) {
        std += (value - mean) * (value - mean);
    }

    return std::sqrt(std / etl::size(values));
}

namespace detail {

/*!
 * \brief Helper to compute the return type for max/min operation
 */
template <typename E>
using value_return_t =
    std::conditional_t<decay_traits<E>::is_value,
                       std::conditional_t<std::is_lvalue_reference_v<E>,
                                          std::conditional_t<std::is_const_v<std::remove_reference_t<E>>, const value_t<E>&, value_t<E>&>,
                                          value_t<E>>,
                       value_t<E>>;

} //end of namespace detail

/*!
 * \brief Returns the index of the maximum element contained in the expression
 * \param values The expression to search
 * \return The index of the maximum element of the expression
 */
template <typename E>
size_t max_index(E&& values) {
    static_assert(is_etl_expr<E>, "etl::max can only be used on ETL expressions");

    //Reduction force evaluation
    force(values);

    size_t m = 0;

    for (size_t i = 1; i < etl::size(values); ++i) {
        if (values[i] > values[m]) {
            m = i;
        }
    }

    return m;
}

/*!
 * \brief Returns the maximum element contained in the expression
 * When possible, this returns a reference to the element.
 * \param values The expression to search
 * \return The maximum element of the expression
 */
template <typename E>
detail::value_return_t<E> max(E&& values) {
    static_assert(is_etl_expr<E>, "etl::max can only be used on ETL expressions");

    auto m = max_index(values);
    return values[m];
}

/*!
 * \brief Returns the index of the minimum element contained in the expression
 * \param values The expression to search
 * \return The index of the minimum element of the expression
 */
template <typename E>
size_t min_index(E&& values) {
    static_assert(is_etl_expr<E>, "etl::min can only be used on ETL expressions");

    //Reduction force evaluation
    force(values);

    size_t m = 0;

    for (size_t i = 1; i < etl::size(values); ++i) {
        if (values[i] < values[m]) {
            m = i;
        }
    }

    return m;
}

/*!
 * \brief Returns the minimum element contained in the expression
 * When possible, this returns a reference to the element.
 * \param values The expression to search
 * \return The minimum element of the expression
 */
template <typename E>
detail::value_return_t<E> min(E&& values) {
    static_assert(is_etl_expr<E>, "etl::min can only be used on ETL expressions");

    auto m = min_index(values);
    return values[m];
}

// Generate data

/*!
 * \brief Create an expression generating numbers from a normal distribution
 * \param mean The mean of the distribution
 * \param stddev The standard deviation of the distribution
 * \return An expression generating numbers from the normal distribution
 */
template <typename T = double>
auto normal_generator(T mean = 0.0, T stddev = 1.0) -> generator_expr<normal_generator_op<T>> {
    return generator_expr<normal_generator_op<T>>{mean, stddev};
}

/*!
 * \brief Create an expression generating numbers from a normal distribution
 * using the given custom random engine.
 *
 * \param g The random engine
 * \param mean The mean of the distribution
 * \param stddev The standard deviation of the distribution
 *
 * \return An expression generating numbers from the normal distribution
 */
template <typename T = double, typename G>
auto normal_generator(G& g, T mean = 0.0, T stddev = 1.0) -> generator_expr<normal_generator_g_op<G, T>> {
    return generator_expr<normal_generator_g_op<G, T>>{g, mean, stddev};
}

/*!
 * \brief Create an expression generating numbers from a truncated normal distribution
 * \param mean The mean of the distribution
 * \param stddev The standard deviation of the distribution
 * \return An expression generating numbers from the normal distribution
 */
template <typename T = double>
auto truncated_normal_generator(T mean = 0.0, T stddev = 1.0) -> generator_expr<truncated_normal_generator_op<T>> {
    return generator_expr<truncated_normal_generator_op<T>>{mean, stddev};
}

/*!
 * \brief Create an expression generating numbers from a truncated normal distribution
 * using the given custom random engine.
 *
 * \param g The random engine
 * \param mean The mean of the distribution
 * \param stddev The standard deviation of the distribution
 *
 * \return An expression generating numbers from the normal distribution
 */
template <typename T = double, typename G>
auto truncated_normal_generator(G& g, T mean = 0.0, T stddev = 1.0) -> generator_expr<truncated_normal_generator_g_op<G, T>> {
    return generator_expr<truncated_normal_generator_g_op<G, T>>{g, mean, stddev};
}

/*!
 * \brief Create an expression generating numbers from an uniform distribution
 * \param start The beginning of the range
 * \param end The end of the range
 * \return An expression generating numbers from the uniform distribution
 */
template <typename T = double>
auto uniform_generator(T start, T end) -> generator_expr<uniform_generator_op<T>> {
    return generator_expr<uniform_generator_op<T>>{start, end};
}

/*!
 * \brief Create an expression generating numbers from an uniform distribution
 * using the given custom random engine.
 *
 * \param g The random engine
 * \param start The beginning of the range
 * \param end The end of the range
 *
 * \return An expression generating numbers from the uniform distribution
 */
template <typename T = double, typename G>
auto uniform_generator(G& g, T start, T end) -> generator_expr<uniform_generator_g_op<G, T>> {
    return generator_expr<uniform_generator_g_op<G, T>>{g, start, end};
}

/*!
 * \brief Create an expression generating numbers from a consecutive sequence
 * \param current The first number to generate
 * \return an expression generating numbers from a consecutive sequence
 */
template <typename T = double>
auto sequence_generator(T current = 0) -> generator_expr<sequence_generator_op<T>> {
    return generator_expr<sequence_generator_op<T>>{current};
}

/*!
 * \brief Create an expression generating numbers for a dropout mask
 *
 * \param probability The probability of dropout
 *
 * \return An expression generating numbers for a dropout mask
 */
template <typename T = float>
auto dropout_mask(T probability) -> generator_expr<dropout_mask_generator_op<T>> {
    return generator_expr<dropout_mask_generator_op<T>>{probability};
}

/*!
 * \brief Create an expression generating numbers for a dropout mask
 * using the given custom random engine.
 *
 * \param g The random engine
 * \param probability The probability of dropout
 *
 * \return An expression generating numbers for a dropout mask
 */
template <typename T = float, typename G>
auto dropout_mask(G& g, T probability) -> generator_expr<dropout_mask_generator_g_op<G, T>> {
    return generator_expr<dropout_mask_generator_g_op<G, T>>{g, probability};
}

/*!
 * \brief Create an expression generating numbers for a dropout mask
 *
 * \param probability The probability of dropout
 *
 * \return An expression generating numbers for a dropout mask
 */
template <typename T = float>
auto state_dropout_mask(T probability) -> generator_expr<state_dropout_mask_generator_op<T>> {
    return generator_expr<state_dropout_mask_generator_op<T>>{probability};
}

/*!
 * \brief Create an expression generating numbers for a dropout mask
 * using the given custom random engine.
 *
 * \param g The random engine
 * \param probability The probability of dropout
 *
 * \return An expression generating numbers for a dropout mask
 */
template <typename T = float, typename G>
auto state_dropout_mask(G& g, T probability) -> generator_expr<state_dropout_mask_generator_g_op<G, T>> {
    return generator_expr<state_dropout_mask_generator_g_op<G, T>>{g, probability};
}

/*!
 * \brief Create an expression generating numbers for an inverted dropout mask
 *
 * \param probability The probability of dropout
 *
 * \return An expression generating numbers for an inverted dropout mask
 */
template <typename T = float>
auto inverted_dropout_mask(T probability) -> generator_expr<inverted_dropout_mask_generator_op<T>> {
    return generator_expr<inverted_dropout_mask_generator_op<T>>{probability};
}

/*!
 * \brief Create an expression generating numbers for an inverted dropout mask
 * using the given custom random engine.
 *
 * \param g The random engine
 * \param probability The probability of dropout
 *
 * \return An expression generating numbers for an inverted dropout mask
 */
template <typename T = float, typename G>
auto state_inverted_dropout_mask(G& g, T probability) -> generator_expr<state_inverted_dropout_mask_generator_g_op<G, T>> {
    return generator_expr<state_inverted_dropout_mask_generator_g_op<G, T>>{g, probability};
}

/*!
 * \brief Create an expression generating numbers for an inverted dropout mask
 *
 * \param probability The probability of dropout
 *
 * \return An expression generating numbers for an inverted dropout mask
 */
template <typename T = float>
auto state_inverted_dropout_mask(T probability) -> generator_expr<state_inverted_dropout_mask_generator_op<T>> {
    return generator_expr<state_inverted_dropout_mask_generator_op<T>>{probability};
}

/*!
 * \brief Create an expression generating numbers for an inverted dropout mask
 * using the given custom random engine.
 *
 * \param g The random engine
 * \param probability The probability of dropout
 *
 * \return An expression generating numbers for an inverted dropout mask
 */
template <typename T = float, typename G>
auto inverted_dropout_mask(G& g, T probability) -> generator_expr<inverted_dropout_mask_generator_g_op<G, T>> {
    return generator_expr<inverted_dropout_mask_generator_g_op<G, T>>{g, probability};
}

/*!
 * \brief Force evaluation of an expression
 *
 * The temporary sub expressions will be evaluated and all the results are guaranteed to be in CPU memory.
 *
 * \return The expression
 */
template <typename Expr, cpp_enable_iff(is_etl_expr<Expr>)>
decltype(auto) operator*(Expr&& expr) {
    force(expr);
    return std::forward<Expr>(expr);
}

} //end of namespace etl

#include "etl/builder/function_expression_builder.hpp"
