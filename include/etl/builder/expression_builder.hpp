//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file expression_builder.hpp
 * \brief Contains all the operators and functions to build expressions.
*/

#pragma once

#include "etl/config.hpp"
#include "etl/expression_helpers.hpp"

//Include implementations
#include "etl/impl/dot.hpp"
#include "etl/impl/scalar_op.hpp"

namespace etl {

// Build binary expressions from two ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator-(LE&& lhs, RE&& rhs) -> detail::left_binary_helper<LE, RE, minus_binary_op> {
    validate_expression(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator+(LE&& lhs, RE&& rhs) -> detail::left_binary_helper<LE, RE, plus_binary_op> {
    validate_expression(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value, is_element_wise_mul_default> = cpp::detail::dummy>
auto operator*(LE&& lhs, RE&& rhs) -> detail::left_binary_helper<LE, RE, mul_binary_op> {
    validate_expression(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp_enable_if(is_etl_expr<LE>::value && is_etl_expr<RE>::value)>
auto operator>>(LE&& lhs, RE&& rhs) -> detail::left_binary_helper<LE, RE, mul_binary_op> {
    validate_expression(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp_enable_if(is_etl_expr<LE>::value && is_etl_expr<RE>::value)>
auto scale(LE&& lhs, RE&& rhs) -> detail::left_binary_helper<LE, RE, mul_binary_op> {
    validate_expression(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator/(LE&& lhs, RE&& rhs) -> detail::left_binary_helper<LE, RE, div_binary_op> {
    validate_expression(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator%(LE&& lhs, RE&& rhs) -> detail::left_binary_helper<LE, RE, mod_binary_op> {
    validate_expression(lhs, rhs);

    return {lhs, rhs};
}

// Mix scalars and ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator-(LE&& lhs, RE rhs) -> detail::left_binary_helper<LE, scalar<value_t<LE>>, minus_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator-(LE lhs, RE&& rhs) -> detail::right_binary_helper<scalar<value_t<RE>>, RE, minus_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator+(LE&& lhs, RE rhs) -> detail::left_binary_helper<LE, scalar<value_t<LE>>, plus_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator+(LE lhs, RE&& rhs) -> detail::right_binary_helper<scalar<value_t<RE>>, RE, plus_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator*(LE&& lhs, RE rhs) -> detail::left_binary_helper<LE, scalar<value_t<LE>>, mul_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator*(LE lhs, RE&& rhs) -> detail::right_binary_helper<scalar<value_t<RE>>, RE, mul_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator>>(LE&& lhs, RE rhs) -> detail::left_binary_helper<LE, scalar<value_t<LE>>, mul_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator>>(LE lhs, RE&& rhs) -> detail::right_binary_helper<scalar<value_t<RE>>, RE, mul_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp_enable_if(std::is_convertible<RE, value_t<LE>>::value && is_etl_expr<LE>::value && (is_div_strict || !std::is_floating_point<RE>::value))>
auto operator/(LE&& lhs, RE rhs) -> detail::left_binary_helper<LE, scalar<value_t<LE>>, div_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp_enable_if(std::is_convertible<RE, value_t<LE>>::value && is_etl_expr<LE>::value && !is_div_strict && std::is_floating_point<RE>::value)>
auto operator/(LE&& lhs, RE rhs) -> detail::left_binary_helper<LE, scalar<value_t<LE>>, mul_binary_op> {
    return {lhs, scalar<value_t<LE>>(value_t<LE>(1.0) / rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator/(LE lhs, RE&& rhs) -> detail::right_binary_helper<scalar<value_t<RE>>, RE, div_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator%(LE&& lhs, RE rhs) -> detail::left_binary_helper<LE, scalar<value_t<LE>>, mod_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator%(LE lhs, RE&& rhs) -> detail::right_binary_helper<scalar<value_t<RE>>, RE, mod_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

// Compound operators

template<typename T, typename Enable = void>
struct is_etl_assignable : std::false_type {};

template<typename T>
struct is_etl_assignable<T, std::enable_if_t<is_etl_value<T>::value>> : std::true_type {};

template <typename T, typename Expr>
struct is_etl_assignable<unary_expr<T, Expr, identity_op>> : std::true_type {};

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator+=(LE&& lhs, RE rhs){
    detail::scalar_add<LE>::apply(std::forward<LE>(lhs), rhs);
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator+=(LE&& lhs, RE&& rhs){
    validate_expression(lhs, rhs);
    add_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator-=(LE&& lhs, RE rhs){
    detail::scalar_sub<LE>::apply(std::forward<LE>(lhs), rhs);
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator-=(LE&& lhs, RE&& rhs){
    validate_expression(lhs, rhs);
    sub_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator*=(LE&& lhs, RE rhs){
    detail::scalar_mul<LE>::apply(std::forward<LE>(lhs), rhs);
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator*=(LE&& lhs, RE&& rhs){
    validate_expression(lhs, rhs);
    mul_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator>>=(LE&& lhs, RE rhs){
    detail::scalar_mul<LE>::apply(std::forward<LE>(lhs), rhs);
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator>>=(LE&& lhs, RE&& rhs){
    validate_expression(lhs, rhs);
    mul_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator/=(LE&& lhs, RE rhs){
    detail::scalar_div<LE>::apply(std::forward<LE>(lhs), rhs);
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator/=(LE&& lhs, RE&& rhs){
    validate_expression(lhs, rhs);
    div_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator%=(LE&& lhs, RE rhs){
    detail::scalar_mod<LE>::apply(std::forward<LE>(lhs), rhs);
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator%=(LE&& lhs, RE&& rhs){
    validate_expression(lhs, rhs);
    mod_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

// Apply an unary expression on an ETL expression (vector,matrix,binary,unary)

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto operator-(E&& value) -> detail::unary_helper<E, minus_unary_op> {
    return detail::unary_helper<E, minus_unary_op>{value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto operator+(E&& value) -> detail::unary_helper<E, plus_unary_op> {
    return detail::unary_helper<E, plus_unary_op>{value};
}

template<typename E>
auto abs(E&& value) -> detail::unary_helper<E, abs_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::abs can only be used on ETL expressions");
    return detail::unary_helper<E, abs_unary_op>{value};
}

template<typename E, typename T, cpp_enable_if(std::is_arithmetic<T>::value)>
auto max(E&& value, T v){
    static_assert(is_etl_expr<E>::value, "etl::max can only be used on ETL expressions");
    return detail::make_stateful_unary_expr<E, max_scalar_op<value_t<E>, value_t<E>>>(value, value_t<E>(v));
}

template<typename L, typename R, cpp_enable_if(!std::is_arithmetic<R>::value)>
auto max(L&& lhs, R&& rhs) -> detail::left_binary_helper_op<L, R, max_binary_op<value_t<L>, value_t<R>>> {
    static_assert(is_etl_expr<L>::value, "etl::max can only be used on ETL expressions");
    return {lhs, rhs};
}

template<typename E, typename T, cpp_enable_if(std::is_arithmetic<T>::value)>
auto min(E&& value, T v){
    static_assert(is_etl_expr<E>::value, "etl::max can only be used on ETL expressions");
    return detail::make_stateful_unary_expr<E, min_scalar_op<value_t<E>, value_t<E>>>(value, value_t<E>(v));
}

template<typename L, typename R, cpp_enable_if(!std::is_arithmetic<R>::value)>
auto min(L&& lhs, R&& rhs) -> detail::left_binary_helper_op<L, R, min_binary_op<value_t<L>, value_t<R>>> {
    static_assert(is_etl_expr<L>::value, "etl::max can only be used on ETL expressions");
    return {lhs, rhs};
}

template<typename E, typename T>
auto clip(E&& value, T min, T max){
    static_assert(is_etl_expr<E>::value, "etl::clip can only be used on ETL expressions");
    static_assert(std::is_arithmetic<T>::value, "etl::clip can only be used with arithmetic values");
    return detail::make_stateful_unary_expr<E, clip_scalar_op<value_t<E>, value_t<E>>>(value, value_t<E>(min), value_t<E>(max));
}

template<typename E, typename T>
auto pow(E&& value, T v) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, pow_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>::value, "etl::pow can only be used on ETL expressions");
    static_assert(std::is_arithmetic<T>::value, "etl::pow can only be used with arithmetic values");
    return {value, scalar<value_t<E>>(v)};
}

template<typename E, typename T>
auto one_if(E&& value, T v) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, one_if_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>::value, "etl::one_if can only be used on ETL expressions");
    static_assert(std::is_arithmetic<T>::value, "etl::one_if can only be used with arithmetic values");
    return {value, scalar<value_t<E>>(v)};
}

template<typename E>
auto one_if_max(E&& value) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, one_if_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>::value, "etl::one_if_max can only be used on ETL expressions");
    return {value, scalar<value_t<E>>(max(value))};
}

template<typename E>
auto sqrt(E&& value) -> detail::unary_helper<E, sqrt_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::sqrt can only be used on ETL expressions");
    return detail::unary_helper<E, sqrt_unary_op>{value};
}

template<typename E>
auto log(E&& value) -> detail::unary_helper<E, log_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::log can only be used on ETL expressions");
    return detail::unary_helper<E, log_unary_op>{value};
}

template<typename E>
auto tan(E&& value) -> detail::unary_helper<E, tan_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::tan can only be used on ETL expressions");
    return detail::unary_helper<E, tan_unary_op>{value};
}

template<typename E>
auto cos(E&& value) -> detail::unary_helper<E, cos_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::cos can only be used on ETL expressions");
    return detail::unary_helper<E, cos_unary_op>{value};
}

template<typename E>
auto sin(E&& value) -> detail::unary_helper<E, sin_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::sin can only be used on ETL expressions");
    return detail::unary_helper<E, sin_unary_op>{value};
}

template<typename E>
auto tanh(E&& value) -> detail::unary_helper<E, tanh_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::tanh can only be used on ETL expressions");
    return detail::unary_helper<E, tanh_unary_op>{value};
}

template<typename E>
auto cosh(E&& value) -> detail::unary_helper<E, cosh_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::cosh can only be used on ETL expressions");
    return detail::unary_helper<E, cosh_unary_op>{value};
}

template<typename E>
auto sinh(E&& value) -> detail::unary_helper<E, sinh_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::sinh can only be used on ETL expressions");
    return detail::unary_helper<E, sinh_unary_op>{value};
}

template<typename E>
auto uniform_noise(E&& value) -> detail::unary_helper<E, uniform_noise_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::uniform_noise can only be used on ETL expressions");
    return detail::unary_helper<E, uniform_noise_unary_op>{value};
}

template<typename E>
auto normal_noise(E&& value) -> detail::unary_helper<E, normal_noise_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::normal_noise can only be used on ETL expressions");
    return detail::unary_helper<E, normal_noise_unary_op>{value};
}

template<typename E>
auto logistic_noise(E&& value) -> detail::unary_helper<E, logistic_noise_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::logistic_noise can only be used on ETL expressions");
    return detail::unary_helper<E, logistic_noise_unary_op>{value};
}

template<typename E, typename T>
auto ranged_noise(E&& value, T v) -> detail::left_binary_helper_op<E, scalar<value_t<E>>, ranged_noise_binary_op<value_t<E>, value_t<E>>> {
    static_assert(is_etl_expr<E>::value, "etl::ranged_noise can only be used on ETL expressions");
    static_assert(std::is_arithmetic<T>::value, "etl::ranged_noise can only be used with arithmetic values");
    return {value, scalar<value_t<E>>(v)};
}

template<typename E>
auto exp(E&& value) -> detail::unary_helper<E, exp_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::exp can only be used on ETL expressions");
    return detail::unary_helper<E, exp_unary_op>{value};
}

template<typename E>
auto sign(E&& value) -> detail::unary_helper<E, sign_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::sign can only be used on ETL expressions");
    return detail::unary_helper<E, sign_unary_op>{value};
}

template<typename E>
auto identity(E&& value){
    return std::forward<E>(value);
}

template<typename E>
auto identity_derivative(E&&){
    return 1.0;
}

//Note: Use of decltype here should not be necessary, but g++ does
//not like it without it for some reason

template<typename E>
auto sigmoid(E&& value) -> decltype(1.0 / (1.0 + exp(-value))) {
    static_assert(is_etl_expr<E>::value, "etl::sigmoid can only be used on ETL expressions");
    return 1.0 / (1.0 + exp(-value));
}

template<typename E>
auto sigmoid_derivative(E&& value) -> decltype(value >> (1.0 - value)) {
    static_assert(is_etl_expr<E>::value, "etl::sigmoid_derivative can only be used on ETL expressions");
    return value >> (1.0 - value);
}

template<typename E>
auto fast_sigmoid(const E& value) -> detail::unary_helper<E, fast_sigmoid_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::fast_sigmoid can only be used on ETL expressions");
    return detail::unary_helper<E, fast_sigmoid_unary_op>{value};
}

template<typename E>
auto hard_sigmoid(E&& x) -> decltype(etl::clip(x * 0.2 + 0.5, 0.0, 1.0)) {
    static_assert(is_etl_expr<E>::value, "etl::hard_sigmoid can only be used on ETL expressions");
    return etl::clip(x * 0.2 + 0.5, 0.0, 1.0);
}

template<typename E>
auto softplus(E&& value) -> decltype(log(1.0 + exp(value))) {
    static_assert(is_etl_expr<E>::value, "etl::softplus can only be used on ETL expressions");
    return log(1.0 + exp(value));
}

template<typename E>
auto softmax(E&& e){
    static_assert(is_etl_expr<E>::value, "etl::softmax can only be used on ETL expressions");
    return exp(e) / sum(exp(e));
}

template<typename E>
auto stable_softmax(E&& e){
    static_assert(is_etl_expr<E>::value, "etl::stable_softmax can only be used on ETL expressions");
    auto m = max(e);
    return exp(e - m) / sum(exp(e - m));
}

template<typename E>
auto bernoulli(const E& value) -> detail::unary_helper<E, bernoulli_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::bernoulli can only be used on ETL expressions");
    return detail::unary_helper<E, bernoulli_unary_op>{value};
}

template<typename E>
auto r_bernoulli(const E& value) -> detail::unary_helper<E, reverse_bernoulli_unary_op> {
    static_assert(is_etl_expr<E>::value, "etl::r_bernoulli can only be used on ETL expressions");
    return detail::unary_helper<E, reverse_bernoulli_unary_op>{value};
}

template<typename E>
auto tanh_derivative(E&& value) -> decltype(1.0 - (value >> value)) {
    static_assert(is_etl_expr<E>::value, "etl::tanh_derivative can only be used on ETL expressions");
    return 1.0 - (value >> value);
}

template<typename E>
auto relu(E&& value) -> decltype(max(value, 0.0)) {
    static_assert(is_etl_expr<E>::value, "etl::relu can only be used on ETL expressions");
    return max(value, 0.0);
}

template<typename E>
auto relu_derivative(const E& value) -> detail::unary_helper<E, relu_derivative_op> {
    static_assert(is_etl_expr<E>::value, "etl::relu_derivative can only be used on ETL expressions");
    return detail::unary_helper<E, relu_derivative_op>{value};
}

// Views that returns lvalues

template<std::size_t D, typename E>
auto dim(E&& value, std::size_t i) -> detail::identity_helper<E, dim_view<detail::build_identity_type<E>, D>> {
    static_assert(is_etl_expr<E>::value, "etl::dim can only be used on ETL expressions");
    return detail::identity_helper<E, dim_view<detail::build_identity_type<E>, D>>{{value, i}};
}

template<typename E>
auto row(E&& value, std::size_t i) -> detail::identity_helper<E, dim_view<detail::build_identity_type<E>, 1>> {
    static_assert(is_etl_expr<E>::value, "etl::row can only be used on ETL expressions");
    return detail::identity_helper<E, dim_view<detail::build_identity_type<E>, 1>>{{value, i}};
}

template<typename E>
auto col(E&& value, std::size_t i) -> detail::identity_helper<E, dim_view<detail::build_identity_type<E>, 2>> {
    static_assert(is_etl_expr<E>::value, "etl::col can only be used on ETL expressions");
    return detail::identity_helper<E, dim_view<detail::build_identity_type<E>, 2>>{{value, i}};
}

template<typename E>
auto sub(E&& value, std::size_t i) -> detail::identity_helper<E, sub_view<detail::build_identity_type<E>>> {
    static_assert(is_etl_expr<E>::value, "etl::sub can only be used on ETL expressions");
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Cannot use sub on vector");
    return detail::identity_helper<E, sub_view<detail::build_identity_type<E>>>{{value, i}};
}

template<std::size_t... Dims, typename E>
auto reshape(E&& value) -> detail::identity_helper<E, fast_matrix_view<detail::build_identity_type<E>, Dims...>> {
    static_assert(is_etl_expr<E>::value, "etl::reshape can only be used on ETL expressions");
    cpp_assert(size(value) == mul_all<Dims...>::value, "Invalid size for reshape");

    return detail::identity_helper<E, fast_matrix_view<detail::build_identity_type<E>, Dims...>>{fast_matrix_view<detail::build_identity_type<E>, Dims...>(value)};
}

template<typename E>
auto reshape(E&& value, std::size_t rows, std::size_t columns) -> detail::identity_helper<E, dyn_matrix_view<detail::build_identity_type<E>>> {
    static_assert(is_etl_expr<E>::value, "etl::reshape can only be used on ETL expressions");
    cpp_assert(size(value) == rows * columns, "Invalid size for reshape");

    return detail::identity_helper<E, dyn_matrix_view<detail::build_identity_type<E>>>{{value, rows, columns}};
}

// Virtual Views that returns rvalues

template<typename D = double>
auto magic(std::size_t i) -> detail::virtual_helper<D, magic_view<D>> {
    return detail::virtual_helper<D, magic_view<D>>{magic_view<D>{i}};
}

template<std::size_t N, typename D = double>
auto magic() -> detail::virtual_helper<D, fast_magic_view<D, N>> {
    return detail::virtual_helper<D, fast_magic_view<D, N>>{{}};
}

// Apply a stable transformation

template<std::size_t D1, std::size_t... D, typename E>
auto rep(E&& value) -> unary_expr<value_t<E>, rep_r_transformer<detail::build_type<E>, D1, D...>, transform_op> {
    static_assert(is_etl_expr<E>::value, "etl::rep can only be used on ETL expressions");
    return unary_expr<value_t<E>, rep_r_transformer<detail::build_type<E>, D1, D...>, transform_op>{rep_r_transformer<detail::build_type<E>, D1, D...>(value)};
}

template<std::size_t D1, std::size_t... D, typename E>
auto rep_r(E&& value) -> unary_expr<value_t<E>, rep_r_transformer<detail::build_type<E>, D1, D...>, transform_op> {
    static_assert(is_etl_expr<E>::value, "etl::rep_r can only be used on ETL expressions");
    return unary_expr<value_t<E>, rep_r_transformer<detail::build_type<E>, D1, D...>, transform_op>{rep_r_transformer<detail::build_type<E>, D1, D...>(value)};
}

template<std::size_t D1, std::size_t... D, typename E>
auto rep_l(E&& value) -> unary_expr<value_t<E>, rep_l_transformer<detail::build_type<E>, D1, D...>, transform_op> {
    static_assert(is_etl_expr<E>::value, "etl::rep_l can only be used on ETL expressions");
    return unary_expr<value_t<E>, rep_l_transformer<detail::build_type<E>, D1, D...>, transform_op>{rep_l_transformer<detail::build_type<E>, D1, D...>(value)};
}

template<typename... D, typename E, cpp_enable_if(cpp::all_convertible_to<std::size_t, std::size_t, D...>::value)>
auto rep(E&& value, std::size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op> {
    static_assert(is_etl_expr<E>::value, "etl::rep can only be used on ETL expressions");
    return unary_expr<value_t<E>, dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op>{
        dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<std::size_t>(d)...}})};
}

template<typename... D, typename E, cpp_enable_if(cpp::all_convertible_to<std::size_t, std::size_t, D...>::value)>
auto rep_r(E&& value, std::size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op> {
    static_assert(is_etl_expr<E>::value, "etl::rep_r can only be used on ETL expressions");
    return unary_expr<value_t<E>, dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op>{
        dyn_rep_r_transformer<detail::build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<std::size_t>(d)...}})};
}

template<typename... D, typename E, cpp_enable_if(cpp::all_convertible_to<std::size_t, std::size_t, D...>::value)>
auto rep_l(E&& value, std::size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_l_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op> {
    static_assert(is_etl_expr<E>::value, "etl::rep_l can only be used on ETL expressions");
    return unary_expr<value_t<E>, dyn_rep_l_transformer<detail::build_type<E>, 1 + sizeof...(D)>, transform_op>{
        dyn_rep_l_transformer<detail::build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<std::size_t>(d)...}})};
}

template<typename E>
auto sum_r(E&& value) -> detail::stable_transform_helper<E, sum_r_transformer> {
    static_assert(is_etl_expr<E>::value, "etl::sum_r can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() > 1, "Can only use sum_r on matrix");
    return detail::make_transform_expr<E, sum_r_transformer>(value);
}

template<typename E>
auto sum_l(E&& value) -> detail::stable_transform_helper<E, sum_l_transformer> {
    static_assert(is_etl_expr<E>::value, "etl::sum_l can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() > 1, "Can only use sum_l on matrix");
    return detail::make_transform_expr<E, sum_l_transformer>(value);
}

template<typename E>
auto mean_r(E&& value) -> detail::stable_transform_helper<E, mean_r_transformer> {
    static_assert(is_etl_expr<E>::value, "etl::mean_r can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() > 1, "Can only use mean_r on matrix");
    return detail::make_transform_expr<E, mean_r_transformer>(value);
}

template<typename E>
auto mean_l(E&& value) -> detail::stable_transform_helper<E, mean_l_transformer> {
    static_assert(is_etl_expr<E>::value, "etl::mean_l can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() > 1, "Can only use mean_l on matrix");
    return detail::make_transform_expr<E, mean_l_transformer>(value);
}

// Apply a special expression that can change order of elements

template<typename E>
auto hflip(const E& value) -> detail::stable_transform_helper<E, hflip_transformer> {
    static_assert(is_etl_expr<E>::value, "etl::hflip can only be used on ETL expressions");
    static_assert(etl_traits<std::decay_t<E>>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return detail::make_transform_expr<E, hflip_transformer>(value);
}

template<typename E>
auto vflip(const E& value) -> detail::stable_transform_helper<E, vflip_transformer> {
    static_assert(is_etl_expr<E>::value, "etl::vflip can only be used on ETL expressions");
    static_assert(etl_traits<std::decay_t<E>>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return detail::make_transform_expr<E, vflip_transformer>(value);
}

template<typename E>
auto fflip(const E& value) -> detail::stable_transform_helper<E, fflip_transformer> {
    static_assert(is_etl_expr<E>::value, "etl::fflip can only be used on ETL expressions");
    static_assert(etl_traits<std::decay_t<E>>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return detail::make_transform_expr<E, fflip_transformer>(value);
}

template<typename E>
auto transpose(const E& value) -> detail::stable_transform_helper<E, transpose_transformer> {
    static_assert(is_etl_expr<E>::value, "etl::transpose can only be used on ETL expressions");
    static_assert(decay_traits<E>::dimensions() <= 2, "Transpose not defined for matrix > 2D");
    return detail::make_transform_expr<E, transpose_transformer>(value);
}

// Apply a reduction on an ETL expression (vector,matrix,binary,unary)

template<typename A, typename B>
value_t<A> dot(const A& a, const B& b){
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "etl::dot can only be used on ETL expressions");
    validate_expression(a, b);
    return detail::dot_impl<A, B>::apply(a, b);
}

template<typename E>
value_t<E> sum(E&& values){
    static_assert(is_etl_expr<E>::value, "etl::sum can only be used on ETL expressions");

    //Reduction force evaluation
    force(values);

    value_t<E> acc(0);

    for(std::size_t i = 0; i < size(values); ++i){
        acc += values[i];
    }

    return acc;
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
value_t<E> mean(E&& values){
    static_assert(is_etl_expr<E>::value, "etl::mean can only be used on ETL expressions");

    return sum(values) / size(values);
}

template<typename E>
struct value_return_type {
using type =
    std::conditional_t<
        decay_traits<E>::is_value,
        std::conditional_t<
            std::is_lvalue_reference<E>::value,
            std::conditional_t<
                std::is_const<std::remove_reference_t<E>>::value,
                const value_t<E>&,
                value_t<E>&
            >,
            value_t<E>
        >,
        value_t<E>
    >;
};

template<typename E>
using value_return_t = typename value_return_type<E>::type;

template<typename E>
value_return_t<E> max(E&& values){
    static_assert(is_etl_expr<E>::value, "etl::max can only be used on ETL expressions");

    //Reduction force evaluation
    force(values);

    std::size_t m = 0;

    for(std::size_t i = 1; i < size(values); ++i){
        if(values[i] > values[m]){
            m = i;
        }
    }

    return values[m];
}

template<typename E>
value_return_t<E> min(E&& values){
    static_assert(is_etl_expr<E>::value, "etl::min can only be used on ETL expressions");

    //Reduction force evaluation
    force(values);

    std::size_t m = 0;

    for(std::size_t i = 1; i < size(values); ++i){
        if(values[i] < values[m]){
            m = i;
        }
    }

    return values[m];
}

// Generate data

template<typename T = double>
auto normal_generator(T mean = 0.0, T stddev = 1.0) -> generator_expr<normal_generator_op<T>> {
    return generator_expr<normal_generator_op<T>>{mean, stddev};
}

template<typename T = double>
auto sequence_generator(T current = 0) -> generator_expr<sequence_generator_op<T>> {
    return generator_expr<sequence_generator_op<T>>{current};
}

//Force optimization of an expression

template <typename Expr>
auto opt(Expr&& expr) -> optimized_expr<detail::build_type<Expr>> {
    return {expr};
}

//Force evaluation of an expression

template<typename Expr, cpp_enable_if(is_etl_expr<std::decay_t<Expr>>::value)>
decltype(auto) operator*(Expr&& expr){
    force(expr);
    return std::forward<Expr>(expr);
}

} //end of namespace etl
