//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_EXPR_HPP
#define ETL_FAST_EXPR_HPP

#include "config.hpp"

//Include implementations
#include "impl/dot.hpp"
#include "impl/scale.hpp"

namespace etl {

template<typename T>
using build_type = std::conditional_t<
    is_etl_value<T>::value,
    const std::decay_t<T>&,
    std::decay_t<T>>;

template<typename T>
using build_identity_type = std::conditional_t<
    is_etl_value<T>::value,
    std::conditional_t<
        std::is_const<std::remove_reference_t<T>>::value,
        const std::decay_t<T>&,
        std::decay_t<T>&>,
    std::decay_t<T>>;

template<typename LE, typename RE, template<typename> class OP>
using left_binary_helper = binary_expr<value_t<LE>, build_type<LE>, OP<value_t<LE>>, build_type<RE>>;

template<typename LE, typename RE, typename OP>
using left_binary_helper_op = binary_expr<value_t<LE>, build_type<LE>, OP, build_type<RE>>;

template<typename LE, typename RE, template<typename> class OP>
using right_binary_helper = binary_expr<value_t<RE>, build_type<LE>, OP<value_t<RE>>, build_type<RE>>;

template<typename E, template<typename> class OP>
using unary_helper = unary_expr<value_t<E>, build_type<E>, OP<value_t<E>>>;

template<typename E, typename OP>
using identity_helper = unary_expr<value_t<E>, OP, identity_op>;

template<typename E, typename OP>
using virtual_helper = unary_expr<E, OP, virtual_op>;

template<typename E, template<typename> class OP>
using stable_transform_helper = unary_expr<value_t<E>, OP<build_type<E>>, virtual_op>;

template<typename LE, typename RE, template<typename,typename> class OP>
using stable_transform_binary_helper = unary_expr<value_t<LE>, OP<build_type<LE>, build_type<RE>>, virtual_op>;

template<typename A, typename B, template<typename> class OP>
using temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>>, void>;

template<typename A, template<typename> class OP>
using temporary_unary_helper = temporary_unary_expr<value_t<A>, build_type<A>, OP<value_t<A>>, void>;

template<typename T, typename A, template<typename> class OP>
using temporary_unary_helper_type = temporary_unary_expr<T, build_type<A>, OP<T>, void>;

template<typename A, typename B, typename C, template<typename> class OP>
using forced_temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>>, build_identity_type<C>>;

template<typename A, typename C, template<typename> class OP>
using forced_temporary_unary_helper = temporary_unary_expr<value_t<A>, build_type<A>, OP<value_t<A>>, build_identity_type<C>>;

template<typename T, typename A, typename C, template<typename> class OP>
using forced_temporary_unary_helper_type = temporary_unary_expr<T, build_type<A>, OP<T>, build_identity_type<C>>;

template<typename A, typename B, template<typename, std::size_t> class OP, std::size_t D>
using dim_temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>, D>, void>;

template<typename A, typename B, typename C, template<typename, std::size_t> class OP, std::size_t D>
using dim_forced_temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>, D>, build_identity_type<C>>;

//{{{ Build binary expressions from two ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator-(LE&& lhs, RE&& rhs) -> left_binary_helper<LE, RE, minus_binary_op> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator+(LE&& lhs, RE&& rhs) -> left_binary_helper<LE, RE, plus_binary_op> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value, is_element_wise_mul_default::value> = cpp::detail::dummy>
auto operator*(LE&& lhs, RE&& rhs) -> left_binary_helper<LE, RE, mul_binary_op> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp_enable_if(is_etl_expr<LE>::value && is_etl_expr<RE>::value)>
auto operator>>(LE&& lhs, RE&& rhs) -> left_binary_helper<LE, RE, mul_binary_op> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp_enable_if(is_etl_expr<LE>::value && is_etl_expr<RE>::value)>
auto scale(LE&& lhs, RE&& rhs) -> left_binary_helper<LE, RE, mul_binary_op> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator/(LE&& lhs, RE&& rhs) -> left_binary_helper<LE, RE, div_binary_op> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator%(LE&& lhs, RE&& rhs) -> left_binary_helper<LE, RE, mod_binary_op> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

//}}}

//{{{ Mix scalars and ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator-(LE&& lhs, RE rhs) -> left_binary_helper<LE, scalar<value_t<LE>>, minus_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator-(LE lhs, RE&& rhs) -> right_binary_helper<scalar<value_t<RE>>, RE, minus_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator+(LE&& lhs, RE rhs) -> left_binary_helper<LE, scalar<value_t<LE>>, plus_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator+(LE lhs, RE&& rhs) -> right_binary_helper<scalar<value_t<RE>>, RE, plus_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator*(LE&& lhs, RE rhs) -> left_binary_helper<LE, scalar<value_t<LE>>, mul_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator*(LE lhs, RE&& rhs) -> right_binary_helper<scalar<value_t<RE>>, RE, mul_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator/(LE&& lhs, RE rhs) -> left_binary_helper<LE, scalar<value_t<LE>>, div_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator/(LE lhs, RE&& rhs) -> right_binary_helper<scalar<value_t<RE>>, RE, div_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, value_t<LE>>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator%(LE&& lhs, RE rhs) -> left_binary_helper<LE, scalar<value_t<LE>>, mod_binary_op> {
    return {lhs, scalar<value_t<LE>>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, value_t<RE>>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator%(LE lhs, RE&& rhs) -> right_binary_helper<scalar<value_t<RE>>, RE, mod_binary_op> {
    return {scalar<value_t<RE>>(lhs), rhs};
}

//}}}

//{{{ Compound operators

template<typename T, typename Enable = void>
struct is_etl_assignable : std::false_type {};

template<typename T>
struct is_etl_assignable<T, std::enable_if_t<is_etl_value<T>::value>> : std::true_type {};

template <typename T, typename Expr>
struct is_etl_assignable<unary_expr<T, Expr, identity_op>> : std::true_type {};

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator+=(LE&& lhs, RE rhs){
    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] += rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator+=(LE&& lhs, RE&& rhs){
    ensure_same_size(lhs, rhs);
    add_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator-=(LE&& lhs, RE rhs){
    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] -= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator-=(LE&& lhs, RE&& rhs){
    ensure_same_size(lhs, rhs);
    sub_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator*=(LE&& lhs, RE rhs){
    detail::scalar_scale<LE>::apply(std::forward<LE>(lhs), rhs);
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator*=(LE&& lhs, RE&& rhs){
    ensure_same_size(lhs, rhs);
    mul_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator>>=(LE&& lhs, RE rhs){
    detail::scalar_scale<LE>::apply(std::forward<LE>(lhs), rhs);
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator>>=(LE&& lhs, RE&& rhs){
    ensure_same_size(lhs, rhs);
    mul_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator/=(LE&& lhs, RE rhs){
    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] /= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator/=(LE&& lhs, RE&& rhs){
    ensure_same_size(lhs, rhs);
    div_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator%=(LE&& lhs, RE rhs){
    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] %= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator%=(LE&& lhs, RE&& rhs){
    ensure_same_size(lhs, rhs);
    mod_evaluate(std::forward<RE>(rhs), std::forward<LE>(lhs));
    return lhs;
}

//}}}

//{{{ Apply an unary expression on an ETL expression (vector,matrix,binary,unary)

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto operator-(E&& value) -> unary_helper<E, minus_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<std::decay_t<E>>::value> = cpp::detail::dummy>
auto abs(E&& value) -> unary_helper<E, abs_unary_op> {
    return {value};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto max(E&& value, T v) -> left_binary_helper_op<E, scalar<value_t<E>>, max_binary_op<value_t<E>, value_t<E>>> {
    return {value, scalar<value_t<E>>(v)};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto min(E&& value, T v) -> left_binary_helper_op<E, scalar<value_t<E>>, min_binary_op<value_t<E>, value_t<E>>> {
    return {value, scalar<value_t<E>>(v)};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto pow(E&& value, T v) -> left_binary_helper_op<E, scalar<value_t<E>>, pow_binary_op<value_t<E>, value_t<E>>> {
    return {value, scalar<value_t<E>>(v)};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto one_if(E&& value, T v) -> left_binary_helper_op<E, scalar<value_t<E>>, one_if_binary_op<value_t<E>, value_t<E>>> {
    return {value, scalar<value_t<E>>(v)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto one_if_max(E&& value) -> left_binary_helper_op<E, scalar<value_t<E>>, one_if_binary_op<value_t<E>, value_t<E>>> {
    return {value, scalar<value_t<E>>(max(value))};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sqrt(E&& value) -> unary_helper<E, sqrt_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto log(E&& value) -> unary_helper<E, log_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto uniform_noise(E&& value) -> unary_helper<E, uniform_noise_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto normal_noise(E&& value) -> unary_helper<E, normal_noise_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto logistic_noise(E&& value) -> unary_helper<E, logistic_noise_unary_op> {
    return {value};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto ranged_noise(E&& value, T v) -> left_binary_helper_op<E, scalar<value_t<E>>, ranged_noise_binary_op<value_t<E>, value_t<E>>> {
    return {value, scalar<value_t<E>>(v)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto exp(E&& value) -> unary_helper<E, exp_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sign(E&& value) -> unary_helper<E, sign_unary_op> {
    return {value};
}

//Note: Use of decltype here should not be necessary, but g++ does
//not like it without it for some reason

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sigmoid(E&& value) -> decltype(1.0 / (1.0 + exp(-value))) {
    return 1.0 / (1.0 + exp(-value));
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto softplus(E&& value){
    return log(1.0 + exp(value));
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto softmax(E&& e){
    return exp(e) / sum(exp(e));
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto bernoulli(const E& value) -> unary_helper<E, bernoulli_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto r_bernoulli(const E& value) -> unary_helper<E, reverse_bernoulli_unary_op> {
    return {value};
}

//}}}

//{{{ Views that returns lvalues

template<std::size_t D, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto dim(E&& value, std::size_t i) -> identity_helper<E, dim_view<build_identity_type<E>, D>> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto row(E&& value, std::size_t i) -> identity_helper<E, dim_view<build_identity_type<E>, 1>> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto col(E&& value, std::size_t i) -> identity_helper<E, dim_view<build_identity_type<E>, 2>> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sub(E&& value, std::size_t i) -> identity_helper<E, sub_view<build_identity_type<E>>> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Cannot use sub on vector");
    return {{value, i}};
}

template<std::size_t Rows, std::size_t Columns, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(E&& value) -> identity_helper<E, fast_matrix_view<build_identity_type<E>, Rows, Columns>> {
    cpp_assert(size(value) == Rows * Columns, "Invalid size for reshape");

    return {fast_matrix_view<build_identity_type<E>, Rows, Columns>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(E&& value, std::size_t rows, std::size_t columns) -> identity_helper<E, dyn_matrix_view<build_identity_type<E>>> {
    cpp_assert(size(value) == rows * columns, "Invalid size for reshape");

    return {dyn_matrix_view<build_identity_type<E>>(value, rows, columns)};
}

//}}}

//{{{ Virtual Views that returns rvalues

template<typename D = double>
auto magic(std::size_t i) -> virtual_helper<D, magic_view<D>> {
    return {{i}};
}

template<std::size_t N, typename D = double>
auto magic() -> virtual_helper<D, fast_magic_view<D, N>> {
    return {{}};
}

//}}}


//{{{ Apply a stable transformation

template<std::size_t D1, std::size_t... D, typename E, cpp_enable_if(is_etl_expr<E>::value)>
auto rep(E&& value) -> unary_expr<value_t<E>, rep_r_transformer<build_type<E>, D1, D...>, virtual_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() == 1, "Can only use rep on vector");
    return {rep_r_transformer<build_type<E>, D1, D...>(value)};
}

template<std::size_t D1, std::size_t... D, typename E, cpp_enable_if(is_etl_expr<E>::value)>
auto rep_r(E&& value) -> unary_expr<value_t<E>, rep_r_transformer<build_type<E>, D1, D...>, virtual_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() == 1, "Can only use rep on vector");
    return {rep_r_transformer<build_type<E>, D1, D...>(value)};
}

template<std::size_t D1, std::size_t... D, typename E, cpp_enable_if(is_etl_expr<E>::value)>
auto rep_l(E&& value) -> unary_expr<value_t<E>, rep_l_transformer<build_type<E>, D1, D...>, virtual_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() == 1, "Can only use rep on vector");
    return {rep_l_transformer<build_type<E>, D1, D...>(value)};
}

template<typename... D, typename E, cpp_enable_if(is_etl_expr<E>::value && cpp::all_convertible_to<std::size_t, std::size_t, D...>::value)>
auto rep(E&& value, std::size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_r_transformer<build_type<E>, 1 + sizeof...(D)>, virtual_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() == 1, "Can only use rep on vector");
    return {dyn_rep_r_transformer<build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<std::size_t>(d)...}})};
}

template<typename... D, typename E, cpp_enable_if(is_etl_expr<E>::value && cpp::all_convertible_to<std::size_t, std::size_t, D...>::value)>
auto rep_r(E&& value, std::size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_r_transformer<build_type<E>, 1 + sizeof...(D)>, virtual_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() == 1, "Can only use rep on vector");
    return {dyn_rep_r_transformer<build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<std::size_t>(d)...}})};
}

template<typename... D, typename E, cpp_enable_if(is_etl_expr<E>::value && cpp::all_convertible_to<std::size_t, std::size_t, D...>::value)>
auto rep_l(E&& value, std::size_t d1, D... d) -> unary_expr<value_t<E>, dyn_rep_l_transformer<build_type<E>, 1 + sizeof...(D)>, virtual_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() == 1, "Can only use rep on vector");
    return {dyn_rep_l_transformer<build_type<E>, 1 + sizeof...(D)>(value, {{d1, static_cast<std::size_t>(d)...}})};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sum_r(E&& value) -> stable_transform_helper<E, sum_r_transformer> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Can only use sum_r on matrix");
    return {sum_r_transformer<build_type<E>>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sum_l(E&& value) -> stable_transform_helper<E, sum_l_transformer> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Can only use sum_r on matrix");
    return {sum_l_transformer<build_type<E>>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto mean_r(E&& value) -> stable_transform_helper<E, mean_r_transformer> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Can only use sum_r on matrix");
    return {mean_r_transformer<build_type<E>>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto mean_l(E&& value) -> stable_transform_helper<E, mean_l_transformer> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Can only use sum_r on matrix");
    return {mean_l_transformer<build_type<E>>(value)};
}

//}}}

//{{{ Apply a special expression that can change order of elements

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto hflip(const E& value) -> stable_transform_helper<E, hflip_transformer> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return {hflip_transformer<build_type<E>>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto vflip(const E& value) -> stable_transform_helper<E, vflip_transformer> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return {vflip_transformer<build_type<E>>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto fflip(const E& value) -> stable_transform_helper<E, fflip_transformer> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() <= 2, "Can only use flips on 1D/2D");
    return {fflip_transformer<build_type<E>>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto transpose(const E& value) -> stable_transform_helper<E, transpose_transformer> {
    static_assert(decay_traits<E>::dimensions() <= 2, "Transpose not defined for matrix > 2D");
    return {transpose_transformer<build_type<E>>(value)};
}

template<std::size_t C1, std::size_t C2, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto p_max_pool_h(E&& value) -> unary_expr<value_t<E>, p_max_pool_h_transformer<build_type<E>, C1, C2>, virtual_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() == 2 || etl_traits<std::decay_t<E>>::dimensions() == 3,
        "Max pool is only implemented for 2D and 3D");
    return {p_max_pool_h_transformer<build_type<E>, C1, C2>(value)};
}

template<std::size_t C1, std::size_t C2, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto p_max_pool_p(E&& value) -> unary_expr<value_t<E>, p_max_pool_p_transformer<build_type<E>, C1, C2>, virtual_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() == 2 || etl_traits<std::decay_t<E>>::dimensions() == 3,
        "Max pool is only implemented for 2D and 3D");
    return {p_max_pool_p_transformer<build_type<E>, C1, C2>(value)};
}

template<typename A>
auto convmtx(A&& a, std::size_t h) -> stable_transform_helper<A, dyn_convmtx_transformer> {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 1, "Convolutional matrix only works in 1D");

    return {dyn_convmtx_transformer<build_type<A>>(a, h)};
}

template<typename A>
auto convmtx2(A&& a, std::size_t k1, std::size_t k2) -> stable_transform_helper<A, dyn_convmtx2_transformer> {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2, "Convolutional matrix only works in 2D");

    return {dyn_convmtx2_transformer<build_type<A>>(a, k1, k2)};
}

//}}}

//{{{ mul expressions

template<typename A, typename B,
    cpp_enable_if(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && !is_element_wise_mul_default::value)>
auto operator*(A&& a, B&& b) -> temporary_binary_helper<A, B, mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

template<typename A, typename B, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2,
    !is_element_wise_mul_default::value
> = cpp::detail::dummy>
auto operator*(A&& a, B&& b) -> temporary_binary_helper<A, B, vm_mul_expr> {
    return {a, b};
}

template<typename A, typename B, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1,
    !is_element_wise_mul_default::value
> = cpp::detail::dummy>
auto operator*(A&& a, B&& b) -> temporary_binary_helper<A, B, mv_mul_expr> {
    return {a, b};
}

template<typename A, typename B, cpp_enable_if(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2)>
auto mul(A&& a, B&& b) -> temporary_binary_helper<A, B, mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

template<typename A, typename B, typename C, cpp_enable_if(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2)>
auto mul(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b, c};
}

template<typename A, typename B>
auto lazy_mul(A&& a, B&& b) -> stable_transform_binary_helper<A, B, mm_mul_transformer> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {mm_mul_transformer<build_type<A>, build_type<B>>(a, b)};
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2
> = cpp::detail::dummy>
auto mul(A&& a, B&& b, C& c) -> forced_temporary_binary_helper<A, B, C, vm_mul_expr> {
    return {a, b, c};
}

template<typename A, typename B, typename C, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1
> = cpp::detail::dummy>
auto mul(A&& a, B&& b, C& c) -> forced_temporary_binary_helper<A, B, C, mv_mul_expr> {
    return {a, b, c};
}

template<typename A, typename B, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 1, decay_traits<B>::dimensions() == 2
> = cpp::detail::dummy>
auto mul(A&& a, B&& b) -> temporary_binary_helper<A, B, vm_mul_expr> {
    return {a, b};
}

template<typename A, typename B, cpp::enable_if_all_u<
    decay_traits<A>::dimensions() == 2, decay_traits<B>::dimensions() == 1
> = cpp::detail::dummy>
auto mul(A&& a, B&& b) -> temporary_binary_helper<A, B, mv_mul_expr> {
    return {a, b};
}

template<typename A, typename B>
auto strassen_mul(A&& a, B&& b) -> temporary_binary_helper<A, B, strassen_mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

template<typename A, typename B, typename C>
auto strassen_mul(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, strassen_mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b, c};
}

template<typename A, typename B>
auto outer(A&& a, B&& b) -> temporary_binary_helper<A, B, outer_product_expr> {
    return {std::forward<A>(a), std::forward<B>(b)};
}

template<typename A, typename B, typename C>
auto outer(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, outer_product_expr> {
    return {std::forward<A>(a), std::forward<B>(b), std::forward<C>(c)};
}

//}}}

//{{{ Convolution expressions

template<typename A, typename B>
auto conv_1d_valid(A&& a, B&& b) -> temporary_binary_helper<A, B, conv1_valid_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_1d_valid(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, conv1_valid_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto conv_1d_same(A&& a, B&& b) -> temporary_binary_helper<A, B, conv1_same_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_1d_same(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, conv1_same_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto conv_1d_full(A&& a, B&& b) -> temporary_binary_helper<A, B, conv1_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_1d_full(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, conv1_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto fft_conv_1d_full(A&& a, B&& b) -> temporary_binary_helper<A, B, fft_conv1_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto fft_conv_1d_full(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, fft_conv1_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto conv_2d_valid(A&& a, B&& b) -> temporary_binary_helper<A, B, conv2_valid_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_2d_valid(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, conv2_valid_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto conv_2d_same(A&& a, B&& b) -> temporary_binary_helper<A, B, conv2_same_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_2d_same(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, conv2_same_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto conv_2d_full(A&& a, B&& b) -> temporary_binary_helper<A, B, conv2_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_2d_full(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, conv2_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto fft_conv_2d_full(A&& a, B&& b) -> temporary_binary_helper<A, B, fft_conv2_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto fft_conv_2d_full(A&& a, B&& b, C&& c) -> forced_temporary_binary_helper<A, B, C, fft_conv2_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto conv_deep_valid(A&& a, B&& b) -> dim_temporary_binary_helper<A, B, conv_deep_valid_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_deep_valid(A&& a, B&& b, C&& c) -> dim_forced_temporary_binary_helper<A, B, C, conv_deep_valid_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto conv_deep_same(A&& a, B&& b) -> dim_temporary_binary_helper<A, B, conv_deep_same_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_deep_same(A&& a, B&& b, C&& c) -> dim_forced_temporary_binary_helper<A, B, C, conv_deep_same_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

template<typename A, typename B>
auto conv_deep_full(A&& a, B&& b) -> dim_temporary_binary_helper<A, B, conv_deep_full_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

template<typename A, typename B, typename C>
auto conv_deep_full(A&& a, B&& b, C&& c) -> dim_forced_temporary_binary_helper<A, B, C, conv_deep_full_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    return {a, b, c};
}

//Special convolutions

//TODO This should be moved
//TODO This should be adapted to an expression
//TODO For now, the fast version only works with square kernels

template<typename A, typename B, typename C>
void conv_2d_valid_multi(A&& input, B&& kernels, C&& features){
    //TODO Validate inputs

    //TODO This version of the implementation should only be used if very fast MMUL is available

    if(input.is_square() && kernels.is_sub_square()){
        const auto v1 = etl::dim<0>(input);
        const auto v2 = etl::dim<1>(input);
        const auto k1 = etl::dim<1>(kernels);
        const auto k2 = etl::dim<2>(kernels);

        etl::dyn_matrix<value_t<A>, 2> input_col(k1 * k2, (v1 - k1 + 1) * (v2 - k2 + 1));

        conv_2d_valid_multi(std::forward<A>(input), std::forward<B>(kernels), std::forward<C>(features), input_col);
    } else {
        //Standard version
        for(size_t k = 0; k < etl::dim<0>(kernels); ++k){
            features(k) = conv_2d_valid(input, kernels(k));
        }
    }
}

template<typename A, typename B, typename C, typename D>
void conv_2d_valid_multi(A&& input, B&& kernels, C&& features, D&& input_col){
    cpp_assert(input.is_square() && kernels.is_sub_square(), "Only implemented for square input and kernels");

    //TODO Validate inputs

    etl::dyn_matrix<value_t<B>, 3> prepared_k(etl::dim<0>(kernels), etl::dim<1>(kernels), etl::dim<2>(kernels));

    for(std::size_t i = 0; i < etl::dim<0>(kernels); ++i){
        prepared_k(i) = transpose(fflip(kernels(i)));
    }

    conv_2d_valid_multi_prepared(std::forward<A>(input), prepared_k, std::forward<C>(features), std::forward<D>(input_col));
}

template<typename A, typename B, typename C, typename D>
void conv_2d_valid_multi_prepared(A&& input, B&& kernels, C&& features, D&& input_col){
    cpp_assert(input.is_square() && kernels.is_sub_square(), "Only implemented for square input and kernels");

    //TODO Validate inputs

    const auto K  = etl::dim<0>(kernels);
    const auto k1 = etl::dim<1>(kernels);
    const auto k2 = etl::dim<2>(kernels);

    im2col_direct(input_col, input, k1, k2);

    *mul(
        etl::reshape(kernels, K, k1 * k2),
        input_col,
        etl::reshape(features, K, etl::dim<1>(features) * etl::dim<2>(features)));

    for(std::size_t k = 0; k < K; ++k){
        features(k).transpose_inplace();
    }
}

//}}}

//{{{ Fast-Fourrier-Transform

//Helpers to compute the type of the result

template<typename A>
using fft_value_type = std::conditional_t<is_complex<A>::value, value_t<A>, std::complex<value_t<A>>>;

template<typename A>
using ifft_value_type = std::conditional_t<is_complex<A>::value, value_t<A>, std::complex<value_t<A>>>;

template<typename A>
using ifft_real_value_type = std::conditional_t<is_complex<A>::value, typename value_t<A>::value_type, value_t<A>>;

template<typename A>
auto fft_1d(A&& a) -> temporary_unary_helper_type<fft_value_type<A>, A, fft1_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return {a};
}

template<typename A, typename C>
auto fft_1d(A&& a, C&& c) -> forced_temporary_unary_helper_type<fft_value_type<A>, A, C, fft1_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");

    return {a, c};
}

template<typename A>
auto ifft_1d(A&& a) -> temporary_unary_helper_type<ifft_value_type<A>, A, ifft1_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return {a};
}

template<typename A, typename C>
auto ifft_1d(A&& a, C&& c) -> forced_temporary_unary_helper_type<ifft_value_type<A>, A, C, ifft1_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");

    return {a, c};
}

template<typename A>
auto ifft_1d_real(A&& a) -> temporary_unary_helper_type<ifft_real_value_type<A>, A, ifft1_real_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return {a};
}

template<typename A, typename C>
auto ifft_1d_real(A&& a, C&& c) -> forced_temporary_unary_helper_type<ifft_real_value_type<A>, A, C, ifft1_real_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");

    return {a, c};
}

template<typename A>
auto fft_2d(A&& a) -> temporary_unary_helper_type<fft_value_type<A>, A, fft2_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return {a};
}

template<typename A, typename C>
auto fft_2d(A&& a, C&& c) -> forced_temporary_unary_helper_type<fft_value_type<A>, A, C, fft2_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");

    return {a, c};
}

template<typename A>
auto ifft_2d(A&& a) -> temporary_unary_helper_type<fft_value_type<A>, A, ifft2_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return {a};
}

template<typename A, typename C>
auto ifft_2d(A&& a, C&& c) -> forced_temporary_unary_helper_type<fft_value_type<A>, A, C, ifft2_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");

    return {a, c};
}

template<typename A>
auto ifft_2d_real(A&& a) -> temporary_unary_helper_type<ifft_real_value_type<A>, A, ifft2_real_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return {a};
}

template<typename A, typename C>
auto ifft_2d_real(A&& a, C&& c) -> forced_temporary_unary_helper_type<ifft_real_value_type<A>, A, C, ifft2_real_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");

    return {a, c};
}

//}}}

//{{{ Apply a reduction on an ETL expression (vector,matrix,binary,unary)

template<typename A, typename B, cpp::enable_if_all_u<is_etl_expr<A>::value, is_etl_expr<B>::value> = cpp::detail::dummy>
value_t<A> dot(const A& a, const B& b){
    return detail::dot_impl<A, B>::apply(a, b);
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
typename E::value_type sum(const E& values){
    typename E::value_type acc(0);

    for(std::size_t i = 0; i < size(values); ++i){
        acc += values[i];
    }

    return acc;
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
typename E::value_type mean(const E& values){
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

template<typename E, cpp::enable_if_u<is_etl_expr<std::decay_t<E>>::value> = cpp::detail::dummy>
value_return_t<E> max(E&& values){
    std::size_t m = 0;

    for(std::size_t i = 1; i < size(values); ++i){
        if(values[i] > values[m]){
            m = i;
        }
    }

    return values[m];
}

template<typename E, cpp::enable_if_u<is_etl_expr<std::decay_t<E>>::value> = cpp::detail::dummy>
value_return_t<E> min(E&& values){
    std::size_t m = 0;

    for(std::size_t i = 1; i < size(values); ++i){
        if(values[i] < values[m]){
            m = i;
        }
    }

    return values[m];
}

//}}}

//{{{ Generate data

template<typename T = double>
auto normal_generator() -> generator_expr<normal_generator_op<T>> {
    return {};
}

template<typename T = double>
auto sequence_generator(T current = 0) -> generator_expr<sequence_generator_op<T>> {
    return {current};
}

//}}}

//Force evaluation of an expression

template<typename Expr, cpp_enable_if(is_etl_expr<std::decay_t<Expr>>::value)>
decltype(auto) operator*(Expr&& expr){
    force(expr);
    return std::forward<Expr>(expr);
}

} //end of namespace etl

#endif
