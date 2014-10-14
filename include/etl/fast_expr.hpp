//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_EXPR_HPP
#define ETL_FAST_EXPR_HPP

#include "fast_op.hpp"
#include "traits.hpp"
#include "binary_expr.hpp"
#include "unary_expr.hpp"
#include "transform_expr.hpp"
#include "generator_expr.hpp"

namespace etl {

//{{{ Build binary expressions from two ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator-(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, minus_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator+(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, plus_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator*(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, mul_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator/(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, div_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator%(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, mod_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

//}}}

//{{{ Mix scalars and ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator-(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, minus_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator-(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, minus_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator+(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, plus_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator+(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, plus_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator*(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, mul_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}
template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value> = cpp::detail::dummy> auto operator*(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, mul_binary_op<typename RE::value_type>, const RE&> { return {scalar<typename RE::value_type>(lhs), rhs}; }

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator/(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, div_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator/(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, div_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value> = cpp::detail::dummy>
auto operator%(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, mod_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator%(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, mod_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

//}}}

//{{{ Compound operators

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator+=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] += rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator+=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] += rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator-=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] -= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator-=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] -= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator*=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] *= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator*=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] *= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator/=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] /= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator/=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] /= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator%=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] %= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_value<LE>::value> = cpp::detail::dummy>
LE& operator%=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] %= rhs[i];
    }

    return lhs;
}

//}}}

//{{{ Apply an unary expression on an ETL expression (vector,matrix,binary,unary)

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto abs(const E& value) -> unary_expr<typename E::value_type, const E&, abs_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto max(const E& value, T v) -> binary_expr<typename E::value_type, const E&, max_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto min(const E& value, T v) -> binary_expr<typename E::value_type, const E&, min_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto one_if(const E& value, T v) -> binary_expr<typename E::value_type, const E&, one_if_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto one_if_max(const E& value) -> binary_expr<typename E::value_type, const E&, one_if_binary_op<typename E::value_type, typename E::value_type>, scalar<typename E::value_type>> {
    return {value, scalar<typename E::value_type>(max(value))};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto log(const E& value) -> unary_expr<typename E::value_type, const E&, log_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto uniform_noise(const E& value) -> unary_expr<typename E::value_type, const E&, uniform_noise_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto normal_noise(const E& value) -> unary_expr<typename E::value_type, const E&, normal_noise_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto logistic_noise(const E& value) -> unary_expr<typename E::value_type, const E&, logistic_noise_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto ranged_noise(const E& value, T v) -> binary_expr<typename E::value_type, const E&, ranged_noise_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto exp(const E& value) -> unary_expr<typename E::value_type, const E&, exp_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sign(const E& value) -> unary_expr<typename E::value_type, const E&, sign_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sigmoid(const E& value) -> unary_expr<typename E::value_type, const E&, sigmoid_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto softplus(const E& value) -> unary_expr<typename E::value_type, const E&, softplus_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto softmax(const E& e){
    //Unfortunately, it is not possible to directly return exp(e) / sum(exp(e))
    //because the binary_expr would hold a reference to an object whose lifetime
    //is not longer than this function, therefore running into UB
    //it is necessary to ensure that the UE is stored by value and therefore
    //copied
    //It would be good to enhance the system to support deep copying when
    //necessray

    auto s = sum(exp(e));

    using ue = unary_expr<typename E::value_type, const E&, exp_unary_op<typename E::value_type>>;
    using st = scalar<typename E::value_type>;

    return binary_expr<typename E::value_type, ue, div_binary_op<typename E::value_type>, st>(ue(e), st(s));
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto bernoulli(const E& value) -> unary_expr<typename E::value_type, const E&, bernoulli_unary_op<typename E::value_type>> {
    return {value};
}

//}}}

//{{{ Views that returns lvalues

template<std::size_t D, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto dim(E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<E, D>, identity_op> {
    return {{value, i}};
}

template<std::size_t D, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto dim(const E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<const E, D>, identity_op> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto row(E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<E, 1>, identity_op> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto row(const E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<const E, 1>, identity_op> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto col(E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<E, 2>, identity_op> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto col(const E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<const E, 2>, identity_op> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sub(E& value, std::size_t i) -> unary_expr<typename E::value_type, sub_view<E>, identity_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Cannot use sub on vector");
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sub(const E& value, std::size_t i) -> unary_expr<typename E::value_type, sub_view<const E>, identity_op> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Cannot use sub on vector");
    return {{value, i}};
}

template<std::size_t Rows, std::size_t Columns, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(E& value) -> unary_expr<typename E::value_type, fast_matrix_view<E, Rows, Columns>, identity_op> {
    cpp_assert(etl_traits<std::decay_t<E>>::size(value) == Rows * Columns, "Invalid size for reshape");

    return {fast_matrix_view<E, Rows, Columns>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(E& value, std::size_t rows, std::size_t columns) -> unary_expr<typename E::value_type, dyn_matrix_view<E>, identity_op> {
    cpp_assert(etl_traits<std::decay_t<E>>::size(value) == rows * columns, "Invalid size for reshape");

    return {dyn_matrix_view<E>(value, rows, columns)};
}

template<std::size_t Rows, std::size_t Columns, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(const E& value) -> unary_expr<typename E::value_type, fast_matrix_view<const E, Rows, Columns>, identity_op> {
    cpp_assert(etl_traits<std::decay_t<E>>::size(value) == Rows * Columns, "Invalid size for reshape");

    return {fast_matrix_view<const E, Rows, Columns>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(const E& value, std::size_t rows, std::size_t columns) -> unary_expr<typename E::value_type, dyn_matrix_view<const E>, identity_op> {
    cpp_assert(etl_traits<std::decay_t<E>>::size(value) == rows * columns, "Invalid size for reshape");

    return {dyn_matrix_view<const E>(value, rows, columns)};
}

//}}}

//{{{ Apply a special expression that can change order of elements

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto hflip(const E& value) -> transform_expr<typename E::value_type, hflip_transformer<E>> {
    return {hflip_transformer<E>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto vflip(const E& value) -> transform_expr<typename E::value_type, vflip_transformer<E>> {
    return {vflip_transformer<E>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto fflip(const E& value) -> transform_expr<typename E::value_type, fflip_transformer<E>> {
    return {fflip_transformer<E>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto transpose(const E& value) -> transform_expr<typename E::value_type, transpose_transformer<E>> {
    return {transpose_transformer<E>(value)};
}

//}}}

//{{{ Apply a reduction on an ETL expression (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
typename LE::value_type dot(const LE& lhs, const RE& rhs){
    return sum(lhs * rhs);
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
    typename std::conditional<
        std::is_reference<E>::value,
        typename std::conditional<
            std::is_const<std::remove_reference_t<E>>::value,
            const typename std::decay_t<E>::value_type&,
            typename std::decay_t<E>::value_type&
        >::type,
        typename std::decay_t<E>::value_type
    >::type;
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

} //end of namespace etl

#endif
