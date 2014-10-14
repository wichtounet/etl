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

template<typename T, typename Enable = void>
struct build_type {
    using type = std::decay_t<T>;
};

template<typename T>
struct build_type<T, std::enable_if_t<is_etl_value<std::decay_t<T>>::value>> {
    using type = const std::decay_t<T>&;
};

//{{{ Build binary expressions from two ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, template<typename> class OP>
struct left_binary_helper {
    using type = binary_expr<typename std::decay_t<LE>::value_type, typename build_type<LE>::type, OP<typename std::decay_t<LE>::value_type>, typename build_type<RE>::type>;
};

template<typename LE, typename RE, typename OP>
struct left_binary_helper_op {
    using type = binary_expr<typename std::decay_t<LE>::value_type, typename build_type<LE>::type, OP, typename build_type<RE>::type>;
};

template<typename LE, typename RE, template<typename> class OP>
struct right_binary_helper {
    using type = binary_expr<typename std::decay_t<RE>::value_type, typename build_type<LE>::type, OP<typename std::decay_t<RE>::value_type>, typename build_type<RE>::type>;
};

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator-(LE&& lhs, RE&& rhs) -> typename left_binary_helper<LE, RE, minus_binary_op>::type {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator+(LE&& lhs, RE&& rhs) -> typename left_binary_helper<LE, RE, plus_binary_op>::type {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator*(LE&& lhs, RE&& rhs) -> typename left_binary_helper<LE, RE, mul_binary_op>::type {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator/(LE&& lhs, RE&& rhs) -> typename left_binary_helper<LE, RE, div_binary_op>::type {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value> = cpp::detail::dummy>
auto operator%(LE&& lhs, RE&& rhs) -> typename left_binary_helper<LE, RE, mod_binary_op>::type {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

//}}}

//{{{ Mix scalars and ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, typename T = typename std::decay_t<LE>::value_type, cpp::enable_if_all_u<std::is_convertible<RE, T>::value, is_etl_expr<std::decay_t<LE>>::value> = cpp::detail::dummy>
auto operator-(LE&& lhs, RE rhs) -> typename left_binary_helper<LE, scalar<T>, minus_binary_op>::type {
    return {lhs, scalar<T>(rhs)};
}

template<typename LE, typename RE, typename T = typename std::decay_t<RE>::value_type, cpp::enable_if_all_u<std::is_convertible<LE, T>::value, is_etl_expr<std::decay_t<RE>>::value> = cpp::detail::dummy>
auto operator-(LE lhs, RE&& rhs) -> typename right_binary_helper<scalar<T>, RE, minus_binary_op>::type {
    return {scalar<T>(lhs), rhs};
}

template<typename LE, typename RE, typename T = typename std::decay_t<LE>::value_type, cpp::enable_if_all_u<std::is_convertible<RE, T>::value, is_etl_expr<std::decay_t<LE>>::value> = cpp::detail::dummy>
auto operator+(LE&& lhs, RE rhs) -> typename left_binary_helper<LE, scalar<T>, plus_binary_op>::type {
    return {lhs, scalar<T>(rhs)};
}

template<typename LE, typename RE, typename T = typename std::decay_t<RE>::value_type, cpp::enable_if_all_u<std::is_convertible<LE, T>::value, is_etl_expr<std::decay_t<RE>>::value> = cpp::detail::dummy>
auto operator+(LE lhs, RE&& rhs) -> typename right_binary_helper<scalar<T>, RE, plus_binary_op>::type {
    return {scalar<T>(lhs), rhs};
}

template<typename LE, typename RE, typename T = typename std::decay_t<LE>::value_type, cpp::enable_if_all_u<std::is_convertible<RE, T>::value, is_etl_expr<std::decay_t<LE>>::value> = cpp::detail::dummy>
auto operator*(LE&& lhs, RE rhs) -> typename left_binary_helper<LE, scalar<T>, mul_binary_op>::type {
    return {lhs, scalar<T>(rhs)};
}

template<typename LE, typename RE, typename T = typename std::decay_t<RE>::value_type, cpp::enable_if_all_u<std::is_convertible<LE, T>::value, is_etl_expr<std::decay_t<RE>>::value> = cpp::detail::dummy>
auto operator*(LE lhs, RE&& rhs) -> typename right_binary_helper<scalar<T>, RE, mul_binary_op>::type {
    return {scalar<T>(lhs), rhs};
}

template<typename LE, typename RE, typename T = typename std::decay_t<LE>::value_type, cpp::enable_if_all_u<std::is_convertible<RE, T>::value, is_etl_expr<std::decay_t<LE>>::value> = cpp::detail::dummy>
auto operator/(LE&& lhs, RE rhs) -> typename left_binary_helper<LE, scalar<T>, div_binary_op>::type {
    return {lhs, scalar<T>(rhs)};
}

template<typename LE, typename RE, typename T = typename std::decay_t<RE>::value_type, cpp::enable_if_all_u<std::is_convertible<LE, T>::value, is_etl_expr<std::decay_t<RE>>::value> = cpp::detail::dummy>
auto operator/(LE lhs, RE&& rhs) -> typename right_binary_helper<scalar<T>, RE, div_binary_op>::type {
    return {scalar<T>(lhs), rhs};
}

template<typename LE, typename RE, typename T = typename std::decay_t<LE>::value_type, cpp::enable_if_all_u<std::is_convertible<RE, T>::value, is_etl_expr<std::decay_t<LE>>::value> = cpp::detail::dummy>
auto operator%(LE&& lhs, RE rhs) -> typename left_binary_helper<LE, scalar<T>, mod_binary_op>::type {
    return {lhs, scalar<T>(rhs)};
}

template<typename LE, typename RE, typename T = typename std::decay_t<RE>::value_type, cpp::enable_if_all_u<std::is_convertible<LE, T>::value, is_etl_expr<std::decay_t<RE>>::value> = cpp::detail::dummy>
auto operator%(LE lhs, RE&& rhs) -> typename right_binary_helper<scalar<T>, RE, mod_binary_op>::type {
    return {scalar<T>(lhs), rhs};
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
LE& operator+=(LE&& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] += rhs[i];
    }

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
LE& operator-=(LE&& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] -= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<std::is_arithmetic<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator*=(LE&& lhs, RE rhs){
    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] *= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_all_u<is_etl_expr<RE>::value, is_etl_assignable<LE>::value> = cpp::detail::dummy>
LE& operator*=(LE&& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] *= rhs[i];
    }

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
LE& operator/=(LE&& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] /= rhs[i];
    }

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
LE& operator%=(LE&& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < size(lhs); ++i){
        lhs[i] %= rhs[i];
    }

    return lhs;
}

//}}}

//{{{ Apply an unary expression on an ETL expression (vector,matrix,binary,unary)

template<typename E, template<typename> class OP>
struct unary_helper {
    using type = unary_expr<typename std::decay_t<E>::value_type, typename build_type<E>::type, OP<typename std::decay_t<E>::value_type>>;
};

template<typename E, template<typename> class OP>
using unary_helper_t = typename unary_helper<E, OP>::type;

template<typename E, cpp::enable_if_u<is_etl_expr<std::decay_t<E>>::value> = cpp::detail::dummy>
auto abs(E&& value) -> unary_helper_t<E, abs_unary_op> {
    return {value};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto max(E&& value, T v) -> typename left_binary_helper_op<E, scalar<T>, max_binary_op<typename std::decay_t<E>::value_type, T>>::type {
    return {value, scalar<T>(v)};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto min(E&& value, T v) -> typename left_binary_helper_op<E, scalar<T>, min_binary_op<typename std::decay_t<E>::value_type, T>>::type {
    return {value, scalar<T>(v)};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto one_if(E&& value, T v) -> typename left_binary_helper_op<E, scalar<T>, one_if_binary_op<typename std::decay_t<E>::value_type, T>>::type {
    return {value, scalar<T>(v)};
}

template<typename E, typename T = typename std::decay_t<E>::value_type, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto one_if_max(E&& value) -> typename left_binary_helper_op<E, scalar<T>, one_if_binary_op<T, T>>::type {
    return {value, scalar<T>(max(value))};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto log(E&& value) -> unary_helper_t<E, log_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto uniform_noise(E&& value) -> unary_helper_t<E, uniform_noise_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto normal_noise(E&& value) -> unary_helper_t<E, normal_noise_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto logistic_noise(E&& value) -> unary_helper_t<E, logistic_noise_unary_op> {
    return {value};
}

template<typename E, typename T, cpp::enable_if_all_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value> = cpp::detail::dummy>
auto ranged_noise(E&& value, T v) -> typename left_binary_helper_op<E, scalar<T>, ranged_noise_binary_op<typename E::value_type, T>>::type {
    return {value, scalar<T>(v)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto exp(E&& value) -> unary_helper_t<E, exp_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sign(E&& value) -> unary_helper_t<E, sign_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sigmoid(E&& value) -> unary_helper_t<E, sigmoid_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto softplus(E&& value) -> unary_helper_t<E, softplus_unary_op> {
    return {value};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto softmax(const E& e){
    return exp(e) / sum(exp(e));
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto bernoulli(const E& value) -> unary_helper_t<E, bernoulli_unary_op> {
    return {value};
}

//}}}

//{{{ Views that returns lvalues

template<typename E, typename OP>
using identity_helper = unary_expr<typename std::decay_t<E>::value_type, OP, identity_op>;

template<typename T, typename Enable = void>
struct build_identity_type {
    using type = std::decay_t<T>;
};

template<typename T>
struct build_identity_type<T, std::enable_if_t<is_etl_value<std::decay_t<T>>::value>> {
    using type = std::conditional_t<
            std::is_const<std::remove_reference_t<T>>::value,
            const std::decay_t<T>&,
            std::decay_t<T>&
        >;
};

template<std::size_t D, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto dim(E&& value, std::size_t i) -> identity_helper<E, dim_view<typename build_identity_type<E>::type, D>> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto row(E&& value, std::size_t i) -> identity_helper<E, dim_view<typename build_identity_type<E>::type, 1>> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto col(E&& value, std::size_t i) -> identity_helper<E, dim_view<typename build_identity_type<E>::type, 2>> {
    return {{value, i}};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto sub(E&& value, std::size_t i) -> identity_helper<E, sub_view<typename build_identity_type<E>::type>> {
    static_assert(etl_traits<std::decay_t<E>>::dimensions() > 1, "Cannot use sub on vector");
    return {{value, i}};
}

template<std::size_t Rows, std::size_t Columns, typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(E&& value) -> identity_helper<E, fast_matrix_view<typename build_identity_type<E>::type, Rows, Columns>> {
    cpp_assert(etl_traits<std::decay_t<E>>::size(value) == Rows * Columns, "Invalid size for reshape");

    return {fast_matrix_view<typename build_identity_type<E>::type, Rows, Columns>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(E&& value, std::size_t rows, std::size_t columns) -> identity_helper<E, dyn_matrix_view<typename build_identity_type<E>::type>> {
    cpp_assert(etl_traits<std::decay_t<E>>::size(value) == rows * columns, "Invalid size for reshape");

    return {dyn_matrix_view<E>(value, rows, columns)};
}

//}}}

//{{{ Apply a special expression that can change order of elements

template<typename E, template<typename> class OP>
using transform_helper = transform_expr<typename std::decay_t<E>::value_type, OP<typename build_type<E>::type>>;

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto hflip(const E& value) -> transform_helper<E, hflip_transformer> {
    return {hflip_transformer<typename build_type<E>::type>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto vflip(const E& value) -> transform_helper<E, vflip_transformer> {
    return {vflip_transformer<typename build_type<E>::type>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto fflip(const E& value) -> transform_helper<E, fflip_transformer> {
    return {fflip_transformer<typename build_type<E>::type>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto transpose(const E& value) -> transform_helper<E, transpose_transformer> {
    return {transpose_transformer<typename build_type<E>::type>(value)};
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
