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

namespace etl {

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class binary_expr {
private:
    static_assert(or_u<
        and_u<is_etl_expr<LeftExpr>::value, std::is_same<RightExpr, scalar<T>>::value>::value,
        and_u<is_etl_expr<RightExpr>::value, std::is_same<LeftExpr, scalar<T>>::value>::value,
        and_u<is_etl_expr<LeftExpr>::value, is_etl_expr<RightExpr>::value>::value>::value,
        "One argument must be an ETL expression and the other one convertible to T");

    using this_type = binary_expr<T, LeftExpr, BinaryOp, RightExpr>;

    LeftExpr _lhs;
    RightExpr _rhs;

public:
    using value_type = T;

    //Cannot be constructed with no args
    binary_expr() = delete;

    //Construct a new expression
    binary_expr(LeftExpr l, RightExpr r) :
            _lhs(std::forward<LeftExpr>(l)), _rhs(std::forward<RightExpr>(r)){
        //Nothing else to init
    }

    //Copy an expression
    binary_expr(const binary_expr& e) : _lhs(e._lhs), _rhs(e._rhs) {
        //Nothing else to init
    }

    //Move an expression
    binary_expr(binary_expr&& e) : _lhs(e._lhs), _rhs(e._rhs) {
        //Nothing else to init
    }

    //Expressions are invariant
    binary_expr& operator=(const binary_expr&) = delete;
    binary_expr& operator=(binary_expr&&) = delete;

    //Accessors

    typename std::add_lvalue_reference<LeftExpr>::type lhs(){
        return _lhs;
    }

    typename std::add_lvalue_reference<typename std::add_const<LeftExpr>::type>::type lhs() const {
        return _lhs;
    }

    typename std::add_lvalue_reference<RightExpr>::type rhs(){
        return _rhs;
    }

    typename std::add_lvalue_reference<typename std::add_const<RightExpr>::type>::type rhs() const {
        return _rhs;
    }

    //Apply the expression

    //TODO The three next functions should be auto return type
    //However, clang++ and g++ do not support that with -g

    T operator[](std::size_t i) const {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }

    T operator()(std::size_t i) const {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }

    template<typename TT = this_type>
    enable_if_t<etl_traits<TT>::is_matrix, T> operator()(std::size_t i, std::size_t j) const {
        return BinaryOp::apply(lhs()(i,j), rhs()(i,j));
    }
};

template <typename T, typename Expr, typename UnaryOp>
class unary_expr {
private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, UnaryOp>;

    Expr _value;

public:
    using value_type = T;

    //Cannot be constructed with no args
    unary_expr() = delete;

    //Construct a new expression
    unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    unary_expr(const unary_expr& e) : _value(e._value) {
        //Nothing else to init
    }

    unary_expr(unary_expr&& e) : _value(e._value) {
        //Nothing else to init
    }

    //Expression are invariant
    unary_expr& operator=(const unary_expr&) = delete;
    unary_expr& operator=(unary_expr&&) = delete;

    //Accessors

    typename std::add_lvalue_reference<Expr>::type value(){
        return _value;
    }

    typename std::add_lvalue_reference<typename std::add_const<Expr>::type>::type value() const {
        return _value;
    }

    //Apply the expression

    //TODO The three next functions should be auto return type
    //However, clang++ and g++ do not support that with -g

    T operator[](std::size_t i) const {
        return UnaryOp::apply(value()[i]);
    }

    T operator()(std::size_t i) const {
        return UnaryOp::apply(value()[i]);
    }

    template<typename TT = this_type>
    enable_if_t<etl_traits<TT>::is_matrix, T> operator()(std::size_t i, std::size_t j) const {
        return UnaryOp::apply(value()(i,j));
    }
};

//{{{ Build binary expressions from two ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator-(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, minus_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator+(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, plus_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator*(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, mul_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator/(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, div_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator%(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, mod_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

//}}}

//{{{ Mix scalars and ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator-(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, minus_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator-(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, minus_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator+(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, plus_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator+(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, plus_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator*(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, mul_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}
template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy> auto operator*(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, mul_binary_op<typename RE::value_type>, const RE&> { return {scalar<typename RE::value_type>(lhs), rhs}; }

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator/(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, div_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator/(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, div_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator%(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, mod_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator%(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, mod_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

//}}}

//{{{ Compound operators

template<typename LE, typename RE, enable_if_u<and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator+=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] += rhs;
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator+=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] += rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator-=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] -= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator-=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] -= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator*=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] *= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator*=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] *= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator/=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] /= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator/=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] /= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator%=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] %= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = detail::dummy>
LE& operator%=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] %= rhs[i];
    }

    return lhs;
}

//}}}

//{{{ Apply an unary expression on an ETL expression (vector,matrix,binary,unary)

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto abs(const E& value) -> unary_expr<typename E::value_type, const E&, abs_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, typename T, enable_if_u<and_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value>::value> = detail::dummy>
auto max(const E& value, T v) -> binary_expr<typename E::value_type, const E&, max_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
}

template<typename E, typename T, enable_if_u<and_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value>::value> = detail::dummy>
auto min(const E& value, T v) -> binary_expr<typename E::value_type, const E&, min_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto log(const E& value) -> unary_expr<typename E::value_type, const E&, log_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto noise(const E& value) -> unary_expr<typename E::value_type, const E&, noise_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto logistic_noise(const E& value) -> unary_expr<typename E::value_type, const E&, logistic_noise_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, typename T, enable_if_u<and_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value>::value> = detail::dummy>
auto ranged_noise(const E& value, T v) -> binary_expr<typename E::value_type, const E&, ranged_noise_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto exp(const E& value) -> unary_expr<typename E::value_type, const E&, exp_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto sign(const E& value) -> unary_expr<typename E::value_type, const E&, sign_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto sigmoid(const E& value) -> unary_expr<typename E::value_type, const E&, sigmoid_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto softplus(const E& value) -> unary_expr<typename E::value_type, const E&, softplus_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto bernoulli(const E& value) -> unary_expr<typename E::value_type, const E&, bernoulli_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto hflip(const E& value) -> unary_expr<typename E::value_type, hflip_transformer<E>, identity_unary_op<typename E::value_type>> {
    return {hflip_transformer<E>(value)};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto vflip(const E& value) -> unary_expr<typename E::value_type, vflip_transformer<E>, identity_unary_op<typename E::value_type>> {
    return {vflip_transformer<E>(value)};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto fflip(const E& value) -> unary_expr<typename E::value_type, fflip_transformer<E>, identity_unary_op<typename E::value_type>> {
    return {fflip_transformer<E>(value)};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto transpose(const E& value) -> unary_expr<typename E::value_type, transpose_transformer<E>, identity_unary_op<typename E::value_type>> {
    return {transpose_transformer<E>(value)};
}

template<std::size_t D, typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto dim(const E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<E, D>, identity_unary_op<typename E::value_type>> {
    return {{value, i}};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto row(const E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<E, 1>, identity_unary_op<typename E::value_type>> {
    return {{value, i}};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto col(const E& value, std::size_t i) -> unary_expr<typename E::value_type, dim_view<E, 2>, identity_unary_op<typename E::value_type>> {
    return {{value, i}};
}

template<std::size_t Rows, std::size_t Columns, typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto reshape(const E& value) -> unary_expr<typename E::value_type, fast_matrix_view<E, Rows, Columns>, identity_unary_op<typename E::value_type>> {
    etl_assert(etl_traits<E>::size(value) == Rows * Columns, "Invalid size for reshape");

    return {fast_matrix_view<E, Rows, Columns>(value)};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto reshape(const E& value, std::size_t rows, std::size_t columns) -> unary_expr<typename E::value_type, dyn_matrix_view<E>, identity_unary_op<typename E::value_type>> {
    etl_assert(etl_traits<E>::size(value) == rows * columns, "Invalid size for reshape");

    return {dyn_matrix_view<E>(value, rows, columns)};
}

//}}}

//{{{ Apply a reduction on an ETL expression (vector,matrix,binary,unary)

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
typename LE::value_type dot(const LE& lhs, const RE& rhs){
    return sum(lhs * rhs);
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
typename E::value_type sum(const E& values){
    typename E::value_type acc(0);

    for(std::size_t i = 0; i < size(values); ++i){
        acc += values[i];
    }

    return acc;
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
typename E::value_type mean(const E& values){
    return sum(values) / size(values);
}

//}}}

} //end of namespace etl

#endif
