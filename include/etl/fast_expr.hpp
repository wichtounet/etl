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

struct identity_op;

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class binary_expr {
private:
    static_assert(cpp::or_u<
        cpp::and_u<is_etl_expr<LeftExpr>::value, std::is_same<RightExpr, scalar<T>>::value>::value,
        cpp::and_u<is_etl_expr<RightExpr>::value, std::is_same<LeftExpr, scalar<T>>::value>::value,
        cpp::and_u<is_etl_expr<LeftExpr>::value, is_etl_expr<RightExpr>::value>::value>::value,
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

    value_type operator[](std::size_t i) const {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }

    template<typename... S>
    value_type operator()(S... args) const {
        static_assert(sizeof...(S) == etl_traits<this_type>::dimensions(), "Invalid number of parameters");
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return BinaryOp::apply(lhs()(args...), rhs()(args...));
    }
};

template <typename T, typename Expr, typename UnaryOp>
class unary_expr {
public:
    using value_type = T;

private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, UnaryOp>;

    Expr _value;

public:
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

    value_type operator[](std::size_t i) const {
        return UnaryOp::apply(value()[i]);
    }

    template<typename... S>
    value_type operator()(S... args) const {
        static_assert(sizeof...(S) == etl_traits<this_type>::dimensions(), "Invalid number of parameters");
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return UnaryOp::apply(value()(args...));
    }
};

template <typename T, typename Expr>
class unary_expr<T, Expr, identity_op> {
private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, identity_op>;

    Expr _value;

    static constexpr const bool non_const_return_ref = 
        cpp::and_u<
            std::is_lvalue_reference<decltype(_value[0])>::value,
            cpp::not_u<std::is_const<decltype(_value[0])>::value>::value
        >::value;
    
    static constexpr const bool const_return_ref = 
        std::is_lvalue_reference<decltype(_value[0])>::value;

public:
    using value_type = T;
    using return_type = typename std::conditional<non_const_return_ref, value_type&, value_type>::type;
    using const_return_type = typename std::conditional<const_return_ref, const value_type&, value_type>::type;

    //Cannot be constructed with no args
    unary_expr() = delete;

    //Construct a new expression
    unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    unary_expr(const unary_expr& e) : _value(e._value) {
        //Nothing else to init
    }

    unary_expr(unary_expr&& e) : _value(std::move(e._value)) {
        //Nothing else to init
    }

    //Assign expressions to the unary expr

    unary_expr& operator=(const unary_expr& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(*this); ++i){
            (*this)[i] = e[i];
        }

        return *this;
    }

    template<typename E, cpp::enable_if_u<cpp::and_u<non_const_return_ref, is_etl_expr<std::decay_t<E>>::value>::value> = cpp::detail::dummy>
    unary_expr& operator=(const E& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(*this); ++i){
            (*this)[i] = e[i];
        }

        return *this;
    }

    template<bool B = non_const_return_ref, cpp::enable_if_u<B> = cpp::detail::dummy>
    unary_expr& operator=(const value_type& e){
        for(std::size_t i = 0; i < size(*this); ++i){
            (*this)[i] = e;
        }

        return *this;
    }

    //Accessors

    typename std::add_lvalue_reference<Expr>::type value(){
        return _value;
    }

    typename std::add_lvalue_reference<typename std::add_const<Expr>::type>::type value() const {
        return _value;
    }

    //Apply the expression

    return_type operator[](std::size_t i){
        return value()[i];
    }

    const_return_type operator[](std::size_t i) const {
        return value()[i];
    }
    
    template<typename... S>
    return_type operator()(S... args){
        static_assert(sizeof...(S) == etl_traits<this_type>::dimensions(), "Invalid number of parameters");
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }

    template<typename... S>
    const_return_type operator()(S... args) const {
        static_assert(sizeof...(S) == etl_traits<this_type>::dimensions(), "Invalid number of parameters");
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }
};

template <typename T, typename Expr>
class transform_expr {
private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in transform_expr");

    using this_type = transform_expr<T, Expr>;

    Expr _value;

public:
    using value_type = T;

    //Cannot be constructed with no args
    transform_expr() = delete;

    //Construct a new expression
    transform_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    transform_expr(const transform_expr& e) : _value(e._value) {
        //Nothing else to init
    }

    transform_expr(transform_expr&& e) : _value(e._value) {
        //Nothing else to init
    }

    //Expression are invariant
    transform_expr& operator=(const transform_expr&) = delete;
    transform_expr& operator=(transform_expr&&) = delete;

    //Accessors

    typename std::add_lvalue_reference<Expr>::type value(){
        return _value;
    }

    typename std::add_lvalue_reference<typename std::add_const<Expr>::type>::type value() const {
        return _value;
    }

    //Apply the expression

    value_type operator[](std::size_t i) const {
        return value()[i];
    }

    template<typename... S>
    value_type operator()(S... args) const {
        static_assert(sizeof...(S) == etl_traits<this_type>::dimensions(), "Invalid number of parameters");
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }
};

template <typename Generator>
class generator_expr {
private:
    Generator generator;

public:
    using value_type = typename Generator::value_type;

    template<typename... Args>
    generator_expr(Args... args) : generator(std::forward<Args>(args)...) {}

    generator_expr(generator_expr&& e) : generator(std::move(e.generator)) {
        //Nothing else to init
    }

    //Expression are invariant
    generator_expr& operator=(const generator_expr&) = delete;
    generator_expr& operator=(generator_expr&&) = delete;

    //Apply the expression

    value_type operator[](std::size_t) const {
        return generator();
    }

    value_type operator()() const {
        return generator();
    }
};

//{{{ Build binary expressions from two ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator-(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, minus_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator+(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, plus_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator*(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, mul_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator/(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, div_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator%(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, mod_binary_op<typename LE::value_type>, const RE&> {
    ensure_same_size(lhs, rhs);

    return {lhs, rhs};
}

//}}}

//{{{ Mix scalars and ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = cpp::detail::dummy>
auto operator-(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, minus_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator-(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, minus_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = cpp::detail::dummy>
auto operator+(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, plus_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator+(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, plus_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = cpp::detail::dummy>
auto operator*(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, mul_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}
template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy> auto operator*(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, mul_binary_op<typename RE::value_type>, const RE&> { return {scalar<typename RE::value_type>(lhs), rhs}; }

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = cpp::detail::dummy>
auto operator/(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, div_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator/(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, div_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = cpp::detail::dummy>
auto operator%(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, mod_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, scalar<typename LE::value_type>(rhs)};
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
auto operator%(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, mod_binary_op<typename RE::value_type>, const RE&> {
    return {scalar<typename RE::value_type>(lhs), rhs};
}

//}}}

//{{{ Compound operators

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator+=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] += rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator+=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] += rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator-=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] -= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator-=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] -= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator*=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] *= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator*=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] *= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator/=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] /= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator/=(LE& lhs, const RE& rhs){
    ensure_same_size(lhs, rhs);

    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] /= rhs[i];
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<std::is_arithmetic<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
LE& operator%=(LE& lhs, RE rhs){
    for(std::size_t i = 0; i < lhs.size(); ++i){
        lhs[i] %= rhs;
    }

    return lhs;
}

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<RE>::value, is_etl_value<LE>::value>::value> = cpp::detail::dummy>
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

template<typename E, typename T, cpp::enable_if_u<cpp::and_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value>::value> = cpp::detail::dummy>
auto max(const E& value, T v) -> binary_expr<typename E::value_type, const E&, max_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
}

template<typename E, typename T, cpp::enable_if_u<cpp::and_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value>::value> = cpp::detail::dummy>
auto min(const E& value, T v) -> binary_expr<typename E::value_type, const E&, min_binary_op<typename E::value_type, T>, scalar<T>> {
    return {value, scalar<T>(v)};
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

template<typename E, typename T, cpp::enable_if_u<cpp::and_u<is_etl_expr<E>::value, std::is_arithmetic<T>::value>::value> = cpp::detail::dummy>
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

    return {fast_matrix_view<E, Rows, Columns>(value)};
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
auto reshape(const E& value, std::size_t rows, std::size_t columns) -> unary_expr<typename E::value_type, dyn_matrix_view<const E>, identity_op> {
    cpp_assert(etl_traits<std::decay_t<E>>::size(value) == rows * columns, "Invalid size for reshape");

    return {dyn_matrix_view<E>(value, rows, columns)};
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

template<typename LE, typename RE, cpp::enable_if_u<cpp::and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = cpp::detail::dummy>
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

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
typename E::value_type max(const E& values){
    typename E::value_type m = std::numeric_limits<typename E::value_type>::min();

    for(auto& v : values){
        m = std::max(m, v);
    }

    return m;
}

template<typename E, cpp::enable_if_u<is_etl_expr<E>::value> = cpp::detail::dummy>
typename E::value_type min(const E& values){
    typename E::value_type m = std::numeric_limits<typename E::value_type>::max();

    for(auto& v : values){
        m = std::min(m, v);
    }

    return m;
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
