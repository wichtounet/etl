//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_FAST_EXPR_HPP
#define ETL_FAST_EXPR_HPP

#include "fast_op.hpp"
#include "tmp.hpp"

namespace etl {

template<typename LE, typename RE, typename Enable = void>
struct get_etl_size ;

template<typename LE, typename RE>
struct get_etl_size<LE, RE, enable_if_t<is_etl_expr<LE>::value>> 
    : std::integral_constant<std::size_t, std::remove_reference<LE>::type::etl_size> {} ;

template<typename LE, typename RE>
struct get_etl_size<LE, RE, enable_if_t<and_u<is_etl_expr<RE>::value, not_u<is_etl_expr<LE>::value>::value>::value>> 
    : std::integral_constant<std::size_t, std::remove_reference<RE>::type::etl_size> {};

template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
class binary_expr {
private:
    LeftExpr _lhs;
    RightExpr _rhs;

    typedef binary_expr<T, LeftExpr, BinaryOp, RightExpr> this_type;

public:
    static constexpr const bool etl_marker = true;
    static constexpr const bool etl_fast = true;
    static constexpr const std::size_t etl_size = get_etl_size<LeftExpr, RightExpr>::value;

    using value_type = T;

    //Cannot be constructed with no args
    binary_expr() = delete;

    //Construct a new expression
    binary_expr(LeftExpr l, RightExpr r) :
            _lhs(std::forward<LeftExpr>(l)), _rhs(std::forward<RightExpr>(r)){
        //Nothing else to init
    }

    //No copying
    binary_expr(const binary_expr&) = delete;
    binary_expr& operator=(const binary_expr&) = delete;

    //Make sure move is supported
    binary_expr(binary_expr&&) = default;
    binary_expr& operator=(binary_expr&&) = default;

    //Accessors

    //TODO size() can be constexpr if Expr is fast_X

    template<typename LE = LeftExpr, enable_if_u<is_etl_expr<LE>::value> = detail::dummy>
    std::size_t size() const {
        return _lhs.size();
    }

    template<typename LE = LeftExpr, disable_if_u<is_etl_expr<LE>::value> = detail::dummy>
    std::size_t size() const {
        return _rhs.size();
    }

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

    decltype(auto) operator[](std::size_t i) const {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }
};

template <typename T, typename Expr, typename UnaryOp>
class unary_expr {
private:
    Expr _value;

    typedef unary_expr<T, Expr, UnaryOp> this_type;

public:
    static constexpr const bool etl_marker = true;
    static constexpr const bool etl_fast = true;
    static constexpr const std::size_t etl_size = std::remove_reference<Expr>::type::etl_size;

    using value_type = T;

    //Cannot be constructed with no args
    unary_expr() = delete;

    //Construct a new expression
    unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    //No copying
    unary_expr(const unary_expr&) = delete;
    unary_expr& operator=(const unary_expr&) = delete;

    //Make sure move is supported
    unary_expr(unary_expr&&) = default;
    unary_expr& operator=(unary_expr&&) = default;

    //Accessors

    //TODO size() can be constexpr if Expr is fast_X

    std::size_t size() const {
        return _value.size();
    }

    typename std::add_lvalue_reference<Expr>::type value(){
        return _value;
    }

    typename std::add_lvalue_reference<typename std::add_const<Expr>::type>::type value() const {
        return _value;
    }

    //Apply the expression

    decltype(auto) operator[](std::size_t i) const {
        return UnaryOp::apply(value()[i]);
    }
};

//{{{ Build binary expressions from two ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator-(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, minus_binary_op<typename LE::value_type>, const RE&> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator+(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, plus_binary_op<typename LE::value_type>, const RE&> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator*(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, mul_binary_op<typename LE::value_type>, const RE&> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator/(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, div_binary_op<typename LE::value_type>, const RE&> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<is_etl_expr<LE>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator%(const LE& lhs, const RE& rhs) -> binary_expr<typename LE::value_type, const LE&, mod_binary_op<typename LE::value_type>, const RE&> {
    return {lhs, rhs};
}

//}}}

//{{{ Mix scalars and ETL expressions (vector,matrix,binary,unary)

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator-(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, minus_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator-(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, minus_binary_op<typename RE::value_type>, const RE&> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator+(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, plus_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator+(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, plus_binary_op<typename RE::value_type>, const RE&> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator*(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, mul_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator*(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, mul_binary_op<typename RE::value_type>, const RE&> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator/(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, div_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator/(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, div_binary_op<typename RE::value_type>, const RE&> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<RE, typename LE::value_type>::value, is_etl_expr<LE>::value>::value> = detail::dummy>
auto operator%(const LE& lhs, RE rhs) -> binary_expr<typename LE::value_type, const LE&, mod_binary_op<typename LE::value_type>, scalar<typename LE::value_type>> {
    return {lhs, rhs};
}

template<typename LE, typename RE, enable_if_u<and_u<std::is_convertible<LE, typename RE::value_type>::value, is_etl_expr<RE>::value>::value> = detail::dummy>
auto operator%(LE lhs, const RE& rhs) -> binary_expr<typename RE::value_type, scalar<typename RE::value_type>, mod_binary_op<typename RE::value_type>, const RE&> {
    return {lhs, rhs};
}

//}}}

//{{{ Apply an unary expression on an ETL expression (vector,matrix,binary,unary)

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto abs(const E& value) -> unary_expr<typename E::value_type, const E&, abs_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto log(const E& value) -> unary_expr<typename E::value_type, const E&, log_unary_op<typename E::value_type>> {
    return {value};
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
auto sign(const E& value) -> unary_expr<typename E::value_type, const E&, sign_unary_op<typename E::value_type>> {
    return {value};
}

//}}}

//{{{ Apply a reduction on an ETL expression (vector,matrix,binary,unary)

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
typename E::value_type sum(const E& values){
    auto acc = static_cast<typename E::value_type>(0);

    for(std::size_t i = 0; i < values.size(); ++i){
        acc += values[i];
    }

    return acc;
}

template<typename E, enable_if_u<is_etl_expr<E>::value> = detail::dummy>
typename E::value_type mean(const E& values){
    return sum(values) / values.size();
}

//}}}

} //end of namespace etl

#endif
