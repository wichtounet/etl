//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_BINARY_EXPR_HPP
#define ETL_BINARY_EXPR_HPP

#include "traits_fwd.hpp"

namespace etl {

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
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return BinaryOp::apply(lhs()(args...), rhs()(args...));
    }
};

} //end of namespace etl

#endif
