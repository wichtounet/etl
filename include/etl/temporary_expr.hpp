//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_TEMPORARY_BINARY_EXPR_HPP
#define ETL_TEMPORARY_BINARY_EXPR_HPP

#include <iostream>     //For stream support
#include <ostream>
#include <memory>       //For shared_ptr

#include "traits_lite.hpp"
#include "iterator.hpp"
#include "tmp.hpp"

#include "comparable.hpp"
#include "iterable.hpp"

namespace etl {

template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
struct temporary_binary_expr final : comparable<temporary_binary_expr<T, AExpr, BExpr, Op, Forced>>, iterable<temporary_binary_expr<T, AExpr, BExpr, Op, Forced>> {
    using        value_type = T;
    using       result_type = std::conditional_t<std::is_same<Forced, void>::value, typename Op::template result_type<AExpr, BExpr>, Forced>;
    using         data_type = std::conditional_t<std::is_same<Forced, void>::value, std::shared_ptr<result_type>, result_type>;
    using       memory_type = value_type*;
    using const_memory_type = const value_type*;

private:
    static_assert(cpp::and_c<is_etl_expr<AExpr>, is_etl_expr<BExpr>>::value, "Both arguments must be ETL expr");

    using this_type = temporary_binary_expr<T, AExpr, BExpr, Op, Forced>;

    AExpr _a;
    BExpr _b;
    data_type _c;
    bool evaluated = false;

public:
    //Cannot be constructed with no args
    temporary_binary_expr() = delete;

    //Construct a new expression
    temporary_binary_expr(AExpr a, BExpr b) : _a(a), _b(b) {
        //Nothing else to init
    }

    //Construct a new expression
    temporary_binary_expr(AExpr a, BExpr b, std::conditional_t<std::is_same<Forced,void>::value, int, Forced> c) : _a(a), _b(b), _c(c) {
        //Nothing else to init
    }

    //Copy an expression
    temporary_binary_expr(const temporary_binary_expr& e) : _a(e._a), _b(e._b), _c(e._c) {
        //Nothing else to init
    }

    //Move an expression
    temporary_binary_expr(temporary_binary_expr&& e) : _a(e._a), _b(e._b), _c(optional_move<std::is_same<Forced,void>::value>(e._c)), evaluated(e.evaluated) {
        e.evaluated = false;
    }

    //Expressions are invariant
    temporary_binary_expr& operator=(const temporary_binary_expr&) = delete;
    temporary_binary_expr& operator=(temporary_binary_expr&&) = delete;

    //Accessors

    std::add_lvalue_reference_t<AExpr> a(){
        return _a;
    }

    cpp::add_const_lvalue_t<AExpr> a() const {
        return _a;
    }

    std::add_lvalue_reference_t<BExpr> b(){
        return _b;
    }

    cpp::add_const_lvalue_t<BExpr> b() const {
        return _b;
    }

    //Apply the expression

    value_type operator[](std::size_t i){
        return result()[i];
    }

    value_type operator[](std::size_t i) const {
        return result()[i];
    }

    template<typename... S, cpp_enable_if(sizeof...(S) == sub_size_compare<this_type>::value)>
    value_type operator()(S... args){
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return result()(args...);
    }

    template<typename ST = T, typename A = AExpr, typename B = BExpr, typename O = Op, typename F = Forced, cpp_enable_if((sub_size_compare<temporary_binary_expr<ST, A, B, O, F>>::value > 1))>
    auto operator()(std::size_t i){
        return sub(*this, i);
    }

    iterator<const this_type> begin() const noexcept {
        return {*this, 0};
    }

    iterator<const this_type> end() const noexcept {
        return {*this, size(*this)};
    }

    void evaluate(){
        if(!evaluated){
            Op::apply(_a, _b, result());
            evaluated = true;
        }
    }

    template<typename Result, typename F = Forced, cpp_disable_if(std::is_same<F, void>::value)>
    void direct_evaluate(Result&& r){
        evaluate();

        //TODO Normally, this should not be necessary
        if(&r != &result()){
            r = result();
        }
    }

    template<typename Result, typename F = Forced, cpp_enable_if(std::is_same<F, void>::value)>
    void direct_evaluate(Result&& result){
        Op::apply(_a, _b, std::forward<Result>(result));
    }

    template<typename F = Forced, cpp_disable_if(std::is_same<F, void>::value)>
    void allocate_temporary() const {
        //NOP
    }

    template<typename F = Forced, cpp_enable_if(std::is_same<F, void>::value)>
    void allocate_temporary(){
        if(!_c){
            _c.reset(Op::allocate(_a, _b));
        }
    }

    //{{{ Direct memory access

    memory_type memory_start() noexcept {
        return result().memory_start();
    }

    const_memory_type memory_start() const noexcept {
        return result().memory_start();
    }

    memory_type memory_end() noexcept {
        return result().memory_end();
    }

    const_memory_type memory_end() const noexcept {
        return result().memory_end();
    }

    //}}}

private:
    using get_result_op = std::conditional_t<std::is_same<Forced, void>::value, dereference_op, forward_op>;

    result_type& result(){
        return get_result_op::apply(_c);
    }

    const result_type& result() const {
        return get_result_op::apply(_c);
    }
};

template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
std::ostream& operator<<(std::ostream& os, const temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& expr){
    return os << Op::desc() << "(" << expr.a() << ", " << expr.b() << ")";
}

} //end of namespace etl

#endif
