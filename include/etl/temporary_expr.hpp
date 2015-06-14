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

// CRTP classes
#include "crtp/comparable.hpp"
#include "crtp/iterable.hpp"

namespace etl {

template <typename D, typename V>
struct temporary_expr : comparable<D>, iterable<D> {
    using         derived_t = D;
    using        value_type = V;
    using       memory_type = value_type*;
    using const_memory_type = const value_type*;

    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    //Apply the expression

    value_type operator[](std::size_t i) const {
        return as_derived().result()[i];
    }

    template<typename... S, cpp_enable_if(sizeof...(S) == sub_size_compare<derived_t>::value)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return as_derived().result()(args...);
    }

    template<typename DD = D, cpp_enable_if((sub_size_compare<DD>::value > 1))>
    auto operator()(std::size_t i) const {
        return sub(as_derived(), i);
    }

    //{{{ Iterator

    iterator<const derived_t> begin() const noexcept {
        return {as_derived(), 0};
    }

    iterator<const derived_t> end() const noexcept {
        return {as_derived(), size(as_derived())};
    }

    // Direct memory access

    memory_type memory_start() noexcept {
        return as_derived().result().memory_start();
    }

    const_memory_type memory_start() const noexcept {
        return as_derived().result().memory_start();
    }

    memory_type memory_end() noexcept {
        return as_derived().result().memory_end();
    }

    const_memory_type memory_end() const noexcept {
        return as_derived().result().memory_end();
    }
};

template <typename T, typename AExpr, typename Op, typename Forced>
struct temporary_unary_expr final : temporary_expr<temporary_unary_expr<T, AExpr, Op, Forced>, T> {
    using        value_type = T;
    using       result_type = std::conditional_t<std::is_same<Forced, void>::value, typename Op::template result_type<AExpr>, Forced>;
    using         data_type = std::conditional_t<std::is_same<Forced, void>::value, std::shared_ptr<result_type>, result_type>;

private:
    static_assert(is_etl_expr<AExpr>::value, "The argument must be an ETL expr");

    using this_type = temporary_unary_expr<T, AExpr, Op, Forced>;

    AExpr _a;
    data_type _c;
    bool evaluated = false;
    bool allocated = false;

public:
    //Construct a new expression
    explicit temporary_unary_expr(AExpr a) : _a(a) {
        //Nothing else to init
    }

    //Construct a new expression
    temporary_unary_expr(AExpr a, std::conditional_t<std::is_same<Forced,void>::value, int, Forced> c) : _a(a), _c(c) {
        //Nothing else to init
    }

    //Copy an expression
    temporary_unary_expr(const temporary_unary_expr& e) : _a(e._a), _c(e._c) {
        //Nothing else to init
    }

    //Move an expression
    temporary_unary_expr(temporary_unary_expr&& e) : _a(e._a), _c(optional_move<std::is_same<Forced,void>::value>(e._c)), evaluated(e.evaluated) {
        e.evaluated = false;
    }

    //Expressions are invariant
    temporary_unary_expr& operator=(const temporary_unary_expr& /*e*/) = delete;
    temporary_unary_expr& operator=(temporary_unary_expr&& /*e*/) = delete;

    //Accessors

    std::add_lvalue_reference_t<AExpr> a(){
        return _a;
    }

    cpp::add_const_lvalue_t<AExpr> a() const {
        return _a;
    }

    void evaluate(){
        if(!evaluated){
            Op::apply(_a, result());
            evaluated = true;
        }
    }

    template<typename Result, typename F = Forced, cpp_disable_if(std::is_same<F, void>::value)>
    void direct_evaluate(Result&& r){
        evaluate();
        r = result();
    }

    template<typename Result, typename F = Forced, cpp_enable_if(std::is_same<F, void>::value)>
    void direct_evaluate(Result&& result){
        Op::apply(_a, std::forward<Result>(result));
    }

    template<typename F = Forced, cpp_disable_if(std::is_same<F, void>::value)>
    void allocate_temporary() const {
        allocated = true;
    }

    template<typename F = Forced, cpp_enable_if(std::is_same<F, void>::value)>
    void allocate_temporary(){
        if(!_c){
            _c.reset(Op::allocate(_a));
        }

        allocated = true;
    }

    using get_result_op = std::conditional_t<std::is_same<Forced, void>::value, dereference_op, forward_op>;

    result_type& result(){
        cpp_assert(evaluated, "The result has not been evaluated");
        cpp_assert(allocated, "The result has not been allocated");
        return get_result_op::apply(_c);
    }

    const result_type& result() const {
        cpp_assert(evaluated, "The result has not been evaluated");
        cpp_assert(allocated, "The result has not been allocated");
        return get_result_op::apply(_c);
    }
};

template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
struct temporary_binary_expr final : temporary_expr<temporary_binary_expr<T, AExpr, BExpr, Op, Forced>, T> {
    using        value_type = T;
    using       result_type = std::conditional_t<std::is_same<Forced, void>::value, typename Op::template result_type<AExpr, BExpr>, Forced>;
    using         data_type = std::conditional_t<std::is_same<Forced, void>::value, std::shared_ptr<result_type>, result_type>;

private:
    static_assert(is_etl_expr<AExpr>::value && is_etl_expr<BExpr>::value, "Both arguments must be ETL expr");

    using this_type = temporary_binary_expr<T, AExpr, BExpr, Op, Forced>;

    AExpr _a;
    BExpr _b;
    data_type _c;
    bool evaluated = false;

public:
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
    temporary_binary_expr& operator=(const temporary_binary_expr& /*e*/) = delete;
    temporary_binary_expr& operator=(temporary_binary_expr&& /*e*/) = delete;

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

    using get_result_op = std::conditional_t<std::is_same<Forced, void>::value, dereference_op, forward_op>;

    result_type& result(){
        return get_result_op::apply(_c);
    }

    const result_type& result() const {
        return get_result_op::apply(_c);
    }
};

template <typename T, typename AExpr, typename Op, typename Forced>
std::ostream& operator<<(std::ostream& os, const temporary_unary_expr<T, AExpr, Op, Forced>& expr){
    return os << Op::desc() << "(" << expr.a() << ")";
}

template <typename T, typename AExpr, typename BExpr, typename Op, typename Forced>
std::ostream& operator<<(std::ostream& os, const temporary_binary_expr<T, AExpr, BExpr, Op, Forced>& expr){
    return os << Op::desc() << "(" << expr.a() << ", " << expr.b() << ")";
}

} //end of namespace etl

#endif
