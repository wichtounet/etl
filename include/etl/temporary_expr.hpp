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

#include "traits_fwd.hpp"
#include "iterator.hpp"

namespace etl {

//TODO Review constness of this class

template <typename T, typename AExpr, typename BExpr, typename Op>
class temporary_binary_expr final {
public:
    using value_type = T;
    using result_type = typename Op::template result_type<AExpr, BExpr>;

private:
    static_assert(cpp::and_c<is_etl_expr<AExpr>, is_etl_expr<BExpr>>::value,
        "Both arguments must be ETL expr");

    using this_type = temporary_binary_expr<T, AExpr, BExpr, Op>;

    AExpr _a;
    BExpr _b;
    mutable result_type* result_ptr = nullptr;
    mutable bool temporary = true;
    mutable bool evaluated = false;

public:
    //Cannot be constructed with no args
    temporary_binary_expr() = delete;

    //Construct a new expression
    temporary_binary_expr(AExpr a, BExpr b) : _a(a), _b(b) {
        //Nothing else to init
    }

    //Copy an expression
    temporary_binary_expr(const temporary_binary_expr& e) : _a(e._a), _b(e._b) {
        //Nothing else to init
    }

    //Move an expression
    temporary_binary_expr(temporary_binary_expr&& e) : _a(e._a), _b(e._b) {
        //Nothing else to init
    }

    //Expressions are invariant
    temporary_binary_expr& operator=(const temporary_binary_expr&) = delete;
    temporary_binary_expr& operator=(temporary_binary_expr&&) = delete;

    ~temporary_binary_expr(){
        if(temporary){
            if(result_ptr){
                delete result_ptr;
            }
        }
    }

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

    value_type operator[](std::size_t i) const {
        return result()[i];
    }

    template<typename... S, cpp_enable_if(sizeof...(S) == sub_size_compare<this_type>::value)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return result()(args...);
    }

    template<cpp_enable_if_cst(sub_size_compare<this_type>::value > 1)>
    auto operator()(std::size_t i){
        return sub(*this, i);
    }

    template<cpp_enable_if_cst(sub_size_compare<this_type>::value > 1)>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
    }

    iterator<const this_type> begin() const noexcept {
        return {*this, 0};
    }

    iterator<const this_type> end() const noexcept {
        return {*this, size(*this)};
    }

    void evaluate() const {
        if(!evaluated){
            //Note: This is necessary to allow direct to the expression wihout passing by the evaluator
            if(cpp_unlikely(!result_ptr)){
                allocate_temporary();
            }

            Op::apply(a(), b(), *result_ptr);
            evaluated = true;
        }
    }

    template<typename Result>
    void direct_evaluate(Result&& result) const {
        Op::apply(a(), b(), std::forward<Result>(result));
    }

    void allocate_temporary() const {
        result_ptr = Op::allocate(_a, _b);
        temporary = true;
    }

    void give_result(result_type* r) const {
        result_ptr = r;
        temporary = false;
    }

private:
    result_type& result() const {
        //Note: This is necessary to allow direct to the expression wihout passing by the evaluator
        if(cpp_unlikely(!evaluated)){
            evaluate();
        }

        return *result_ptr;
    }
};

template <typename T, typename AExpr, typename BExpr, typename Op>
std::ostream& operator<<(std::ostream& os, const temporary_binary_expr<T, AExpr, BExpr, Op>& expr){
    return os << Op::desc() << "(" << expr.a() << ", " << expr.b() << ")";
}

} //end of namespace etl

#endif
