//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_UNARY_EXPR_HPP
#define ETL_UNARY_EXPR_HPP

#include "cpp_utils/assert.hpp"

#include "traits_fwd.hpp"
#include "iterator.hpp"

namespace etl {

struct identity_op;

template <typename T, typename Expr, typename UnaryOp>
class unary_expr final  {
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
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return UnaryOp::apply(value()(args...));
    }

    iterator<const this_type> begin() const {
        return {*this, 0};
    }

    iterator<const this_type> end() const {
        return {*this, size(*this)};
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
            cpp::not_u<std::is_const<std::remove_reference_t<decltype(_value[0])>>::value>::value
        >::value;

    static constexpr const bool const_return_ref =
        std::is_lvalue_reference<decltype(_value[0])>::value;

public:
    using value_type = T;
    using return_type = std::conditional_t<non_const_return_ref, value_type&, value_type>;
    using const_return_type = std::conditional_t<const_return_ref, const value_type&, value_type>;

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

    //Assign expressions to the unary expr

    unary_expr& operator=(const unary_expr& e){
        ensure_same_size(*this, e);

        for(std::size_t i = 0; i < size(*this); ++i){
            (*this)[i] = e[i];
        }

        return *this;
    }

    template<typename E, cpp::enable_if_all_u<non_const_return_ref, is_copy_expr<E>::value> = cpp::detail::dummy>
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

    template<typename Container, cpp::enable_if_all_u<cpp::not_u<is_etl_expr<Container>::value>::value, std::is_convertible<typename Container::value_type, value_type>::value> = cpp::detail::dummy>
    unary_expr& operator=(const Container& vec){
        cpp_assert(vec.size() == size(*this), "Cannot copy from a vector of different size");

        for(std::size_t i = 0; i < size(*this); ++i){
            (*this)[i] = vec[i];
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

    template<bool B = (sub_size_compare<this_type>::value > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i){
        return sub(*this, i);
    }

    template<bool B = (sub_size_compare<this_type>::value > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, return_type> operator()(S... args){
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, const_return_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }

    iterator<this_type, non_const_return_ref, false> begin(){
        return {*this, 0};
    }

    iterator<this_type, non_const_return_ref, false> end(){
        return {*this, size(*this)};
    }

    iterator<const this_type, true> begin() const {
        return {*this, 0};
    }

    iterator<const this_type, true> end() const {
        return {*this, size(*this)};
    }
};

} //end of namespace etl

#endif
