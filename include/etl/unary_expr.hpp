//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_UNARY_EXPR_HPP
#define ETL_UNARY_EXPR_HPP

#include <iostream>     //For stream support

#include "cpp_utils/assert.hpp"

#include "traits_lite.hpp"
#include "iterator.hpp"

// CRTP classes
#include "crtp/comparable.hpp"
#include "crtp/iterable.hpp"
#include "crtp/inplace_assignable.hpp"

namespace etl {

struct identity_op;
struct virtual_op;

template <typename Generator>
class generator_expr;

template <typename T, typename Expr, typename UnaryOp>
struct unary_expr final : comparable<unary_expr<T, Expr, UnaryOp>>, iterable<unary_expr<T, Expr, UnaryOp>> {
private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, UnaryOp>;

    Expr _value;

public:
    using        value_type = T;
    using       memory_type = void;
    using const_memory_type = void;

    //Cannot be constructed with no args
    unary_expr() = delete;

    //Construct a new expression
    unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& rhs) = delete;
    unary_expr& operator=(unary_expr&& rhs) = delete;

    //Accessors

    std::add_lvalue_reference_t<Expr> value() noexcept {
        return _value;
    }

    cpp::add_const_lvalue_t<Expr> value() const noexcept {
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

    iterator<const this_type> begin() const noexcept {
        return {*this, 0};
    }

    iterator<const this_type> end() const noexcept {
        return {*this, size(*this)};
    }
};

template <typename T, typename Expr>
struct unary_expr<T, Expr, identity_op> : inplace_assignable<unary_expr<T, Expr, identity_op>>, comparable<unary_expr<T, Expr, identity_op>>, iterable<unary_expr<T, Expr, identity_op>>  {
private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, identity_op>;

    Expr _value;

    static constexpr const bool non_const_return_ref =
        cpp::and_c<
            std::is_lvalue_reference<decltype(_value[0])>,
            cpp::not_c<std::is_const<std::remove_reference_t<decltype(_value[0])>>>
        >::value;

    static constexpr const bool const_return_ref =
        std::is_lvalue_reference<decltype(_value[0])>::value;

public:
    using        value_type = T;
    using       memory_type = memory_t<Expr>;
    using const_memory_type = std::add_const_t<memory_t<Expr>>;
    using       return_type = std::conditional_t<non_const_return_ref, value_type&, value_type>;
    using const_return_type = std::conditional_t<const_return_ref, const value_type&, value_type>;

    //Cannot be constructed with no args
    unary_expr() = delete;

    //Construct a new expression
    unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    //Assign expressions to the unary expr

    template<typename E, cpp::enable_if_all_u<non_const_return_ref, is_copy_expr<E>::value> = cpp::detail::dummy>
    unary_expr& operator=(E&& e){
        ensure_same_size(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
        return *this;
    }

    template<typename Generator>
    unary_expr& operator=(generator_expr<Generator>&& e){
        assign_evaluate(e, *this);
        return *this;
    }

    template<bool B = non_const_return_ref, cpp::enable_if_u<B> = cpp::detail::dummy>
    unary_expr& operator=(const value_type& e){
        for(std::size_t i = 0; i < size(*this); ++i){
            (*this)[i] = e;
        }

        return *this;
    }

    template<typename Container, cpp::enable_if_all_c<cpp::not_c<is_etl_expr<Container>>, std::is_convertible<typename Container::value_type, value_type>> = cpp::detail::dummy>
    unary_expr& operator=(const Container& vec){
        cpp_assert(vec.size() == size(*this), "Cannot copy from a vector of different size");

        for(std::size_t i = 0; i < size(*this); ++i){
            (*this)[i] = vec[i];
        }

        return *this;
    }

    //Accessors

    std::add_lvalue_reference_t<Expr> value() noexcept {
        return _value;
    }

    cpp::add_const_lvalue_t<Expr> value() const noexcept {
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

    iterator<this_type, non_const_return_ref, false> begin() noexcept {
        return {*this, 0};
    }

    iterator<this_type, non_const_return_ref, false> end() noexcept {
        return {*this, size(*this)};
    }

    iterator<const this_type, true> begin() const noexcept {
        return {*this, 0};
    }

    iterator<const this_type, true> end() const noexcept {
        return {*this, size(*this)};
    }

    //{{{ Direct memory access

    template<typename SS = Expr, cpp_enable_if(has_direct_access<SS>::value)>
    memory_type memory_start() noexcept {
        return value().memory_start();
    }

    template<typename SS = Expr, cpp_enable_if(has_direct_access<SS>::value)>
    const_memory_type memory_start() const noexcept {
        return value().memory_start();
    }

    template<typename SS = Expr, cpp_enable_if(has_direct_access<SS>::value)>
    memory_type memory_end() noexcept {
        return value().memory_end();
    }

    template<typename SS = Expr, cpp_enable_if(has_direct_access<SS>::value)>
    const_memory_type memory_end() const noexcept {
        return value().memory_end();
    }

    //}}}
};

template <typename T, typename Expr>
struct unary_expr<T, Expr, virtual_op> : comparable<unary_expr<T, Expr, virtual_op>>, iterable<unary_expr<T, Expr, virtual_op>>  {
private:
    using this_type = unary_expr<T, Expr, virtual_op>;

    Expr _value;

public:
    using        value_type = T;
    using       memory_type = void;
    using const_memory_type = void;

    //Cannot be constructed with no args
    unary_expr() = delete;

    //Construct a new expression
    unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr&) = delete;
    unary_expr& operator=(unary_expr&&) = delete;

    //Accessors

    std::add_lvalue_reference_t<Expr> value() noexcept {
        return _value;
    }

    cpp::add_const_lvalue_t<Expr> value() const noexcept {
        return _value;
    }

    //Apply the expression

    value_type operator[](std::size_t i) const {
        return value()[i];
    }

    template<bool B = (sub_size_compare<this_type>::value > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    value_type operator()(std::size_t i) const {
        return sub(*this, i);
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }

    iterator<const this_type, false, false> begin() const noexcept {
        return {*this, 0};
    }

    iterator<const this_type, false, false> end() const noexcept {
        return {*this, size(*this)};
    }
};

template <typename T, typename Expr>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, identity_op>& expr){
    return os << expr.value();
}

template <typename T, typename Expr, typename UnaryOp>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, UnaryOp>& expr){
    return os << UnaryOp::desc() << '(' << expr.value() << ')';
}

} //end of namespace etl

#endif
