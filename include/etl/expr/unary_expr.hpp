//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <iosfwd>     //For stream support

#include "cpp_utils/assert.hpp"

#include "etl/traits_lite.hpp"
#include "etl/iterator.hpp"

// CRTP classes
#include "etl/crtp/comparable.hpp"
#include "etl/crtp/value_testable.hpp"
#include "etl/crtp/dim_testable.hpp"
#include "etl/crtp/inplace_assignable.hpp"

namespace etl {

struct identity_op {
    static constexpr const bool vectorizable = true;
};

struct transform_op {
    static constexpr const bool vectorizable = false;
};

template<typename Sub>
struct stateful_op {
    static constexpr const bool vectorizable = Sub::vectorizable;
    using op = Sub;
};

template <typename Generator>
class generator_expr;

template <typename T, typename Expr, typename UnaryOp>
struct unary_expr final :
          comparable<unary_expr<T, Expr, UnaryOp>>
        , value_testable<unary_expr<T, Expr, UnaryOp>>
        , dim_testable<unary_expr<T, Expr, UnaryOp>>
        {
private:
    static_assert(
        is_etl_expr<Expr>::value || std::is_same<Expr, etl::scalar<T>>::value,
        "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, UnaryOp>;

    Expr _value;

public:
    using        value_type = T;
    using       memory_type = void;
    using const_memory_type = void;

    //Construct a new expression
    explicit unary_expr(Expr l) : _value(std::forward<Expr>(l)){
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

    intrinsic_type<value_type> load(std::size_t i) const {
        return UnaryOp::load(value().load(i));
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
struct unary_expr<T, Expr, identity_op> :
          inplace_assignable<unary_expr<T, Expr, identity_op>>
        , comparable<unary_expr<T, Expr, identity_op>>
        , value_testable<unary_expr<T, Expr, identity_op>>
        , dim_testable<unary_expr<T, Expr, identity_op>>
        {
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
    using          vec_type = intrinsic_type<T>;

    //Construct a new expression
    explicit unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    //Assign expressions to the unary expr

    template<typename E, cpp_enable_if(non_const_return_ref && is_etl_expr<E>::value)>
    unary_expr& operator=(E&& e){
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
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
        validate_assign(*this, vec);

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

    template<typename SS = Expr, cpp_enable_if(has_direct_access<SS>::value)>
    vec_type load(std::size_t i) const noexcept {
        return vec::loadu(memory_start() + i);
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
struct unary_expr<T, Expr, transform_op> :
        comparable<unary_expr<T, Expr, transform_op>>
        , value_testable<unary_expr<T, Expr, transform_op>>
        , dim_testable<unary_expr<T, Expr, transform_op>>
        {
private:
    using this_type = unary_expr<T, Expr, transform_op>;

    Expr _value;

public:
    using        value_type = T;
    using       memory_type = void;
    using const_memory_type = void;

    //Construct a new expression
    explicit unary_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& e) = delete;
    unary_expr& operator=(unary_expr&& e) = delete;

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
    auto operator()(std::size_t i) const {
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

template <typename T, typename Expr, typename Op>
struct unary_expr<T, Expr, stateful_op<Op>> :
        comparable<unary_expr<T, Expr, stateful_op<Op>>>
        , value_testable<unary_expr<T, Expr, stateful_op<Op>>>
        , dim_testable<unary_expr<T, Expr, stateful_op<Op>>>
        {
private:
    using this_type = unary_expr<T, Expr, stateful_op<Op>>;

    Expr _value;
    Op op;

public:
    using        value_type = T;
    using       memory_type = void;
    using const_memory_type = void;

    //Construct a new expression
    template<typename... Args>
    explicit unary_expr(Expr l, Args&&... args) : _value(std::forward<Expr>(l)), op(std::forward<Args>(args)...) {
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& e) = delete;
    unary_expr& operator=(unary_expr&& e) = delete;

    //Accessors

    std::add_lvalue_reference_t<Expr> value() noexcept {
        return _value;
    }

    cpp::add_const_lvalue_t<Expr> value() const noexcept {
        return _value;
    }

    //Apply the expression

    value_type operator[](std::size_t i) const {
        return op.apply(value()[i]);
    }

    intrinsic_type<value_type> load(std::size_t i) const {
        return op.load(value().load(i));
    }

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return op.apply(value()(args...));
    }

    iterator<const this_type, false, false> begin() const noexcept {
        return {*this, 0};
    }

    iterator<const this_type, false, false> end() const noexcept {
        return {*this, size(*this)};
    }
};

template <typename T, typename Expr, typename UnaryOp>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, stateful_op<UnaryOp>>& expr){
    return os << UnaryOp::desc() << '(' << expr.value() << ')';
}

template <typename T, typename Expr>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, identity_op>& expr){
    return os << expr.value();
}

template <typename T, typename Expr>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, transform_op>& expr){
    return os << expr.value();
}

template <typename T, typename Expr, typename UnaryOp>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, UnaryOp>& expr){
    return os << UnaryOp::desc() << '(' << expr.value() << ')';
}

} //end of namespace etl
