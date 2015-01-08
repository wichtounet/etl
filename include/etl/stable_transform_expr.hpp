//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_STABLE_TRANSFORM_EXPR_HPP
#define ETL_STABLE_TRANSFORM_EXPR_HPP

#include "traits_fwd.hpp"
#include "iterator.hpp"

namespace etl {

template <typename T, typename Expr>
class stable_transform_expr final {
private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in stable_transform_expr");

    using this_type = stable_transform_expr<T, Expr>;

    Expr _value;

public:
    using value_type = T;
    using expr_type = Expr;

    //Cannot be constructed with no args
    stable_transform_expr() = delete;

    //Construct a new expression
    stable_transform_expr(Expr l) : _value(std::forward<Expr>(l)){
        //Nothing else to init
    }

    stable_transform_expr(const stable_transform_expr& e) : _value(e._value) {
        //Nothing else to init
    }

    stable_transform_expr(stable_transform_expr&& e) : _value(e._value) {
        //Nothing else to init
    }

    //Expression are invariant
    stable_transform_expr& operator=(const stable_transform_expr&) = delete;
    stable_transform_expr& operator=(stable_transform_expr&&) = delete;

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

    template<typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }

    template<bool B = (sub_size_compare<this_type>::value > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i){
        return sub(*this, i);
    }

    template<bool B = (sub_size_compare<this_type>::value > 1), cpp::enable_if_u<B> = cpp::detail::dummy>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
    }

    iterator<const this_type> begin() const noexcept {
        return {*this, 0};
    }

    iterator<const this_type> end() const noexcept {
        return {*this, size(*this)};
    }
};

//}}}

} //end of namespace etl

#endif
