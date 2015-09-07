//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file optimized_expr.hpp
 * \brief Contains an optimized expression implementation.
 *
 * An optimized expression will first pass throught the optimizer before it is passed through the evaluator.
*/

#pragma once

#include <iosfwd>     //For stream support

#include "etl/traits_lite.hpp"

namespace etl {

template <typename Expr>
struct optimized_expr final {
private:
    Expr _value;

public:
    using        value_type = value_t<Expr>;

    //Cannot be constructed with no args
    optimized_expr() = delete;

    //Construct a new expression
    optimized_expr(Expr l) : _value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    //Expresison can be copied and moved
    optimized_expr(const optimized_expr& e) = default;
    optimized_expr(optimized_expr&& e) = default;

    //Expressions are invariant
    optimized_expr& operator=(const optimized_expr& e) = delete;
    optimized_expr& operator=(optimized_expr&& e) = delete;

    //Accessors

    std::add_lvalue_reference_t<Expr> value(){
        return _value;
    }

    cpp::add_const_lvalue_t<Expr> value() const {
        return _value;
    }
};

template <typename Expr>
std::ostream& operator<<(std::ostream& os, const optimized_expr<Expr>& expr){
    return os << "OPT(" << expr.value() << ")";
}

} //end of namespace etl
