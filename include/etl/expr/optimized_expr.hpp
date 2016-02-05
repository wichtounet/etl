//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

#include <iosfwd> //For stream support

#include "etl/wrapper_traits.hpp"

namespace etl {

/*!
 * \brief A wrapper for expressions that need to be optimized
 */
template <typename Expr>
struct optimized_expr final {
private:
    Expr _value;

public:
    using expr_t     = Expr;          ///< The wrapped expression type
    using value_type = value_t<Expr>; ///< The value type

    //Cannot be constructed with no args
    optimized_expr() = delete;

    /*!
     * \brief Construt a new optimized expression around the given ETL expression
     * \param l The ETL expression
     */
    optimized_expr(Expr l)
            : _value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    //Expresison can be copied and moved
    optimized_expr(const optimized_expr& e) = default;
    optimized_expr(optimized_expr&& e) noexcept = default;

    //Expressions are invariant
    optimized_expr& operator=(const optimized_expr& e) = delete;
    optimized_expr& operator=(optimized_expr&& e) = delete;

    /*!
     * \brief Return the sub expression
     * \return a reference to the sub expression
     */
    std::add_lvalue_reference_t<Expr> value() {
        return _value;
    }

    /*!
     * \brief Return the sub expression
     * \return a reference to the sub expression
     */
    cpp::add_const_lvalue_t<Expr> value() const {
        return _value;
    }
};

/*!
 * \brief Specilization of the traits for optimized_expr
 * Optimized expression simply use the same traits as its expression
 */
template <typename Expr>
struct etl_traits<etl::optimized_expr<Expr>> : wrapper_traits<etl::optimized_expr<Expr>> {};

/*!
 * \brief Prints the type of the optimized expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename Expr>
std::ostream& operator<<(std::ostream& os, const optimized_expr<Expr>& expr) {
    return os << "OPT(" << expr.value() << ")";
}

} //end of namespace etl
