//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains an expression whose implementation is selected
*/

#pragma once

#include "etl/wrapper_traits.hpp"

namespace etl {

/*!
 * \brief A wrapper for expressions that is forced to be selected.
 */
template <typename Selector, Selector V, typename Expr>
struct selected_expr final {
private:
    Expr _value;

public:
    using expr_t     = Expr;          ///< The wrapped expression type
    using value_type = value_t<Expr>; ///< The value type

    using selector_t = Selector; ///< The enum selector type

    static constexpr selector_t selector_value = V; ///< The enum selector value

    //Cannot be constructed with no args
    selected_expr() = delete;

    /*!
     * \brief Construt a new optimized expression around the given ETL expression
     * \param l The ETL expression
     */
    explicit selected_expr(Expr l)
            : _value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    //Expresison can be copied and moved
    selected_expr(const selected_expr& e) = default;
    selected_expr(selected_expr&& e) noexcept = default;

    //Expressions are invariant
    selected_expr& operator=(const selected_expr& e) = delete;
    selected_expr& operator=(selected_expr&& e) = delete;

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
 * \brief Specilization of the traits for selected_expr
 * selected expression simply use the same traits as its expression
 */
template <typename Selector, Selector V, typename Expr>
struct etl_traits<etl::selected_expr<Selector, V, Expr>> : wrapper_traits<etl::selected_expr<Selector, V, Expr>> {};

/*!
 * \brief Prints the type of the optimized expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename Selector, Selector V, typename Expr>
std::ostream& operator<<(std::ostream& os, const selected_expr<Selector, V, Expr>& expr) {
    return os << "selected(" << expr.value() << ")";
}

} //end of namespace etl
