//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        decltype(auto) forced = detail::get_forced_impl<selector_t>();

        auto old_forced = forced;

        forced.impl = selector_value;
        forced.forced = true;

        _value.assign_to(lhs);

        forced = old_forced;
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_add_to(L&& lhs)  const {
        decltype(auto) forced = detail::get_forced_impl<selector_t>();

        auto old_forced = forced;

        forced.impl = selector_value;
        forced.forced = true;

        _value.assign_add_to(lhs);

        forced = old_forced;
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_sub_to(L&& lhs)  const {
        decltype(auto) forced = detail::get_forced_impl<selector_t>();

        auto old_forced = forced;

        forced.impl = selector_value;
        forced.forced = true;

        _value.assign_sub_to(lhs);

        forced = old_forced;
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mul_to(L&& lhs)  const {
        decltype(auto) forced = detail::get_forced_impl<selector_t>();

        auto old_forced = forced;

        forced.impl = selector_value;
        forced.forced = true;

        _value.assign_mul_to(lhs);

        forced = old_forced;
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_div_to(L&& lhs)  const {
        decltype(auto) forced = detail::get_forced_impl<selector_t>();

        auto old_forced = forced;

        forced.impl = selector_value;
        forced.forced = true;

        _value.assign_div_to(lhs);

        forced = old_forced;
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mod_to(L&& lhs)  const {
        decltype(auto) forced = detail::get_forced_impl<selector_t>();

        auto old_forced = forced;

        forced.impl = selector_value;
        forced.forced = true;

        _value.assign_mod_to(lhs);

        forced = old_forced;
    }

    /*!
     * \brief Prints the type of the optimized expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const selected_expr& expr) {
        return os << "selected(" << expr._value << ")";
    }
};

/*!
 * \brief Specilization of the traits for selected_expr
 * selected expression simply use the same traits as its expression
 */
template <typename Selector, Selector V, typename Expr>
struct etl_traits<etl::selected_expr<Selector, V, Expr>> : wrapper_traits<etl::selected_expr<Selector, V, Expr>> {};

} //end of namespace etl
