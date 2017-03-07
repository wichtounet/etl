//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
    explicit optimized_expr(Expr l)
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

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs) {
        //Note: This is more than ugly...
        optimized_forward(_value,
                      [&lhs](auto&& optimized) mutable {
                          using tt = std::remove_const_t<std::remove_reference_t<decltype(optimized)>>;
                          const_cast<tt&>(optimized).assign_to(lhs);
                      });
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_add_to(L&& lhs) {
        //Note: This is more than ugly...
        optimized_forward(_value,
                      [&lhs](auto&& optimized) mutable {
                          using tt = std::remove_const_t<std::remove_reference_t<decltype(optimized)>>;
                          const_cast<tt&>(optimized).assign_add_to(lhs);
                      });
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_sub_to(L&& lhs) {
        //Note: This is more than ugly...
        optimized_forward(_value,
                      [&lhs](auto&& optimized) mutable {
                          using tt = std::remove_const_t<std::remove_reference_t<decltype(optimized)>>;
                          const_cast<tt&>(optimized).assign_sub_to(lhs);
                      });
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mul_to(L&& lhs) {
        //Note: This is more than ugly...
        optimized_forward(_value,
                      [&lhs](auto&& optimized) mutable {
                          using tt = std::remove_const_t<std::remove_reference_t<decltype(optimized)>>;
                          const_cast<tt&>(optimized).assign_mul_to(lhs);
                      });
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_div_to(L&& lhs) {
        //Note: This is more than ugly...
        optimized_forward(_value,
                      [&lhs](auto&& optimized) mutable {
                          using tt = std::remove_const_t<std::remove_reference_t<decltype(optimized)>>;
                          const_cast<tt&>(optimized).assign_div_to(lhs);
                      });
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mod_to(L&& lhs) {
        //Note: This is more than ugly...
        optimized_forward(_value,
                      [&lhs](auto&& optimized) mutable {
                          using tt = std::remove_const_t<std::remove_reference_t<decltype(optimized)>>;
                          const_cast<tt&>(optimized).assign_mod_to(lhs);
                      });
    }

    /*!
     * \brief Prints the type of the optimized expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const optimized_expr& expr) {
        return os << "OPT(" << expr._value << ")";
    }
};

/*!
 * \brief Specilization of the traits for optimized_expr
 * Optimized expression simply use the same traits as its expression
 */
template <typename Expr>
struct etl_traits<etl::optimized_expr<Expr>> : wrapper_traits<etl::optimized_expr<Expr>> {};

} //end of namespace etl
