//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains an expression that is forced to be executed in parallel
 *
 * If ETL is configured to use parallel, this has no effect. When
 * ETL is not configured in parallel, this allows for fine-grained
 * parallelism selection.
 */

#pragma once

#include "etl/wrapper_traits.hpp"

namespace etl {

/*!
 * \brief A wrapper for expressions that is to be executed in
 * parallel
 */
template <typename Expr>
struct parallel_expr final {
    using expr_t     = Expr;          ///< The wrapped expression type
    using value_type = value_t<Expr>; ///< The value type

private:
    Expr value;

    friend struct wrapper_traits<parallel_expr>;

public:
    //Cannot be constructed with no args
    parallel_expr() = delete;

    /*!
     * \brief Construt a new optimized expression around the given ETL expression
     * \param l The ETL expression
     */
    explicit parallel_expr(Expr l) : value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    //Expresison can be copied and moved
    parallel_expr(const parallel_expr& e)     = default;
    parallel_expr(parallel_expr&& e) noexcept = default;

    //Expressions are invariant
    parallel_expr& operator=(const parallel_expr& e) = delete;
    parallel_expr& operator=(parallel_expr&& e) = delete;

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param other The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& other) const noexcept {
        return value.alias(other);
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        auto old_parallel = local_context().parallel;

        local_context().parallel = true;

        value.assign_to(lhs);

        local_context().parallel = old_parallel;
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        auto old_parallel = local_context().parallel;

        local_context().parallel = true;

        value.assign_add_to(lhs);

        local_context().parallel = old_parallel;
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        auto old_parallel = local_context().parallel;

        local_context().parallel = true;

        value.assign_sub_to(lhs);

        local_context().parallel = old_parallel;
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        auto old_parallel = local_context().parallel;

        local_context().parallel = true;

        value.assign_mul_to(lhs);

        local_context().parallel = old_parallel;
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        auto old_parallel = local_context().parallel;

        local_context().parallel = true;

        value.assign_div_to(lhs);

        local_context().parallel = old_parallel;
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        auto old_parallel = local_context().parallel;

        local_context().parallel = true;

        value.assign_mod_to(lhs);

        local_context().parallel = old_parallel;
    }

    /*!
     * \brief Prints the type of the optimized expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const parallel_expr& expr) {
        return os << "parallel(" << expr.value << ")";
    }
};

/*!
 * \brief Specilization of the traits for parallel_expr
 * parallel expression simply use the same traits as its expression
 */
template <typename Expr>
struct etl_traits<etl::parallel_expr<Expr>> : wrapper_traits<etl::parallel_expr<Expr>> {};

} //end of namespace etl
