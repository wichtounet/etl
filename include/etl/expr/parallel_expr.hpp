//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
private:
    Expr _value;

public:
    using expr_t     = Expr;          ///< The wrapped expression type
    using value_type = value_t<Expr>; ///< The value type

    //Cannot be constructed with no args
    parallel_expr() = delete;

    /*!
     * \brief Construt a new optimized expression around the given ETL expression
     * \param l The ETL expression
     */
    explicit parallel_expr(Expr l)
            : _value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    //Expresison can be copied and moved
    parallel_expr(const parallel_expr& e) = default;
    parallel_expr(parallel_expr&& e) noexcept = default;

    //Expressions are invariant
    parallel_expr& operator=(const parallel_expr& e) = delete;
    parallel_expr& operator=(parallel_expr&& e) = delete;

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
        std_assign_evaluate(*this, lhs);
    }
};

/*!
 * \brief Specilization of the traits for parallel_expr
 * parallel expression simply use the same traits as its expression
 */
template <typename Expr>
struct etl_traits<etl::parallel_expr<Expr>> : wrapper_traits<etl::parallel_expr<Expr>> {};

/*!
 * \brief Prints the type of the optimized expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename Expr>
std::ostream& operator<<(std::ostream& os, const parallel_expr<Expr>& expr) {
    return os << "parallel(" << expr.value() << ")";
}

} //end of namespace etl
