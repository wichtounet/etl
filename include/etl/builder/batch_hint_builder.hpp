//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains all the operators and functions to build binary expressions.
 */

#pragma once

#include "etl/expr/batch_k_scale_expr.hpp"

namespace etl {

/*!
 * \brief Build a special expression for batched expressions
 * \param expr The expression to be transformed.
 * \return Either the same expression or a transformed batch expression if possible
 */
template <typename Expr>
auto batch_hint(Expr&& expr) {
    using expr_t = std::decay_t<Expr>;

    if constexpr (is_binary_expr<Expr>) {
        using value_type    = typename expr_t::value_type;
        using operator_type = typename expr_t::operator_type;

        using left_type  = typename expr_t::left_type;
        using right_type = typename expr_t::right_type;

        constexpr size_t left_dimensions  = decay_traits<left_type>::dimensions();
        constexpr size_t right_dimensions = decay_traits<right_type>::dimensions();

        if constexpr (std::is_same_v<operator_type, mul_binary_op<value_type>>) {
            if constexpr (left_dimensions == 1 && right_dimensions == 4) {
                // Detect gamma[K] * beta[B, K, W, H]
                return batch_k_scale(expr.get_lhs(), expr.get_rhs());
            } else {
                return std::forward<Expr>(expr);
            }
        } else {
            return std::forward<Expr>(expr);
        }
    } else {
        return std::forward<Expr>(expr);
    }
}

} //end of namespace etl
