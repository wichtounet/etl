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
#include "etl/expr/batch_k_scale_plus_expr.hpp"
#include "etl/expr/batch_k_minus_scale_expr.hpp"

namespace etl {

constexpr bool is_2d4d(size_t dimensions) {
    return dimensions == 2 || dimensions == 4;
}

/*!
 * \brief Build a special expression for batched expressions
 * \param expr The expression to be transformed.
 * \return Either the same expression or a transformed batch expression if possible
 */
template <typename Expr>
auto batch_hint(Expr&& expr) {
    using expr_t = std::decay_t<Expr>;

    // If this becomes more complicated, make one detection function for each 
    // possible type instead of craming everything here

    if constexpr (is_binary_expr<Expr>) {
        using value_type    = typename expr_t::value_type;
        using operator_type = typename expr_t::operator_type;

        using left_type  = typename expr_t::left_type;
        using right_type = typename expr_t::right_type;

        constexpr size_t left_dimensions  = decay_traits<left_type>::dimensions();
        constexpr size_t right_dimensions = decay_traits<right_type>::dimensions();

        if constexpr (std::is_same_v<operator_type, mul_binary_op<value_type>>) {
            if constexpr (is_binary_expr<right_type>) {
                auto& right_expr = expr.get_rhs();

                using right_value_type    = typename right_type::value_type;
                using right_operator_type = typename right_type::operator_type;

                using right_left_type  = typename right_type::left_type;
                using right_right_type = typename right_type::right_type;

                constexpr size_t right_left_dimensions  = decay_traits<right_left_type>::dimensions();
                constexpr size_t right_right_dimensions = decay_traits<right_right_type>::dimensions();

                if constexpr (std::is_same_v<right_operator_type, minus_binary_op<right_value_type>>) {
                    if constexpr (right_left_dimensions == 4 && right_right_dimensions == 1 && left_dimensions == 1 && all_dma<right_left_type, right_right_type, left_type>) {
                        // Detect gamma[K] * (input[B, K, W, H]) - beta[k])
                        return batch_k_minus_scale(expr.get_lhs(), right_expr.get_lhs(), right_expr.get_rhs());
                    } else {
                        return std::forward<Expr>(expr);
                    }
                } else {
                    return std::forward<Expr>(expr);
                }
            } else {
                if constexpr (left_dimensions == 1 && right_dimensions == 4 && all_dma<left_type, right_type>) {
                    // Detect gamma[K] * beta[B, K, W, H]
                    return batch_k_scale(expr.get_lhs(), expr.get_rhs());
                } else {
                }
            }
        } else if constexpr (std::is_same_v<operator_type, plus_binary_op<value_type>>) {
            if constexpr (is_binary_expr<left_type>) {
                auto& left_expr = expr.get_lhs();

                using left_value_type    = typename left_type::value_type;
                using left_operator_type = typename left_type::operator_type;

                using left_left_type  = typename left_type::left_type;
                using left_right_type = typename left_type::right_type;

                constexpr size_t left_left_dimensions  = decay_traits<left_left_type>::dimensions();
                constexpr size_t left_right_dimensions = decay_traits<left_right_type>::dimensions();

                if constexpr (std::is_same_v<left_operator_type, mul_binary_op<left_value_type>>) {
                    if constexpr (left_left_dimensions == 1 && is_2d4d(left_right_dimensions) && right_dimensions == 1 && all_dma<left_left_type, left_right_type, right_type>) {
                        // Detect (gamma[K] * input[B, K, W, H]) + beta[k]
                        return batch_k_scale_plus(left_expr.get_lhs(), left_expr.get_rhs(), expr.get_rhs());
                    } else {
                        return std::forward<Expr>(expr);
                    }
                } else {
                    return std::forward<Expr>(expr);
                }

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
