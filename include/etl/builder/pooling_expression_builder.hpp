//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file pooling_expression_builder.hpp
 * \brief Contains all the pooling operators and functions to build expressions.
*/

#pragma once

#include "etl/expression_helpers.hpp"

namespace etl {

/* Max Pool 3D */

/* Avg Pool 2D Derivative */

/*!
 * \brief Derivative of the 2D Average Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 2D Average Pooling of the input expression.
 */
template <size_t C1, size_t C2, typename E, typename F>
auto avg_pool_derivative_2d(E&& input, F&& output) {
    cpp_unused(input);
    cpp_unused(output);
    return 1.0 / (C1 * C2);
}

/*!
 * \brief Derivative of the 2D Average Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \param c1 the first pooling ratio
 * \param c2 the second pooling ratio
 * \return A expression representing the Derivative of 2D Average Pooling of the input expression.
 */
template <typename E, typename F>
auto avg_pool_derivative_2d(E&& input, F&& output, size_t c1, size_t c2) {
    cpp_unused(input);
    cpp_unused(output);
    return 1.0 / (c1 * c2);
}

/* Avg Pool 3D Derivative */

/*!
 * \brief Derivative of the 3D Average Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \tparam C3 The third pooling ratio
 * \return A expression representing the Derivative of 3D Average Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, typename E, typename F>
auto avg_pool_derivative_3d(E&& input, F&& output) {
    cpp_unused(input);
    cpp_unused(output);
    return 1.0 / (C1 * C2 * C3);
}

/*!
 * \brief Derivative of the 3D Average Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \param c1 the first pooling ratio
 * \param c2 the second pooling ratio
 * \param c3 the third pooling ratio
 * \return A expression representing the Derivative of 3D Average Pooling of the input expression.
 */
template <typename E, typename F>
auto avg_pool_derivative_3d(E&& input, F&& output, size_t c1, size_t c2, size_t c3) {
    cpp_unused(input);
    cpp_unused(output);
    return 1.0 / (c1 * c2 * c3);
}

/* Probabilistic Max Pooling (hidden) */

/*!
 * \brief Probabilistic Max Pooling for hidden units
 * \param value The input expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Probabilistic Max Pooling of hidden units
 */
template <size_t C1, size_t C2, typename E>
prob_pool_2d_expr<detail::build_type<E>, C1, C2, impl::pmp_h_impl> p_max_pool_h(E&& value) {
    validate_pmax_pooling<C1, C2>(value);
    return prob_pool_2d_expr<detail::build_type<E>, C1, C2, impl::pmp_h_impl>{value};
}

/*!
 * \brief Probabilistic Max Pooling for hidden units
 * \param value The input expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Probabilistic Max Pooling of hidden units
 */
template <typename E>
dyn_prob_pool_2d_expr<detail::build_type<E>, impl::dyn_pmp_h_impl> p_max_pool_h(E&& value, size_t c1, size_t c2) {
    validate_pmax_pooling(value, c1, c2);
    return dyn_prob_pool_2d_expr<detail::build_type<E>, impl::dyn_pmp_h_impl>{value, c1, c2};
}

} //end of namespace etl
