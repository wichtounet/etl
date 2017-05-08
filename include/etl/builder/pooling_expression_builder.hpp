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

/* Max Pool 2D */

/*!
 * \brief 2D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the 2D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1 = C1, size_t S2 = C2, size_t P1 = 0, size_t P2 = 0, typename E>
pool_2d_expr<E, C1, C2, S1, S2, P1, P2, impl::max_pool_2d> max_pool_2d(E&& value) {
    return pool_2d_expr<E, C1, C2, S1, S2, P1, P2, impl::max_pool_2d>{value};
}

/*!
 * \brief 2D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the 2D Max Pooling of the input expression.
 */
template <typename E>
dyn_pool_2d_expr<E, impl::max_pool_2d> max_pool_2d(E&& value, size_t c1, size_t c2) {
    return dyn_pool_2d_expr<E, impl::max_pool_2d>{value, c1, c2, c1, c2, 0, 0};
}

/*!
 * \brief 2D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the 2D Max Pooling of the input expression.
 */
template <typename E>
dyn_pool_2d_expr<E, impl::max_pool_2d> max_pool_2d(E&& value, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    return dyn_pool_2d_expr<E, impl::max_pool_2d>{value, c1, c2, s1, s2, p1, p2};
}

/* AVG Pool 2D */

/*!
 * \brief 2D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the 2D Average Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1 = C1, size_t S2 = C2, size_t P1 = 0, size_t P2 = 0, typename E>
pool_2d_expr<E, C1, C2, S1, S2, P1, P2, impl::avg_pool_2d> avg_pool_2d(E&& value) {
    return pool_2d_expr<E, C1, C2, S1, S2, P1, P2, impl::avg_pool_2d>{value};
}

/*!
 * \brief 2D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the 2D Average Pooling of the input expression.
 */
template <typename E>
dyn_pool_2d_expr<E, impl::avg_pool_2d> avg_pool_2d(E&& value, size_t c1, size_t c2) {
    return dyn_pool_2d_expr<E, impl::avg_pool_2d>{value, c1, c2, c1, c2, 0, 0};
}

/*!
 * \brief 2D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the 2D Average Pooling of the input expression.
 */
template <typename E>
dyn_pool_2d_expr<E, impl::avg_pool_2d> avg_pool_2d(E&& value, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    return dyn_pool_2d_expr<E, impl::avg_pool_2d>{value, c1, c2, s1, s2, p1, p2};
}

/* Max Pool 3D */

/*!
 * \brief 3D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \tparam C3 The third pooling ratio
 * \return A expression representing the 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, size_t S1 = C1, size_t S2 = C2, size_t S3 = C3, size_t P1 = 0, size_t P2 = 0, size_t P3 = 0, typename E>
pool_3d_expr<E, C1, C2, C3, S1, S2, S3, P1, P2, P3, impl::max_pool_3d> max_pool_3d(E&& value) {
    return pool_3d_expr<E, C1, C2, C3, S1, S2, S3, P1, P2, P3, impl::max_pool_3d>{value};
}

/*!
 * \brief 3D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the 3D Max Pooling of the input expression.
 */
template <typename E>
dyn_pool_3d_expr<E, impl::max_pool_3d> max_pool_3d(E&& value, size_t c1, size_t c2, size_t c3) {
    return dyn_pool_3d_expr<E, impl::max_pool_3d>{value, c1, c2, c3, c1, c2, c3, 0, 0, 0};
}

/*!
 * \brief 3D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the 3D Max Pooling of the input expression.
 */
template <typename E>
dyn_pool_3d_expr<E, impl::max_pool_3d> max_pool_3d(E&& value, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1 = 0, size_t p2 = 0, size_t p3 = 0) {
    return dyn_pool_3d_expr<E, impl::max_pool_3d>{value, c1, c2, c3, s1, s2, s3, p1, p2, p3};
}

/* Avg Pool 3D */

/*!
 * \brief 3D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \tparam C3 The third pooling ratio
 * \return A expression representing the 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, size_t S1 = C1, size_t S2 = C2, size_t S3 = C3, size_t P1 = 0, size_t P2 = 0, size_t P3 = 0, typename E>
pool_3d_expr<E, C1, C2, C3, S1, S2, S3, P1, P2, P3, impl::avg_pool_3d> avg_pool_3d(E&& value) {
    return pool_3d_expr<E, C1, C2, C3, S1, S2, S3, P1, P2, P3, impl::avg_pool_3d>{value};
}

/*!
 * \brief 3D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the 3D Average Pooling of the input expression.
 */
template <typename E>
dyn_pool_3d_expr<E, impl::avg_pool_3d> avg_pool_3d(E&& value, size_t c1, size_t c2, size_t c3) {
    return dyn_pool_3d_expr<E, impl::avg_pool_3d>{value, c1, c2, c3, c1, c2, c3, 0, 0, 0};
}

/*!
 * \brief 3D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the 3D Average Pooling of the input expression.
 */
template <typename E>
dyn_pool_3d_expr<E, impl::avg_pool_3d> avg_pool_3d(E&& value, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1 = 0, size_t p2 = 0, size_t p3 = 0) {
    return dyn_pool_3d_expr<E, impl::avg_pool_3d>{value, c1, c2, c3, s1, s2, s3, p1, p2, p3};
}

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

/* Max Pool 2D Derivative */

/*!
 * \brief Derivative of the 2D Max Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 2D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, typename E, typename F>
pool_derivative_expr<E, F, C1, C2, 0, impl::max_pool_derivative_2d> max_pool_derivative_2d(E&& input, F&& output) {
    return pool_derivative_expr<E, F, C1, C2, 0, impl::max_pool_derivative_2d>{input, output};
}

/*!
 * \brief Derivative of the 2D Max Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Derivative of 2D Max Pooling of the input expression.
 */
template <typename E, typename F>
dyn_pool_derivative_expr<E, F, impl::max_pool_derivative_2d> max_pool_derivative_2d(E&& input, F&& output, size_t c1, size_t c2) {
    return dyn_pool_derivative_expr<E, F, impl::max_pool_derivative_2d>{input, output, c1, c2, 0};
}

/* Max Pool 3D Derivative */

/*!
 * \brief Derivative of the 3D Max Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \tparam C3 The third pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, typename E, typename F>
auto max_pool_derivative_3d(E&& input, F&& output) {
    return pool_derivative_expr<E, F, C1, C2, C3, impl::max_pool_derivative_3d>{input, output};
}

/*!
 * \brief Derivative of the 3D Max Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <typename E, typename F>
dyn_pool_derivative_expr<E, F, impl::max_pool_derivative_3d> max_pool_derivative_3d(E&& input, F&& output, size_t c1, size_t c2, size_t c3) {
    return dyn_pool_derivative_expr<E, F, impl::max_pool_derivative_3d>{input, output, c1, c2, c3};
}

/* Max Pool 2D Upsample */

/*!
 * \brief Derivative of the 2D Max Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, typename A, typename B, typename C>
auto max_pool_upsample_2d(A&& input, B&& output, C&& errors) {
    using detail::build_type;
    return max_pool_upsample_2d_expr<build_type<A>, build_type<B>, build_type<C>, C1, C2>{input, output, errors};
}

/*!
 * \brief Derivative of the 2D Max Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <typename A, typename B, typename C>
auto max_pool_upsample_2d(A&& input, B&& output, C&& errors, size_t c1, size_t c2) {
    using detail::build_type;
    return dyn_max_pool_upsample_2d_expr<build_type<A>, build_type<B>, build_type<C>>{input, output, errors, c1, c2};
}

/* Max Pool 3D Upsample */

/*!
 * \brief Derivative of the 3D Max Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \tparam C3 The third pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C>
auto max_pool_upsample_3d(A&& input, B&& output, C&& errors) {
    using detail::build_type;
    return max_pool_upsample_3d_expr<build_type<A>, build_type<B>, build_type<C>, C1, C2, C3>{input, output, errors};
}

/*!
 * \brief Derivative of the 3D Max Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <typename A, typename B, typename C>
auto max_pool_upsample_3d(A&& input, B&& output, C&& errors, size_t c1, size_t c2, size_t c3) {
    using detail::build_type;
    return dyn_max_pool_upsample_3d_expr<build_type<A>, build_type<B>, build_type<C>>{input, output, errors, c1, c2, c3};
}

/* Upsample 2D */

/*!
 * \brief Upsample the given 2D matrix expression
 * \param value The input expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Upsampling of the given expression
 */
template <size_t C1, size_t C2, typename E>
upsample_2d_expr<E, C1, C2, impl::upsample_2d> upsample_2d(E&& value) {
    return upsample_2d_expr<E, C1, C2, impl::upsample_2d>{value};
}

/*!
 * \brief Upsample the given 2D matrix expression
 * \param value The input expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Upsampling of the given expression
 */
template <typename E>
dyn_upsample_2d_expr<E, impl::upsample_2d> upsample_2d(E&& value, size_t c1, size_t c2) {
    return dyn_upsample_2d_expr<E, impl::upsample_2d>{value, c1, c2};
}

/* Upsample 3D */

/*!
 * \brief Upsample the given 3D matrix expression
 * \param value The input expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \tparam C3 The third pooling ratio
 * \return A expression representing the Upsampling of the given expression
 */
template <size_t C1, size_t C2, size_t C3, typename E>
upsample_3d_expr<E, C1, C2, C3, impl::upsample_3d> upsample_3d(E&& value) {
    return upsample_3d_expr<E, C1, C2, C3, impl::upsample_3d>{value};
}

/*!
 * \brief Upsample the given 3D matrix expression
 * \param value The input expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the Upsampling of the given expression
 */
template <typename E>
dyn_upsample_3d_expr<E, impl::upsample_3d> upsample_3d(E&& value, size_t c1, size_t c2, size_t c3) {
    return dyn_upsample_3d_expr<E, impl::upsample_3d>{value, c1, c2, c3};
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
prob_pool_2d_expr<E, C1, C2, impl::pmp_h_impl> p_max_pool_h(E&& value) {
    validate_pmax_pooling<C1, C2>(value);
    return prob_pool_2d_expr<E, C1, C2, impl::pmp_h_impl>{value};
}

/*!
 * \brief Probabilistic Max Pooling for hidden units
 * \param value The input expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Probabilistic Max Pooling of hidden units
 */
template <typename E>
dyn_prob_pool_2d_expr<E, impl::dyn_pmp_h_impl> p_max_pool_h(E&& value, size_t c1, size_t c2) {
    validate_pmax_pooling(value, c1, c2);
    return dyn_prob_pool_2d_expr<E, impl::dyn_pmp_h_impl>{value, c1, c2};
}

/* Probabilistic Max Pooling (pooling) */

/*!
 * \brief Probabilistic Max Pooling for pooling units
 * \param value The input expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Probabilistic Max Pooling of pooling units
 */
template <size_t C1, size_t C2, typename E>
pool_2d_expr<E, C1, C2, C1, C2, 0, 0, impl::pmp_p_impl> p_max_pool_p(E&& value) {
    validate_pmax_pooling<C1, C2>(value);
    return pool_2d_expr<E, C1, C2, C1, C2, 0, 0, impl::pmp_p_impl>{value};
}

/*!
 * \brief Probabilistic Max Pooling for pooling units
 * \param value The input expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Probabilistic Max Pooling of pooling units
 */
template <typename E>
dyn_pool_2d_expr<E, impl::dyn_pmp_p_impl> p_max_pool_p(E&& value, size_t c1, size_t c2) {
    validate_pmax_pooling(value, c1, c2);
    return dyn_pool_2d_expr<E, impl::dyn_pmp_p_impl>{value, c1, c2, c1, c2, 0, 0};
}

} //end of namespace etl
