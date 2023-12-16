//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains all the operators and functions to build expressions for
 * wrapping expressions.
 */

#pragma once

namespace etl {

/*!
 * \brief Create an optimized expression wrapping the given expression.
 *
 * The expression will be optimized before being evaluated.
 * \param expr The expression to be wrapped
 * \return an optimized expression wrapping the given expression
 */
template <typename Expr>
auto opt(Expr&& expr) -> optimized_expr<detail::build_type<Expr>> {
    return optimized_expr<detail::build_type<Expr>>{expr};
}

/*!
 * \brief Create a timed expression wrapping the given expression.
 *
 * The evaluation (and assignment) of the expression will be timed.
 *
 * \param expr The expression to be wrapped
 * \return a timed expression wrapping the given expression
 */
template <typename Expr>
auto timed(Expr&& expr) -> timed_expr<detail::build_type<Expr>> {
    return timed_expr<detail::build_type<Expr>>{expr};
}

/*!
 * \brief Create a timed expression wrapping the given expression with the given resolution.
 *
 * The evaluation (and assignment) of the expression will be timed.
 *
 * \tparam R The clock resolution (std::chrono resolutions)
 * \param expr The expression to be wrapped
 * \return a timed expression wrapping the given expression
 */
template <typename R, typename Expr>
auto timed_res(Expr&& expr) -> timed_expr<detail::build_type<Expr>, R> {
    return timed_expr<detail::build_type<Expr>, R>{expr};
}

/*!
 * \brief Create a serial expression wrapping the given expression.
 *
 * The evaluation (and assignment) of the expression is guaranteed to be evaluated serially.
 *
 * \param expr The expression to be wrapped
 * \return a serial expression wrapping the given expression
 */
template <typename Expr>
auto serial(Expr&& expr) -> serial_expr<detail::build_type<Expr>> {
    return serial_expr<detail::build_type<Expr>>{expr};
}

/*!
 * \brief Create a parallel expression wrapping the given expression.
 *
 * The evaluation (and assignment) of the expression will be done parallelly, if possible.
 *
 * \param expr The expression to be wrapped
 * \return a parallel expression wrapping the given expression
 */
template <typename Expr>
auto parallel(Expr&& expr) -> parallel_expr<detail::build_type<Expr>> {
    return parallel_expr<detail::build_type<Expr>>{expr};
}

#ifdef ETL_MANUAL_SELECT

/*!
 * \brief Create selectedd expression wrapping the given expression.
 * \param expr The expression to be wrapped
 * \return a selected expression wrapping the given expression
 */
template <typename Selector, Selector V, typename Expr>
auto selected(Expr&& expr) -> selected_expr<Selector, V, detail::build_type<Expr>> {
    return selected_expr<Selector, V, detail::build_type<Expr>>{expr};
}

#define selected_helper(v, expr) etl::selected<decltype(v), v>(expr)

#endif

} //end of namespace etl
