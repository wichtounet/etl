//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file expression_helpers.hpp
 * \brief Contains internal helpers to build expressions.
*/

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief Helper to build the type for a sub expression
 *
 * This means a reference for a value type and a copy for another
 * expression.
 */
template <typename T>
using build_type = std::conditional_t<
    is_etl_value<T>::value,
    const std::decay_t<T>&,
    std::decay_t<T>>;

/*!
 * \brief Helper to build the identity type for a sub expression
 *
 * This means a reference for a value type (preserving constness)
 * and a copy for another expression.
 */
template <typename T>
using build_identity_type = std::conditional_t<
    is_etl_value<T>::value,
    std::conditional_t<
        std::is_const<std::remove_reference_t<T>>::value,
        const std::decay_t<T>&,
        std::decay_t<T>&>,
    std::decay_t<T>>;

/*!
 * \brief Helper to create a binary expr with left typing
 */
template <typename LE, typename RE, template <typename> class OP>
using left_binary_helper = binary_expr<value_t<LE>, build_type<LE>, OP<value_t<LE>>, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with left typing and a
 * direct operation
 */
template <typename LE, typename RE, typename OP>
using left_binary_helper_op = binary_expr<value_t<LE>, build_type<LE>, OP, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with right typing
 */
template <typename LE, typename RE, template <typename> class OP>
using right_binary_helper = binary_expr<value_t<RE>, build_type<LE>, OP<value_t<RE>>, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with right typing and a
 * direct operation
 */
template <typename LE, typename RE, typename OP>
using right_binary_helper_op = binary_expr<value_t<RE>, build_type<LE>, OP, build_type<RE>>;

/*!
 * \brief Helper to create an unary expression
 */
template <typename E, template <typename> class OP>
using unary_helper = unary_expr<value_t<E>, build_type<E>, OP<value_t<E>>>;

/*!
 * \brief Helper to create an identity unary expression
 */
template <typename E, typename OP>
using identity_helper = unary_expr<value_t<E>, OP, identity_op>;

/*!
 * \brief Helper to create a virtual unary expression
 */
template <typename E, typename OP>
using virtual_helper = unary_expr<E, OP, transform_op>;

/*!
 * \brief Helper to create a stable transform unary expression
 */
template <typename E, template <typename> class OP>
using stable_transform_helper = unary_expr<value_t<E>, OP<build_type<E>>, transform_op>;

/*!
 * \brief Helper to create a stable binary transform unary expression
 */
template <typename LE, typename RE, template <typename, typename> class OP>
using stable_transform_binary_helper = unary_expr<value_t<LE>, OP<build_type<LE>, build_type<RE>>, transform_op>;

/*!
 * \brief Make a stable unary transform unary expression
 * \param args Arguments to be forward to the op.
 */
template <typename E, template <typename> class OP, typename... Args>
auto make_transform_expr(Args&&... args) {
    return stable_transform_helper<E, OP>{OP<build_type<E>>(std::forward<Args>(args)...)};
}

/*!
 * \brief Make a stable unary transform unary expression for
 * a stateful op.
 * \param args Arguments to be forward to the op.
 */
template <typename E, typename OP, typename... Args>
auto make_stateful_unary_expr(Args&&... args) {
    return unary_expr<value_t<E>, build_type<E>, stateful_op<OP>>(std::forward<Args>(args)...);
}

/*!
 * \brief Helper to create a temporary binary expression.
 */
template <typename A, typename B, template <typename> class OP>
using temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>>>;

/*!
 * \brief Helper to create a temporary binary expression with an
 * operation that takes a number of dimensions as input template
 * type.
 */
template <typename A, typename B, template <typename, size_t> class OP, size_t D>
using dim_temporary_binary_helper = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP<value_t<A>, D>>;

/*!
 * \brief Helper to create a temporary binary expression with
 * a direct op.
 */
template <typename A, typename B, typename OP>
using temporary_binary_helper_op = temporary_binary_expr<value_t<A>, build_type<A>, build_type<B>, OP>;

} //end of namespace detail

} //end of namespace etl
