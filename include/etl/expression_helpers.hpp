//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file expression_helpers.hpp
 * \brief Contains internal helpers to build expressions.
 */

#pragma once

namespace etl::detail {

/*!
 * \brief Helper to build the type for a sub expression
 *
 * This means a reference for a value type and a copy for another
 * expression.
 */
template <typename T>
using build_type = std::conditional_t<is_etl_value<T>, const std::decay_t<T>&, std::decay_t<T>>;

/*!
 * \brief Helper to build the identity type for a sub expression
 *
 * This means a reference for a value type (preserving constness)
 * and a copy for another expression.
 */
template <typename T>
using build_identity_type = std::conditional_t<is_etl_value<T>,
                                               std::conditional_t<std::is_const_v<std::remove_reference_t<T>>, const std::decay_t<T>&, std::decay_t<T>&>,
                                               std::decay_t<T>>;

/*!
 * \brief Wraps a type either into a scalar or keep the ETL expression.
 */
template <typename T>
using wrap_scalar_t = std::conditional_t<etl::is_etl_expr<T>, T, etl::scalar<std::decay_t<T>>>;

/*!
 * \brief Wraps a type either into a scalar or keep the ETL expression.
 *
 * If the type is not an ETL expression, we use the type of the hint
 * in order to create the correct scalar type.
 */
template <typename H, typename T>
using smart_wrap_scalar_t = std::conditional_t<etl::is_etl_expr<T>, T, etl::scalar<etl::value_t<H>>>;

/*!
 * \brief Extract the value type of the given type taking scalar into account
 */
template <typename T, typename Enable = void>
struct wrap_scalar_value_t_impl;

/*!
 * \brief Extract the value type of the given type taking scalar into account
 */
template <typename T>
struct wrap_scalar_value_t_impl<T, std::enable_if_t<etl::is_etl_expr<T>>> {
    /*!
     * \brief The resulting type of the traits.
     */
    using type = etl::value_t<T>;
};

/*!
 * \brief Extract the value type of the given type taking scalar into account
 */
template <typename T>
struct wrap_scalar_value_t_impl<T, std::enable_if_t<!etl::is_etl_expr<T>>> {
    /*!
     * \brief The resulting type of the traits.
     */
    using type = std::decay_t<T>;
};

/*!
 * \brief Extract the value type of the given type taking scalar into account
 */
template <typename T>
using wrap_scalar_value_t = typename wrap_scalar_value_t_impl<T>::type;

/*!
 * \brief Transform a scalar value into an etl::scalar
 * \param value The value to wraps
 * \return an etl::scalar or a forwarded expression
 */
template <typename T, cpp_enable_iff(is_etl_expr<T>)>
decltype(auto) wrap_scalar(T&& value) {
    return std::forward<T>(value);
}

/*!
 * \brief Transform a scalar value into an etl::scalar
 * \param value The value to wraps
 * \return an etl::scalar or a forwarded expression
 */
template <typename T, cpp_enable_iff(!is_etl_expr<T>)>
etl::scalar<std::decay_t<T>> wrap_scalar(T&& value) {
    return etl::scalar<std::decay_t<T>>{value};
}

/*!
 * \brief Transform a scalar value into an etl::scalar
 * \param value The value to wraps
 * \return an etl::scalar or a forwarded expression
 */
template <typename Hint, typename T>
decltype(auto) smart_wrap_scalar(T&& value) {
    if constexpr (is_etl_expr<T>) {
        return std::forward<T>(value);
    } else {
        using scalar_type = etl::value_t<Hint>;
        return etl::scalar<scalar_type>{scalar_type(value)};
    }
}

/*!
 * \brief Helper to create a binary expr with left typing
 */
template <typename LE, typename RE, template <typename> typename OP>
using left_binary_helper = binary_expr<value_t<LE>, build_type<LE>, OP<value_t<LE>>, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with left typing and a
 * direct operation
 */
template <typename LE, typename RE, typename OP>
using left_binary_helper_op = binary_expr<value_t<LE>, build_type<LE>, OP, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with left typing and a
 * direct operation
 */
template <typename LE, typename RE, typename OP>
using left_binary_helper_op_scalar = binary_expr<wrap_scalar_value_t<LE>, build_type<wrap_scalar_t<LE>>, OP, build_type<smart_wrap_scalar_t<LE, RE>>>;

/*!
 * \brief Helper to create a binary expr with right typing
 */
template <typename LE, typename RE, template <typename> typename OP>
using right_binary_helper = binary_expr<value_t<RE>, build_type<LE>, OP<value_t<RE>>, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with right typing and a
 * direct operation
 */
template <typename LE, typename RE, typename OP>
using right_binary_helper_op = binary_expr<value_t<RE>, build_type<LE>, OP, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with left typing
 */
template <typename LE, typename RE, template <typename> typename OP>
using bool_left_binary_helper = binary_expr<bool, build_type<LE>, OP<value_t<LE>>, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with right typing
 */
template <typename LE, typename RE, template <typename> typename OP>
using bool_right_binary_helper = binary_expr<bool, build_type<LE>, OP<value_t<RE>>, build_type<RE>>;

/*!
 * \brief Helper to create a binary expr with left typing
 */
template <typename LE, typename RE, template <typename> typename OP>
using bool_left_binary_helper_scalar = binary_expr<bool, build_type<wrap_scalar_t<LE>>, OP<wrap_scalar_value_t<LE>>, build_type<wrap_scalar_t<RE>>>;

/*!
 * \brief Helper to create an unary expression
 */
template <typename E, template <typename> typename OP>
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
template <typename E, template <typename> typename OP>
using stable_transform_helper = unary_expr<value_t<E>, OP<build_type<E>>, transform_op>;

/*!
 * \brief Helper to create a stable binary transform unary expression
 */
template <typename LE, typename RE, template <typename, typename> typename OP>
using stable_transform_binary_helper = unary_expr<value_t<LE>, OP<build_type<LE>, build_type<RE>>, transform_op>;

/*!
 * \brief Make a stable unary transform unary expression
 * \param args Arguments to be forward to the op.
 */
template <typename E, template <typename> typename OP, typename... Args>
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

} //end of namespace etl::detail
