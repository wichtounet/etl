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

namespace etl {

/*!
 * \brief Builds an expression representing the subtraction of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the subtraction of lhs and rhs
 */
template <etl_expr LE, etl_expr RE>
auto operator-(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);

    return detail::left_binary_helper<LE, RE, minus_binary_op>{lhs, rhs};
}

/*!
 * \brief Builds an expression representing the addition of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the addition of lhs and rhs
 */
template <etl_expr LE, etl_expr RE>
auto operator+(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);

    return detail::left_binary_helper<LE, RE, plus_binary_op>{lhs, rhs};
}

/*!
 * \brief Builds an expression representing the scalar multipliation of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the scalar multipliation of lhs and rhs
 */
template <etl_expr LE, etl_expr RE>
auto operator>>(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);

    return detail::left_binary_helper<LE, RE, mul_binary_op>{lhs, rhs};
}

/*!
 * \brief Builds an expression representing the scalar multiplication of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the scalar multiplication of lhs and rhs
 */
template <typename LE, typename RE>
auto scale(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);

    return detail::left_binary_helper<LE, RE, mul_binary_op>{lhs, rhs};
}

/*!
 * \brief Builds an expression representing the division of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the division of lhs and rhs
 */
template <etl_expr LE, etl_expr RE>
auto operator/(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);

    return detail::left_binary_helper<LE, RE, div_binary_op>{lhs, rhs};
}

/*!
 * \brief Builds an expression representing the modulo of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the modulo of lhs and rhs
 */
template <etl_expr LE, etl_expr RE>
auto operator%(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);

    return detail::left_binary_helper<LE, RE, mod_binary_op>{lhs, rhs};
}

// Mix scalars and ETL expressions (vector,matrix,binary,unary)

/*!
 * \brief Builds an expression representing the subtraction of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the subtraction of lhs and rhs (scalar)
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE>)>
auto operator-(LE&& lhs, RE rhs) {
    return detail::left_binary_helper<LE, scalar<value_t<LE>>, minus_binary_op>{lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the subtraction of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the subtraction of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto operator-(LE lhs, RE&& rhs) {
    return detail::right_binary_helper<scalar<value_t<RE>>, RE, minus_binary_op>{scalar<value_t<RE>>(lhs), rhs};
}

/*!
 * \brief Builds an expression representing the addition of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the addition of lhs and rhs (scalar)
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE>)>
auto operator+(LE&& lhs, RE rhs) {
    return detail::left_binary_helper<LE, scalar<value_t<LE>>, plus_binary_op> {lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the addition of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the addition of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto operator+(LE lhs, RE&& rhs) {
    return detail::right_binary_helper<scalar<value_t<RE>>, RE, plus_binary_op> {scalar<value_t<RE>>(lhs), rhs};
}

/*!
 * \brief Builds an expression representing the multiplication of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the multiplication of lhs and rhs (scalar)
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE>)>
auto operator*(LE&& lhs, RE rhs) {
    return detail::left_binary_helper<LE, scalar<value_t<LE>>, mul_binary_op> {lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the multiplication of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the multiplication of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto operator*(LE lhs, RE&& rhs) {
    return detail::right_binary_helper<scalar<value_t<RE>>, RE, mul_binary_op> {scalar<value_t<RE>>(lhs), rhs};
}

/*!
 * \brief Builds an expression representing the multiplication of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the multiplication of lhs and rhs (scalar)
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE>)>
auto operator>>(LE&& lhs, RE rhs) {
    return detail::left_binary_helper<LE, scalar<value_t<LE>>, mul_binary_op> {lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the multiplication of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the multiplication of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto operator>>(LE lhs, RE&& rhs) {
    return detail::right_binary_helper<scalar<value_t<RE>>, RE, mul_binary_op> {scalar<value_t<RE>>(lhs), rhs};
}

/*!
 * \brief Builds an expression representing the division of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the division of lhs and rhs (scalar)
 */
template <typename LE,
          typename RE,
          cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE> && (is_div_strict || !std::is_floating_point_v<RE>))>
auto operator/(LE&& lhs, RE rhs) {
    return detail::left_binary_helper<LE, scalar<value_t<LE>>, div_binary_op> {lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the division of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the division of lhs and rhs (scalar)
 */
template <typename LE,
          typename RE,
          cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE> && !is_div_strict && std::is_floating_point_v<RE>)>
auto operator/(LE&& lhs, RE rhs) {
    return detail::left_binary_helper<LE, scalar<value_t<LE>>, mul_binary_op> {lhs, scalar<value_t<LE>>(value_t<LE>(1.0) / rhs)};
}

/*!
 * \brief Builds an expression representing the division of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the division of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto operator/(LE lhs, RE&& rhs) {
    return detail::right_binary_helper<scalar<value_t<RE>>, RE, div_binary_op> {scalar<value_t<RE>>(lhs), rhs};
}

/*!
 * \brief Builds an expression representing the modulo of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the modulo of lhs and rhs (scalar)
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE>)>
auto operator%(LE&& lhs, RE rhs) {
    return detail::left_binary_helper<LE, scalar<value_t<LE>>, mod_binary_op> {lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the modulo of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the modulo of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto operator%(LE lhs, RE&& rhs) {
    return detail::right_binary_helper<scalar<value_t<RE>>, RE, mod_binary_op> {scalar<value_t<RE>>(lhs), rhs};
}

// Compound operators

// TODO: Generalize the change to std::forward and decltype(auto)

/*!
 * \brief Compound addition of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_arithmetic_v<RE> && is_simple_lhs<LE>)>
decltype(auto) operator+=(LE&& lhs, RE rhs) {
    etl::scalar<RE>(rhs).assign_add_to(lhs);
    return std::forward<LE>(lhs);
}

/*!
 * \brief Compound addition of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(is_etl_expr<RE>&& is_simple_lhs<LE>)>
decltype(auto) operator+=(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);
    rhs.assign_add_to(lhs);
    return std::forward<LE>(lhs);
}

/*!
 * \brief Compound subtraction of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_arithmetic_v<RE> && is_simple_lhs<LE>)>
LE& operator-=(LE&& lhs, RE rhs) {
    etl::scalar<RE>(rhs).assign_sub_to(lhs);
    return lhs;
}

/*!
 * \brief Compound subtraction of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(is_etl_expr<RE>&& is_simple_lhs<LE>)>
LE& operator-=(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);
    rhs.assign_sub_to(lhs);
    return lhs;
}

/*!
 * \brief Compound multiplication of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_arithmetic_v<RE> && is_simple_lhs<LE>)>
LE& operator*=(LE&& lhs, RE rhs) {
    etl::scalar<RE>(rhs).assign_mul_to(lhs);
    return lhs;
}

/*!
 * \brief Compound multiplication of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(is_etl_expr<RE>&& is_simple_lhs<LE>)>
LE& operator*=(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);
    rhs.assign_mul_to(lhs);
    return lhs;
}

/*!
 * \brief Compound multiplication of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_arithmetic_v<RE> && is_simple_lhs<LE>)>
LE& operator>>=(LE&& lhs, RE rhs) {
    etl::scalar<RE>(rhs).assign_mul_to(lhs);
    return lhs;
}

/*!
 * \brief Compound multiplication of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(is_etl_expr<RE>&& is_simple_lhs<LE>)>
LE& operator>>=(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);
    rhs.assign_mul_to(lhs);
    return lhs;
}

/*!
 * \brief Compound division of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_arithmetic_v<RE> && is_simple_lhs<LE>)>
LE& operator/=(LE&& lhs, RE rhs) {
    etl::scalar<RE>(rhs).assign_div_to(lhs);
    return lhs;
}

/*!
 * \brief Compound division of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(is_etl_expr<RE>&& is_simple_lhs<LE>)>
LE& operator/=(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);
    rhs.assign_div_to(lhs);
    return lhs;
}

/*!
 * \brief Compound modulo of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_arithmetic_v<RE> && is_simple_lhs<LE>)>
LE& operator%=(LE&& lhs, RE rhs) {
    etl::scalar<RE>(rhs).assign_mod_to(lhs);
    return lhs;
}

/*!
 * \brief Compound modulo of the right hand side to the left hand side
 * \param lhs The left hand side, will be changed
 * \param rhs The right hand side
 * \return the left hand side
 */
template <typename LE, typename RE, cpp_enable_iff(is_etl_expr<RE>&& is_simple_lhs<LE>)>
LE& operator%=(LE&& lhs, RE&& rhs) {
    validate_expression(lhs, rhs);
    rhs.assign_mod_to(lhs);
    return lhs;
}

// Comparison

/*!
 * \brief Builds an expression representing the elementwise comparison of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise comparison of lhs and rhs
 */
template <typename LE, typename RE>
auto equal(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper_scalar<LE, RE, equal_binary_op> {detail::wrap_scalar(lhs), detail::wrap_scalar(rhs)};
}

/*!
 * \brief Builds an expression representing the elementwise comparison of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise comparison of lhs and rhs
 */
template <typename LE, typename RE>
auto not_equal(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper_scalar<LE, RE, not_equal_binary_op> {detail::wrap_scalar(lhs), detail::wrap_scalar(rhs)};
}

/*!
 * \brief Builds an expression representing the elementwise less than comparison of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise less than comparison of lhs and rhs
 */
template <typename LE, typename RE>
auto less(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper_scalar<LE, RE, less_binary_op> {detail::wrap_scalar(lhs), detail::wrap_scalar(rhs)};
}

/*!
 * \brief Builds an expression representing the elementwise less than or equals comparison of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise less than or equals comparison of lhs and rhs
 */
template <typename LE, typename RE>
auto less_equal(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper_scalar<LE, RE, less_equal_binary_op> {detail::wrap_scalar(lhs), detail::wrap_scalar(rhs)};
}

/*!
 * \brief Builds an expression representing the elementwise greater than comparison of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise greater than comparison of lhs and rhs
 */
template <typename LE, typename RE>
auto greater(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper_scalar<LE, RE, greater_binary_op> {detail::wrap_scalar(lhs), detail::wrap_scalar(rhs)};
}

/*!
 * \brief Builds an expression representing the elementwise greater than or equals comparison of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise greater than or equals comparison of lhs and rhs
 */
template <typename LE, typename RE>
auto greater_equal(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper_scalar<LE, RE, greater_equal_binary_op> {detail::wrap_scalar(lhs), detail::wrap_scalar(rhs)};
}

// Logical operators

/*!
 * \brief Builds an expression representing the elementwise logical and of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical and of lhs and rhs
 */
template <etl_expr LE, etl_expr RE>
auto logical_and(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper<LE, LE, logical_and_binary_op> {lhs, rhs};
}

/*!
 * \brief Builds an expression representing the elementwise logical and of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical and of lhs and rhs (scalar)
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE>)>
auto logical_and(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper<LE, scalar<value_t<LE>>, logical_and_binary_op> {lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the elementwise logical and of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical and of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto logical_and(LE lhs, RE&& rhs) {
    return detail::bool_right_binary_helper<scalar<value_t<RE>>, RE, logical_and_binary_op> {scalar<value_t<RE>>(lhs), rhs};
}

/*!
 * \brief Builds an expression representing the elementwise logical xor of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical xor of lhs and rhs
 */
template <etl_expr LE, etl_expr RE>
auto logical_xor(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper<LE, LE, logical_xor_binary_op> {lhs, rhs};
}

/*!
 * \brief Builds an expression representing the elementwise logical xor of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical xor of lhs and rhs (scalar)
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE>)>
auto logical_xor(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper<LE, scalar<value_t<LE>>, logical_xor_binary_op> {lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the elementwise logical xor of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical xor of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto logical_xor(LE lhs, RE&& rhs) {
    return detail::bool_right_binary_helper<scalar<value_t<RE>>, RE, logical_xor_binary_op> {scalar<value_t<RE>>(lhs), rhs};
}

/*!
 * \brief Builds an expression representing the elementwise logical or of lhs and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical or of lhs and rhs
 */
template <etl_expr LE, etl_expr RE>
auto logical_or(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper<LE, LE, logical_or_binary_op> {lhs, rhs};
}

/*!
 * \brief Builds an expression representing the elementwise logical or of lhs and rhs (scalar)
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical or of lhs and rhs (scalar)
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<RE, value_t<LE>> && is_etl_expr<LE>)>
auto logical_or(LE&& lhs, RE rhs) {
    return detail::bool_left_binary_helper<LE, scalar<value_t<LE>>, logical_or_binary_op> {lhs, scalar<value_t<LE>>(rhs)};
}

/*!
 * \brief Builds an expression representing the elementwise logical or of lhs (scalar) and rhs
 * \param lhs The left hand side expression
 * \param rhs The right hand side expression
 * \return An expression representing the element wise logical or of lhs (scalar) and rhs
 */
template <typename LE, typename RE, cpp_enable_iff(std::is_convertible_v<LE, value_t<RE>> && is_etl_expr<RE>)>
auto logical_or(LE lhs, RE&& rhs) {
    return detail::bool_right_binary_helper<scalar<value_t<RE>>, RE, logical_or_binary_op> {scalar<value_t<RE>>(lhs), rhs};
}

} //end of namespace etl
