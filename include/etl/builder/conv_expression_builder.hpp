//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file conv_expression_builder.hpp
 * \brief Contains all the operators and functions to build convolution expressions.
*/

#pragma once

#include "etl/expression_helpers.hpp"

namespace etl {

/*!
 * \brief Creates an expression representing the valid 1D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the valid 1D convolution of a and b
 */
template <typename A, typename B>
auto conv_1d_valid(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv1_valid_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the valid 1D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the valid 1D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_1d_valid(A&& a, B&& b, C&& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_1d_valid(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the same 1D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the same 1D convolution of a and b
 */
template <typename A, typename B>
auto conv_1d_same(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv1_same_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the same 1D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the same 1D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_1d_same(A&& a, B&& b, C&& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_1d_same(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the full 1D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the full 1D convolution of a and b
 */
template <typename A, typename B>
auto conv_1d_full(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv1_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the full 1D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the full 1D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_1d_full(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_1d_full(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the valid 2D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the valid 2D convolution of a and b
 */
template <size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
auto conv_2d_valid(A&& a, B&& b) -> detail::temporary_binary_helper_op<A, B, conv2_valid_expr<value_t<A>, S1, S2, P1, P2>> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the valid 2D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the valid 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_2d_valid(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_valid(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the valid 2D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the valid 2D convolution of a and b
 */
template <typename A, typename B>
auto conv_2d_valid_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_valid_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the valid 2D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the valid 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_2d_valid_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_valid_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the valid 2D convolution of a and multiple kernels from b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the valid 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B>
auto conv_2d_valid_multi(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_valid_multi_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the valid 2D convolution of a and multiple kernels from b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expressions
 * \param c The result
 * \return an expression representing the valid 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B, typename C>
auto conv_2d_valid_multi(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_valid_multi(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the valid 2D convolution of a and multiple flipped kernels from b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the valid 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B>
auto conv_2d_valid_multi_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_valid_multi_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the valid 2D convolution of a and multiple flipped kernels from b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expressions
 * \param c The result
 * \return an expression representing the valid 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B, typename C>
auto conv_2d_valid_multi_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_valid_multi_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, C, H, W] dimensions.
 * The 4D matrix b is assumed to be of [K, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <typename A, typename B>
auto conv_4d_valid(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv4_valid_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, C, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hf, Wf] dimensions.
 * The 4D matrix c is assumed to be of [N, K, Hi - Hf + 1, Wi - Wf + 1] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The output expression
 *
 * \return c
 */
template <typename A, typename B, typename C>
auto conv_4d_valid(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, C, H, W] dimensions.
 * The 4D matrix b is assumed to be of [K, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <typename A, typename B>
auto conv_4d_full(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv4_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, C, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hf, Wf] dimensions.
 * The 4D matrix c is assumed to be of [N, K, Hi - Hf + 1, Wi - Wf + 1] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The output expression
 *
 * \return c
 */
template <typename A, typename B, typename C>
auto conv_4d_full(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_full(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, C, H, W] dimensions.
 * The 4D matrix b is assumed to be of [K, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <typename A, typename B>
auto conv_4d_valid_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv4_valid_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, C, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hf, Wf] dimensions.
 * The 4D matrix c is assumed to be of [N, K, Hi - Hf + 1, Wi - Wf + 1] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The output expression
 *
 * \return c
 */
template <typename A, typename B, typename C>
auto conv_4d_valid_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b, the
 * output is assumed to be filters.
 *
 * The 4D matrix a is assumed to be of [N, C, H, W] dimensions.
 * The 4D matrix b is assumed to be of [N, K, H, W] dimensions.
 * The 4D matrix c is assumed to be of [K, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <typename A, typename B>
auto conv_4d_valid_filter(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv4_valid_filter_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b, the
 * output is assumed to be filters.
 *
 * The 4D matrix a is assumed to be of [N, C, H, W] dimensions.
 * The 4D matrix b is assumed to be of [N, K, H, W] dimensions.
 * The 4D matrix c is assumed to be of [K, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The output expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <typename A, typename B, typename C>
auto conv_4d_valid_filter(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_filter(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b, the
 * output is assumed to be filters.
 *
 * The 4D matrix a is assumed to be of [N, C, H, W] dimensions.
 * The 4D matrix b is assumed to be of [N, K, H, W] dimensions.
 * The 4D matrix c is assumed to be of [K, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <typename A, typename B>
auto conv_4d_valid_filter_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv4_valid_filter_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b, the
 * output is assumed to be filters.
 *
 * The 4D matrix a is assumed to be of [N, C, H, W] dimensions.
 * The 4D matrix b is assumed to be of [N, K, H, W] dimensions.
 * The 4D matrix c is assumed to be of [K, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The output expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <typename A, typename B, typename C>
auto conv_4d_valid_filter_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_filter_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, C, H, W] dimensions.
 * The 4D matrix b is assumed to be of [K, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <typename A, typename B>
auto conv_4d_full_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv4_full_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, C, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hf, Wf] dimensions.
 * The 4D matrix c is assumed to be of [N, K, Hi - Hf + 1, Wi - Wf + 1] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The output expression
 *
 * \return c
 */
template <typename A, typename B, typename C>
auto conv_4d_full_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_full_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the same 2D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the same 2D convolution of a and b
 */
template <typename A, typename B>
auto conv_2d_same(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_same_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the same 2D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the same 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_2d_same(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_same(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the same 2D convolution of a and b,
 * with flipped kernels.
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the same 2D convolution of a and b
 */
template <typename A, typename B>
auto conv_2d_same_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_same_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the same 2D convolution of a and b, the result will be stored in c, with flipped kernels.
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the same 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_2d_same_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_same_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the same 2D convolution of a and multiple kernels from b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the same 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B>
auto conv_2d_same_multi(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_same_multi_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the same 2D convolution of a and multiple kernels from b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expressions
 * \param c The result
 * \return an expression representing the same 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B, typename C>
auto conv_2d_same_multi(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_same_multi(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the same 2D convolution of a and multiple flipped kernels from b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the same 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B>
auto conv_2d_same_multi_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_same_multi_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the same 2D convolution of a and multiple flipped kernels from b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expressions
 * \param c The result
 * \return an expression representing the same 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B, typename C>
auto conv_2d_same_multi_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_same_multi_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the full 2D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the full 2D convolution of a and b
 */
template <typename A, typename B>
auto conv_2d_full(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_full_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the full 2D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the full 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_2d_full(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_full(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the full 2D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the full 2D convolution of a and b
 */
template <typename A, typename B>
auto conv_2d_full_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_full_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the full 2D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the full 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_2d_full_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_full_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the full 2D convolution of a and multiple kernels from b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the full 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B>
auto conv_2d_full_multi(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_full_multi_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the full 2D convolution of a and multiple kernels from b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expressions
 * \param c The result
 * \return an expression representing the full 2D convolution of a and multiple kernels from b
 */
template <typename A, typename B, typename C>
auto conv_2d_full_multi(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_full_multi(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing the full 2D convolution of a and multiple flipped kernels from b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the full 2D convolution of a and multiple flipped kernels from b
 */
template <typename A, typename B>
auto conv_2d_full_multi_flipped(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, conv2_full_multi_flipped_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing the full 2D convolution of a and multiple flipped kernels from b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expressions
 * \param c The result
 * \return an expression representing the full 2D convolution of a and multiple flipped kernels from b
 */
template <typename A, typename B, typename C>
auto conv_2d_full_multi_flipped(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_full_multi_flipped(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing many valid 2D convolution of a and b.
 *
 * Only the last two dimensions are used for the convolution itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing many valid 2D convolution of a and b
 */
template <typename A, typename B>
auto conv_deep_valid(A&& a, B&& b) -> detail::dim_temporary_binary_helper<A, B, conv_deep_valid_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing many valid 2D convolution of a and b, the result is stored in c
 *
 * Only the last two dimensions are used for the convolution itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing many valid 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_deep_valid(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_deep_valid(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing many same 2D convolution of a and b.
 *
 * Only the last two dimensions are used for the convolution itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing many same 2D convolution of a and b
 */
template <typename A, typename B>
auto conv_deep_same(A&& a, B&& b) -> detail::dim_temporary_binary_helper<A, B, conv_deep_same_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing many same 2D convolution of a and b, the result is stored in c
 *
 * Only the last two dimensions are used for the convolution itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing many same 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_deep_same(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_deep_same(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Creates an expression representing many full 2D convolution of a and b.
 *
 * Only the last two dimensions are used for the convolution itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing many full 2D convolution of a and b
 */
template <typename A, typename B>
auto conv_deep_full(A&& a, B&& b) -> detail::dim_temporary_binary_helper<A, B, conv_deep_full_expr, decay_traits<A>::dimensions()> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Creates an expression representing many full 2D convolution of a and b, the result is stored in c
 *
 * Only the last two dimensions are used for the convolution itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing many full 2D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_deep_full(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    c = conv_deep_full(a, b);
    return std::forward<C>(c);
}

//Special convolutions

/*!
 * \brief Construct a matrix to compute a convolution by matrix-matrix multiplication
 * \param a The vector to transform (the input of the convolution)
 * \param h The size of kernel
 * \return a matrix expression for convolution
 */
template <typename A>
auto convmtx(A&& a, std::size_t h) -> detail::stable_transform_helper<A, dyn_convmtx_transformer> {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 1, "Convolutional matrix only works in 1D");

    return detail::stable_transform_helper<A, dyn_convmtx_transformer>{dyn_convmtx_transformer<detail::build_type<A>>(a, h)};
}

/*!
 * \brief Construct a matrix to compute a 2D convolution by matrix-matrix multiplication
 * \param a The 2D matrix to transform (the input of the convolution)
 * \param k1 The first dimension of the kernel
 * \param k2 The second dimension of the kernel
 * \return a matrix expression for convolution
 */
template <typename A>
auto convmtx2(A&& a, std::size_t k1, std::size_t k2) -> detail::stable_transform_helper<A, dyn_convmtx2_transformer> {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2, "Convolutional matrix only works in 2D");

    return detail::stable_transform_helper<A, dyn_convmtx2_transformer>{dyn_convmtx2_transformer<detail::build_type<A>>(a, k1, k2)};
}

/*!
 * \brief Construct a matrix to compute a 2D convolution by matrix-matrix multiplication
 * \param a The 2D matrix to transform (the input of the convolution)
 * \tparam K1 The first dimension of the kernel
 * \tparam K2 The second dimension of the kernel
 * \return a matrix expression for convolution
 */
template <std::size_t K1, std::size_t K2, typename A>
auto convmtx2_direct(A&& a) -> temporary_unary_expr<value_t<A>, detail::build_type<A>, direct_convmtx2_expr<value_t<A>, K1, K2>> {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2, "Convolutional matrix only works in 2D");

    return temporary_unary_expr<value_t<A>, detail::build_type<A>, direct_convmtx2_expr<value_t<A>, K1, K2>>{a};
}

} //end of namespace etl
