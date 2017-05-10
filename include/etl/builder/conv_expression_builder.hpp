//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
auto conv_4d_valid(A&& a, B&& b, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    using op_t = dyn_conv4_valid_expr<value_t<A>>;
    return temporary_binary_expr_state<value_t<A>, detail::build_type<A>, detail::build_type<B>, op_t>{op_t(s1, s2, p1, p2), a, b};
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
auto conv_4d_valid(A&& a, B&& b, C&& c, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid(a, b, s1, s2, p1, p2);
    return c;
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
auto conv_4d_valid_flipped(A&& a, B&& b, size_t s1, size_t s2, size_t p1, size_t p2){
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    using op_t = dyn_conv4_valid_flipped_expr<value_t<A>>;
    return temporary_binary_expr_state<value_t<A>, detail::build_type<A>, detail::build_type<B>, op_t>{op_t(s1, s2, p1, p2), a, b};
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
auto conv_4d_valid_flipped(A&& a, B&& b, C&& c, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_flipped(a, b, s1, s2, p1, p2);
    return c;
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, K, H, W] dimensions.
 * The 4D matrix b is assumed to be of [K, C, H, W] dimensions.
 * The 4D matrix c is assumed to be of [N, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
auto conv_4d_valid_back(A&& a, B&& b) -> detail::temporary_binary_helper_op<A, B, conv4_valid_back_expr<value_t<A>, S1, S2, P1, P2>> {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, K, H, W] dimensions.
 * The 4D matrix b is assumed to be of [K, C, H, W] dimensions.
 * The 4D matrix c is assumed to be of [N, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The output expression
 *
 * \return c
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B, typename C>
auto conv_4d_valid_back(A&& a, B&& b, C&& c) {
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_back<S1, S2, P1, P2>(a, b);
    return c;
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, K, H, W] dimensions.
 * The 4D matrix b is assumed to be of [K, C, H, W] dimensions.
 * The 4D matrix c is assumed to be of [N, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the results of the convolutions.
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
auto conv_4d_valid_back_flipped(A&& a, B&& b) -> detail::temporary_binary_helper_op<A, B, conv4_valid_back_flipped_expr<value_t<A>, S1, S2, P1, P2>> {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    return {a, b};
}

/*!
 * \brief Generic 4D convolution of a with the kernels from b
 *
 * The 4D matrix a is assumed to be of [N, K, H, W] dimensions.
 * The 4D matrix b is assumed to be of [K, C, H, W] dimensions.
 * The 4D matrix c is assumed to be of [N, C, H, W] dimensions.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The output expression
 *
 * \return c
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B, typename C>
auto conv_4d_valid_back_flipped(A&& a, B&& b, C&& c) {
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_back_flipped<S1, S2, P1, P2>(a, b);
    return c;
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
auto conv_4d_valid_back(A&& a, B&& b, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    using op_t = dyn_conv4_valid_back_expr<value_t<A>>;
    return temporary_binary_expr_state<value_t<A>, detail::build_type<A>, detail::build_type<B>, op_t>{op_t(s1, s2, p1, p2), a, b};
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
auto conv_4d_valid_back(A&& a, B&& b, C&& c, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_back(a, b, s1, s2, p1, p2);
    return c;
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
auto conv_4d_valid_back_flipped(A&& a, B&& b, size_t s1, size_t s2, size_t p1, size_t p2){
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    using op_t = dyn_conv4_valid_back_flipped_expr<value_t<A>>;
    return temporary_binary_expr_state<value_t<A>, detail::build_type<A>, detail::build_type<B>, op_t>{op_t(s1, s2, p1, p2), a, b};
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
auto conv_4d_valid_back_flipped(A&& a, B&& b, C&& c, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_back_flipped(a, b, s1, s2, p1, p2);
    return c;
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
auto conv_4d_valid_filter(A&& a, B&& b, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    using op_t = dyn_conv4_valid_filter_expr<value_t<A>>;
    return temporary_binary_expr_state<value_t<A>, detail::build_type<A>, detail::build_type<B>, op_t>{op_t(s1, s2, p1, p2), a, b};
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
auto conv_4d_valid_filter(A&& a, B&& b, C&& c, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_filter(a, b, s1, s2, p1, p2);
    return c;
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
auto conv_4d_valid_filter_flipped(A&& a, B&& b, size_t s1, size_t s2, size_t p1, size_t p2){
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    using op_t = dyn_conv4_valid_filter_flipped_expr<value_t<A>>;
    return temporary_binary_expr_state<value_t<A>, detail::build_type<A>, detail::build_type<B>, op_t>{op_t(s1, s2, p1, p2), a, b};
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
auto conv_4d_valid_filter_flipped(A&& a, B&& b, C&& c, size_t s1, size_t s2, size_t p1, size_t p2) {
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_4d_valid_filter_flipped(a, b, s1, s2, p1, p2);
    return c;
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
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

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
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_deep_valid(a, b);
    return c;
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
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

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
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_deep_same(a, b);
    return c;
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
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

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
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_deep_full(a, b);
    return c;
}

//Special convolutions

/*!
 * \brief Construct a matrix to compute a convolution by matrix-matrix multiplication
 * \param a The vector to transform (the input of the convolution)
 * \param h The size of kernel
 * \return a matrix expression for convolution
 */
template <typename A>
auto convmtx(A&& a, size_t h) -> detail::stable_transform_helper<A, dyn_convmtx_transformer> {
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
auto convmtx2(A&& a, size_t k1, size_t k2) -> detail::stable_transform_helper<A, dyn_convmtx2_transformer> {
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
template <size_t K1, size_t K2, typename A>
convmtx_2d_expr<A, K1, K2> convmtx2_direct(A&& a) {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2, "Convolutional matrix only works in 2D");

    return convmtx_2d_expr<A, K1, K2>{a};
}

} //end of namespace etl
