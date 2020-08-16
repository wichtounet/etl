//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains some special helpers for machine learning.
 *
 * This is mostly a simpler set of names and functions to achieve
 * machine learning features.
 */

#pragma once

#include "etl/impl/cce.hpp"
#include "etl/impl/bce.hpp"
#include "etl/impl/mse.hpp"

namespace etl::ml {

// Convolution wrappers

/*!
 * \brief Forward convolution for a batch of images with a set of kernels.
 *
 * This will compute the 2D convolutions of each image with each given
 * kernels. The results accross channels will be accumulated together.
 *
 * The 4D matrix a is assumed to be of [N, C, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hj, Wj] dimensions.
 * The 4D matrix c is assumed to be of [N, K, (Hi - Hj + 2 * P1) / S1 + 1, (Wi - Hj + 2 * P2) / S2 + 1] dimensions.
 *
 * \param a An expression containing the batch of images
 * \param b An expression containing the set of kernels
 *
 * \tparam S1 The stride in the first dimension
 * \tparam S2 The stride in the second dimension
 * \tparam P1 The padding of the first dimension
 * \tparam P2 The padding of the second dimension
 *
 * \return an expression representing the result of the forward convolution
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
conv_4d_valid_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true> convolution_forward(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return conv_4d_valid_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true>{a, b};
}

/*!
 * \brief Forward convolution for a batch of images with a set of kernels.
 *
 * This will compute the 2D convolutions of each image with each given
 * kernels. The results accross channels will be accumulated together.
 *
 * The 4D matrix a is assumed to be of [N, C, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hj, Wj] dimensions.
 * The 4D matrix c is assumed to be of [N, K, (Hi - Hj + 2 * P1) / S1 + 1, (Wi - Hj + 2 * P2) / S2 + 1] dimensions.
 *
 * \param a An expression containing the batch of images
 * \param b An expression containing the set of kernels
 *
 * \param s1 The stride in the first dimension
 * \param s2 The stride in the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 *
 * \return an expression representing the result of the forward convolution
 */
template <typename A, typename B>
dyn_conv_4d_valid_expr<detail::build_type<A>, detail::build_type<B>, true> convolution_forward(
    A&& a, B&& b, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return dyn_conv_4d_valid_expr<detail::build_type<A>, detail::build_type<B>, true>{a, b, s1, s2, p1, p2};
}

/*!
 * \brief Backward convolution for a batch of images with a set of kernels.
 *
 * This will compute the 2D backward convolutions of each image with each given
 * kernels. The results accross channels will be accumulated together.
 *
 * The 4D matrix a is assumed to be of [N, K, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hj, Wj] dimensions.
 * The 4D matrix c is assumed to be of [N, C, (Hi - Hj + 2 * P1) / S1 + 1, (Wi - Hj + 2 * P2) / S2 + 1] dimensions.
 *
 * \param a An expression containing the batch of images
 * \param b An expression containing the set of kernels
 *
 * \tparam S1 The stride in the first dimension
 * \tparam S2 The stride in the second dimension
 * \tparam P1 The padding of the first dimension
 * \tparam P2 The padding of the second dimension
 *
 * \return an expression representing the result of the forward convolution
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
conv_4d_backward_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true> convolution_backward(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return conv_4d_backward_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true>{a, b};
}

/*!
 * \brief Backward convolution for a batch of images with a set of kernels.
 *
 * This will compute the 2D convolutions of each image with each given
 * kernels. The results accross channels will be accumulated together.
 *
 * The 4D matrix a is assumed to be of [N, K, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hj, Wj] dimensions.
 * The 4D matrix c is assumed to be of [N, C, (Hi - Hj + 2 * P1) / S1 + 1, (Wi - Hj + 2 * P2) / S2 + 1] dimensions.
 *
 * \param a An expression containing the batch of images
 * \param b An expression containing the set of kernels
 *
 * \param s1 The stride in the first dimension
 * \param s2 The stride in the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 *
 * \return an expression representing the result of the forward convolution
 */
template <typename A, typename B>
dyn_conv_4d_backward_expr<detail::build_type<A>, detail::build_type<B>, true> convolution_backward(
    A&& a, B&& b, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return dyn_conv_4d_backward_expr<detail::build_type<A>, detail::build_type<B>, true>{a, b, s1, s2, p1, p2};
}

/*!
 * \brief Backward convolution for a batch of images with a set of kernels.
 *
 * This will compute the 2D backward convolutions of each image with each given
 * kernels. The results accross channels will be accumulated together.
 *
 * The 4D matrix a is assumed to be of [N, K, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hj, Wj] dimensions.
 * The 4D matrix c is assumed to be of [N, C, (Hi - Hj + 2 * P1) / S1 + 1, (Wi - Hj + 2 * P2) / S2 + 1] dimensions.
 *
 * \param a An expression containing the batch of images
 * \param b An expression containing the set of kernels
 *
 * \tparam S1 The stride in the first dimension
 * \tparam S2 The stride in the second dimension
 * \tparam P1 The padding of the first dimension
 * \tparam P2 The padding of the second dimension
 *
 * \return an expression representing the result of the forward convolution
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
conv_4d_backward_filter_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true> convolution_backward_filter(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return conv_4d_backward_filter_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true>{a, b};
}

/*!
 * \brief Backward convolution for a batch of images with a set of kernels.
 *
 * This will compute the 2D convolutions of each image with each given
 * kernels. The results accross channels will be accumulated together.
 *
 * The 4D matrix a is assumed to be of [N, K, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hj, Wj] dimensions.
 * The 4D matrix c is assumed to be of [N, C, (Hi - Hj + 2 * P1) / S1 + 1, (Wi - Hj + 2 * P2) / S2 + 1] dimensions.
 *
 * \param a An expression containing the batch of images
 * \param b An expression containing the set of kernels
 *
 * \param s1 The stride in the first dimension
 * \param s2 The stride in the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 *
 * \return an expression representing the result of the forward convolution
 */
template <typename A, typename B>
dyn_conv_4d_backward_filter_expr<detail::build_type<A>, detail::build_type<B>, true> convolution_backward_filter(
    A&& a, B&& b, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return dyn_conv_4d_backward_filter_expr<detail::build_type<A>, detail::build_type<B>, true>{a, b, s1, s2, p1, p2};
}

// Pooling Wrappers

/*!
 * \brief Forward 2D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the 2D Forward Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1 = C1, size_t S2 = C2, size_t P1 = 0, size_t P2 = 0, typename E>
pool_2d_expr<detail::build_type<E>, C1, C2, C1, C2, 0, 0, impl::max_pool_2d> max_pool_forward(E&& value) {
    return pool_2d_expr<detail::build_type<E>, C1, C2, S1, S2, P1, P2, impl::max_pool_2d>{value};
}

/*!
 * \brief Forward 2D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the 2D Max Pooling of the input expression.
 */
template <typename E>
dyn_pool_2d_expr<detail::build_type<E>, impl::max_pool_2d> max_pool_forward(E&& value, size_t c1, size_t c2) {
    return dyn_pool_2d_expr<detail::build_type<E>, impl::max_pool_2d>{value, c1, c2, c1, c2, 0, 0};
}

/*!
 * \brief Forward 2D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the 2D Max Pooling of the input expression.
 */
template <typename E>
dyn_pool_2d_expr<detail::build_type<E>, impl::max_pool_2d> max_pool_forward(E&& value, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    return dyn_pool_2d_expr<detail::build_type<E>, impl::max_pool_2d>{value, c1, c2, s1, s2, p1, p2};
}

/*!
 * \brief Forward 3D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \tparam C3 The third pooling ratio
 * \return A expression representing the 2D Forward Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, typename E>
pool_3d_expr<detail::build_type<E>, C1, C2, C3, C1, C2, C3, 0, 0, 0, impl::max_pool_3d> max_pool_3d_forward(E&& value) {
    return pool_3d_expr<detail::build_type<E>, C1, C2, C3, C1, C2, C3, 0, 0, 0, impl::max_pool_3d>{value};
}

/*!
 * \brief Forward 3D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The second pooling ratio
 * \return A expression representing the 2D Max Pooling of the input expression.
 */
template <typename E>
dyn_pool_3d_expr<detail::build_type<E>, impl::max_pool_3d> max_pool_3d_forward(E&& value, size_t c1, size_t c2, size_t c3) {
    return dyn_pool_3d_expr<detail::build_type<E>, impl::max_pool_3d>{value, c1, c2, c3, c1, c2, c3, 0, 0, 0};
}

/*!
 * \brief Forward Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the 2D Forward Average Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1 = C1, size_t S2 = C2, size_t P1 = 0, size_t P2 = 0, typename E>
pool_2d_expr<detail::build_type<E>, C1, C2, S1, S2, P1, P2, impl::avg_pool_2d> avg_pool_forward(E&& value) {
    return pool_2d_expr<detail::build_type<E>, C1, C2, S1, S2, P1, P2, impl::avg_pool_2d>{value};
}

/*!
 * \brief 2D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the 2D Average Pooling of the input expression.
 */
template <typename E>
dyn_pool_2d_expr<detail::build_type<E>, impl::avg_pool_2d> avg_pool_forward(E&& value, size_t c1, size_t c2) {
    return dyn_pool_2d_expr<detail::build_type<E>, impl::avg_pool_2d>{value, c1, c2, c1, c2, 0, 0};
}

/*!
 * \brief 2D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the 2D Average Pooling of the input expression.
 */
template <typename E>
dyn_pool_2d_expr<detail::build_type<E>, impl::avg_pool_2d> avg_pool_forward(E&& value, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    return dyn_pool_2d_expr<detail::build_type<E>, impl::avg_pool_2d>{value, c1, c2, s1, s2, p1, p2};
}

/*!
 * \brief Forward 3D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \tparam C3 The third pooling ratio
 * \return A expression representing the 2D Forward Average Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, typename E>
pool_3d_expr<detail::build_type<E>, C1, C2, C3, C1, C2, C3, 0, 0, 0, impl::avg_pool_3d> avg_pool_3d_forward(E&& value) {
    return pool_3d_expr<detail::build_type<E>, C1, C2, C3, C1, C2, C3, 0, 0, 0, impl::avg_pool_3d>{value};
}

/*!
 * \brief Forward 3D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the 2D Average Pooling of the input expression.
 */
template <typename E>
dyn_pool_3d_expr<detail::build_type<E>, impl::avg_pool_3d> avg_pool_3d_forward(E&& value, size_t c1, size_t c2, size_t c3) {
    return dyn_pool_3d_expr<detail::build_type<E>, impl::avg_pool_3d>{value, c1, c2, c3, c1, c2, c3, 0, 0, 0};
}

/*!
 * \brief Derivative of the 2D Max Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename A, typename B, typename C>
pool_upsample_2d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, C1, C2, S1, S2, P1, P2, true> max_pool_backward(A&& input, B&& output, C&& errors) {
    return {input, output, errors};
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
dyn_pool_upsample_2d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, true> max_pool_backward(
    A&& input, B&& output, C&& errors, size_t c1, size_t c2, size_t s1, size_t s2) {
    return {input, output, errors, c1, c2, s1, s2};
}

/*!
 * \brief Derivative of the 3D Max Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C>
pool_upsample_3d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, C1, C2, C3, true> max_pool_3d_backward(A&& input,
                                                                                                                                  B&& output,
                                                                                                                                  C&& errors) {
    return {input, output, errors};
}

/*!
 * \brief Derivative of the 3D Max Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <typename A, typename B, typename C>
dyn_pool_upsample_3d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, true> max_pool_3d_backward(
    A&& input, B&& output, C&& errors, size_t c1, size_t c2, size_t c3) {
    return {input, output, errors, c1, c2, c3};
}

/*!
 * \brief Derivative of the 2D Avg Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename A, typename B, typename C>
pool_upsample_2d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, C1, C2, S1, S2, P1, P2, false> avg_pool_backward(A&& input, B&& output, C&& errors) {
    return {input, output, errors};
}

/*!
 * \brief Derivative of the 2D Average Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Average Pooling of the input expression.
 */
template <typename A, typename B, typename C>
dyn_pool_upsample_2d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, false> avg_pool_backward(
    A&& input, B&& output, C&& errors, size_t c1, size_t c2, size_t s1, size_t s2) {
    return {input, output, errors, c1, c2, s1, s2};
}

/*!
 * \brief Derivative of the 2D Avg Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t C3, typename A, typename B, typename C>
pool_upsample_3d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, C1, C2, C3, false> avg_pool_3d_backward(A&& input,
                                                                                                                                   B&& output,
                                                                                                                                   C&& errors) {
    return {input, output, errors};
}

/*!
 * \brief Derivative of the 2D Average Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Average Pooling of the input expression.
 */
template <typename A, typename B, typename C>
dyn_pool_upsample_3d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, false> avg_pool_3d_backward(
    A&& input, B&& output, C&& errors, size_t c1, size_t c2, size_t c3) {
    return {input, output, errors, c1, c2, c3};
}

// Derivatives with respect to output

/*!
 * \brief Return the derivative of the identiy function for the given output value.
 * \param value The ETL expression
 * \return 1.0
 */
template <typename E>
auto identity_derivative_out([[maybe_unused]] E&& value) {
    return 1.0;
}

/*!
 * \brief Return the derivative of the logistic sigmoid of the given ETL
 * expression, with respect to the output value.
 * \param value The ETL expression
 * \return An ETL expression representing the derivative of the logistic sigmoid of the input.
 */
template <typename E>
auto sigmoid_derivative_out(E&& value) -> decltype(value >> (1.0 - value)) {
    static_assert(is_etl_expr<E>, "etl::sigmoid_derivative can only be used on ETL expressions");
    return value >> (1.0 - value);
}

/*!
 * \brief Return the derivative of the softmax function of the given ETL
 * expression, with respect to output values.
 * \param e The ETL expression
 * \return An ETL expression representing the derivative of the softmax function of the input.
 */
template <typename E>
auto softmax_derivative_out([[maybe_unused]] E&& e) {
    return 1.0;
}

/*!
 * \brief Return the derivative of the tanh function of the given ETL expression,
 * with respect to the output values.
 * \param value The ETL expression
 * \return An ETL expression representing the derivative of the tanh function of the input.
 */
template <typename E>
auto tanh_derivative_out(E&& value) -> decltype(1.0 - (value >> value)) {
    static_assert(is_etl_expr<E>, "etl::tanh_derivative can only be used on ETL expressions");
    return 1.0 - (value >> value);
}

/*!
 * \brief Return the derivative of the relu function of the given ETL expression,
 * with respect for the output values.
 * \param value The ETL expression
 * \return An ETL expression representing the derivative of the relu function of the input.
 */
template <typename E>
auto relu_derivative_out(const E& value) -> detail::unary_helper<E, relu_derivative_op> {
    static_assert(is_etl_expr<E>, "etl::relu_derivative can only be used on ETL expressions");
    return detail::unary_helper<E, relu_derivative_op>{value};
}

// Fully-Fledged backward activation

/*!
 * \brief Return the backward activation of the identity function
 * \param output The output of the forward activation function
 * \param errors The errors at output of this activation function
 * \return the backward activation of the activation function
 */
template <typename O, typename E>
decltype(auto) identity_backward([[maybe_unused]] O&& output, E&& errors) {
    return std::forward<E>(errors);
}

/*!
 * \brief Return the backward activation of the sigmoid function
 * \param output The output of the forward activation function
 * \param errors The errors at output of this activation function
 * \return the backward activation of the activation function
 */
template <typename O, typename E>
auto sigmoid_backward(O&& output, E&& errors) -> detail::left_binary_helper<O, E, sigmoid_derivative_binary_op> {
    return {output, errors};
}

/*!
 * \brief Return the backward activation of the RELU function
 * \param output The output of the forward activation function
 * \param errors The errors at output of this activation function
 * \return the backward activation of the activation function
 */
template <typename O, typename E>
auto relu_backward(O&& output, E&& errors) -> detail::left_binary_helper<O, E, relu_derivative_binary_op> {
    return {output, errors};
}

/*!
 * \brief Return the backward activation of the softmax function
 * \param output The output of the forward activation function
 * \param errors The errors at output of this activation function
 * \return the backward activation of the activation function
 */
template <typename O, typename E>
decltype(auto) softmax_backward([[maybe_unused]] O&& output, E&& errors) {
    return std::forward<E>(errors);
}

/*!
 * \brief Return the backward activation of the tanh function
 * \param output The output of the forward activation function
 * \param errors The errors at output of this activation function
 * \return the backward activation of the activation function
 */
template <typename O, typename E>
auto tanh_backward(O&& output, E&& errors) {
    static_assert(is_etl_expr<E>, "etl::tanh_derivative can only be used on ETL expressions");
    return (1.0 - (output >> output)) >> errors;
}

/*!
 * \brief Returns the Categorical Cross Entropy Loss
 * \param output The outputs
 * \param labels The labels
 * \return The CCE Loss of the output and labels
 */
template <typename O, typename L>
value_t<O> cce_loss(O&& output, L&& labels, value_t<O> scale) {
    static_assert(all_etl_expr<O, L>, "etl::cce_loss can only be used on ETL expressions");

    return detail::cce_loss_impl::apply(output, labels, scale);
}

/*!
 * \brief Returns the Categorical Cross Entropy Error
 * \param output The outputs
 * \param labels The labels
 * \return The CCE Error of the output and labels
 */
template <typename O, typename L>
value_t<O> cce_error(O&& output, L&& labels, value_t<O> scale) {
    static_assert(all_etl_expr<O, L>, "etl::cce_error can only be used on ETL expressions");

    return detail::cce_error_impl::apply(output, labels, scale);
}

/*!
 * \brief Returns the Binary Cross Entropy Loss
 * \param output The outputs
 * \param labels The labels
 * \return The BCE Loss of the output and labels
 */
template <typename O, typename L>
value_t<O> bce_loss(O&& output, L&& labels, value_t<O> scale) {
    static_assert(all_etl_expr<O, L>, "etl::bce_loss can only be used on ETL expressions");

    return detail::bce_loss_impl::apply(output, labels, scale);
}

/*!
 * \brief Returns the Binary Cross Entropy Error
 * \param output The outputs
 * \param labels The labels
 * \return The BCE Error of the output and labels
 */
template <typename O, typename L>
value_t<O> bce_error(O&& output, L&& labels, value_t<O> scale) {
    static_assert(all_etl_expr<O, L>, "etl::bce_error can only be used on ETL expressions");

    return detail::bce_error_impl::apply(output, labels, scale);
}

/*!
 * \brief Returns the Binary Cross Entropy Loss and Error
 * \param output The outputs
 * \param labels The labels
 * \return The BCE Loss and Error of the output and labels
 */
template <typename O, typename L>
std::pair<value_t<O>, value_t<O>> bce(O&& output, L&& labels, value_t<O> alpha, value_t<O> beta) {
    static_assert(all_etl_expr<O, L>, "etl::bce can only be used on ETL expressions");

    return detail::bce_impl::apply(output, labels, alpha, beta);
}

/*!
 * \brief Returns the Categorical Cross Entropy Loss and Error
 * \param output The outputs
 * \param labels The labels
 * \return The BCE Loss and Error of the output and labels
 */
template <typename O, typename L>
std::pair<value_t<O>, value_t<O>> cce(O&& output, L&& labels, value_t<O> alpha, value_t<O> beta) {
    static_assert(all_etl_expr<O, L>, "etl::cce can only be used on ETL expressions");

    return detail::cce_impl::apply(output, labels, alpha, beta);
}

/*!
 * \brief Returns the Binary Cross Entropy Loss
 * \param output The outputs
 * \param labels The labels
 * \return The MSE Loss of the output and labels
 */
    template <typename O, typename L>
    value_t<O> mse_loss(O&& output, L&& labels, value_t<O> scale) {
        static_assert(all_etl_expr<O, L>, "etl::mse_loss can only be used on ETL expressions");

        return detail::mse_loss_impl::apply(output, labels, scale);
    }

/*!
 * \brief Returns the Binary Cross Entropy Error
 * \param output The outputs
 * \param labels The labels
 * \return The MSE Error of the output and labels
 */
    template <typename O, typename L>
    value_t<O> mse_error(O&& output, L&& labels, value_t<O> scale) {
        static_assert(all_etl_expr<O, L>, "etl::mse_error can only be used on ETL expressions");

        return detail::mse_error_impl::apply(output, labels, scale);
    }

/*!
 * \brief Returns the Binary Cross Entropy Loss and Error
 * \param output The outputs
 * \param labels The labels
 * \return The MSE Loss and Error of the output and labels
 */
    template <typename O, typename L>
    std::pair<value_t<O>, value_t<O>> mse(O&& output, L&& labels, value_t<O> alpha, value_t<O> beta) {
        static_assert(all_etl_expr<O, L>, "etl::mse can only be used on ETL expressions");

        return detail::mse_impl::apply(output, labels, alpha, beta);
    }

} //end of namespace etl::ml
