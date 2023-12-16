//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Convolution implementations with NVidia cuDNN library
 */

#pragma once

#ifdef ETL_CUDNN_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"
#include "etl/impl/cudnn/cudnn.hpp"

#endif

namespace etl::impl::cudnn {

#ifdef ETL_CUDNN_MODE

/*!
 * \brief Compute an activation of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 * \param mode The activation function to use
 */
template <typename I, typename C>
void activation(I&& x, C&& y, cudnnActivationMode_t mode) {
    using type = std::remove_const_t<value_t<I>>;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto x_tensor = create_tensor_flat_wrapper(x);
    auto y_tensor = create_tensor_flat_wrapper(y);

    cudnnActivationDescriptor_t func_tensor;
    cudnn_check(cudnnCreateActivationDescriptor(&func_tensor));
    cudnn_check(cudnnSetActivationDescriptor(func_tensor, mode, CUDNN_PROPAGATE_NAN, 0.0));

    // Allocate GPU memory, if necessary

    x.ensure_gpu_up_to_date();
    y.ensure_gpu_allocated();

    // y = activation(x)

    cudnn_check(cudnnActivationForward(handle.get(), func_tensor, alpha, *x_tensor, x.gpu_memory(), beta, *y_tensor, y.gpu_memory()));

    y.validate_gpu();
    y.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyActivationDescriptor(func_tensor));
}

/*!
 * \brief Compute an activation of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 * \param mode The activation function to use
 */
template <typename Y, typename DY, typename DX>
void backward_activation(Y&& y, DY&& dy, DX&& dx, cudnnActivationMode_t mode) {
    using type = std::remove_const_t<value_t<Y>>;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto y_tensor  = create_tensor_flat_wrapper(y);
    auto dy_tensor = create_tensor_flat_wrapper(dy);
    auto dx_tensor = create_tensor_flat_wrapper(dx);

    cudnnActivationDescriptor_t func_tensor;
    cudnn_check(cudnnCreateActivationDescriptor(&func_tensor));
    cudnn_check(cudnnSetActivationDescriptor(func_tensor, mode, CUDNN_PROPAGATE_NAN, 0.0));

    // Allocate GPU memory, if necessary

    y.ensure_gpu_up_to_date();
    dy.ensure_gpu_up_to_date();
    dx.ensure_gpu_allocated();

    // y = activation(x)

    cudnn_check(cudnnActivationBackward(handle.get(), func_tensor, alpha, *y_tensor, y.gpu_memory(), *dy_tensor, dy.gpu_memory(), *y_tensor, y.gpu_memory(),
                                        beta, *dx_tensor, dx.gpu_memory()));

    dx.validate_gpu();
    dx.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyActivationDescriptor(func_tensor));
}

/*!
 * \brief Compute the sigmoid of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void sigmoid(I&& x, C&& y) {
    activation(x, y, CUDNN_ACTIVATION_SIGMOID);
}

/*!
 * \brief Compute the RELU of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void relu(I&& x, C&& y) {
    activation(x, y, CUDNN_ACTIVATION_RELU);
}

/*!
 * \brief Compute the backward sigmoid of o/e and store the result in y
 * \param o The a expression
 * \param e The b expression
 * \param y The c expression
 */
template <typename O, typename E, typename C>
void sigmoid_backward(O&& o, E&& e, C&& y) {
    backward_activation(o, e, y, CUDNN_ACTIVATION_SIGMOID);
}

/*!
 * \brief Compute the backward sigmoid of o/e and store the result in y
 * \param o The a expression
 * \param e The b expression
 * \param y The c expression
 */
template <typename O, typename E, typename C>
void relu_backward(O&& o, E&& e, C&& y) {
    backward_activation(o, e, y, CUDNN_ACTIVATION_RELU);
}

/*!
 * \brief Compute a softmax activation of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 * \param mode The softmax algorithm function to use
 */
template <typename I, typename C>
void softmax_activation(I&& x, C&& y, cudnnSoftmaxAlgorithm_t mode) {
    using type = std::remove_const_t<value_t<I>>;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto x_tensor = create_tensor_front_wrapper(x);
    auto y_tensor = create_tensor_front_wrapper(y);

    // Allocate GPU memory, if necessary

    x.ensure_gpu_up_to_date();
    y.ensure_gpu_allocated();

    // y = activation(x)

    cudnn_check(cudnnSoftmaxForward(handle.get(), mode, CUDNN_SOFTMAX_MODE_INSTANCE, alpha, *x_tensor, x.gpu_memory(), beta, *y_tensor, y.gpu_memory()));

    y.validate_gpu();
    y.invalidate_cpu();
}

/*!
 * \brief Compute the softmax of x (batch) and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void softmax(I&& x, C&& y) {
    softmax_activation(x, y, CUDNN_SOFTMAX_FAST);
}

/*!
 * \brief Compute the stable softmax of x (batch) and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void stable_softmax(I&& x, C&& y) {
    softmax_activation(x, y, CUDNN_SOFTMAX_ACCURATE);
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Compute the sigmoid of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void sigmoid([[maybe_unused]] I&& x, [[maybe_unused]] C&& y) {
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief Compute the RELU of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void relu([[maybe_unused]] I&& x, [[maybe_unused]] C&& y) {
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief Compute the backward sigmoid of o/e and store the result in y
 * \param o The a expression
 * \param e The b expression
 * \param y The c expression
 */
template <typename O, typename E, typename C>
void sigmoid_backward([[maybe_unused]] O&& o, [[maybe_unused]] E&& e, [[maybe_unused]] C&& y) {
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief Compute the backward sigmoid of o/e and store the result in y
 * \param o The a expression
 * \param e The b expression
 * \param y The c expression
 */
template <typename O, typename E, typename C>
void relu_backward([[maybe_unused]] O&& o, [[maybe_unused]] E&& e, [[maybe_unused]] C&& y) {
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief Compute the softmax of x (batch) and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void softmax([[maybe_unused]] I&& x, [[maybe_unused]] C&& y) {
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief Compute the softmax of x (batch) and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void stable_softmax([[maybe_unused]] I&& x, [[maybe_unused]] C&& y) {
    cpp_unreachable("CUDNN not available/enabled");
}

    //COVERAGE_EXCLUDE_END

#endif

} //end of namespace etl::impl::cudnn
