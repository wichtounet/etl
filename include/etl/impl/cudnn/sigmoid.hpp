//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace etl {

namespace impl {

namespace cudnn {

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
    type beta[] = {0.0f};

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto x_tensor = create_tensor_flat(x);
    auto y_tensor = create_tensor_flat(y);

    cudnnActivationDescriptor_t func_tensor;
    cudnn_check(cudnnCreateActivationDescriptor(&func_tensor));
    cudnn_check(cudnnSetActivationDescriptor(func_tensor, mode, CUDNN_PROPAGATE_NAN, 0.0));

    // Allocate GPU memory, if necessary

    x.ensure_gpu_up_to_date();
    y.ensure_gpu_allocated();

    // y = activation(x)

    cudnn_check(cudnnActivationForward(handle.get(),
        func_tensor,
        alpha, *x_tensor, x.gpu_memory(),
        beta, *y_tensor, y.gpu_memory()));

    y.validate_gpu();
    y.invalidate_cpu();

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

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Compute the sigmoid of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void sigmoid(I&& x, C&& y) {
    cpp_unused(x);
    cpp_unused(y);
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief Compute the RELU of x and store the result in y
 * \param x The a expression
 * \param y The c expression
 */
template <typename I, typename C>
void relu(I&& x, C&& y) {
    cpp_unused(x);
    cpp_unused(y);
    cpp_unreachable("CUDNN not available/enabled");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cudnn

} //end of namespace impl

} //end of namespace etl
