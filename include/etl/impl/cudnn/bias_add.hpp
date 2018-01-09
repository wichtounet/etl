//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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
#include "etl/impl/cublas/axpy.hpp"
#include "etl/impl/cudnn/cudnn.hpp"

#endif

namespace etl {

namespace impl {

namespace cudnn {

#ifdef ETL_CUDNN_MODE

/*!
 * \brief Compute the bias addition of b into x and store the result in y
 * \param x The a expression
 * \param b The b expression
 * \param y The c expression
 */
template <typename I, typename K, typename C>
void bias_add_4d(I&& x, K&& b, C&& y) {
    using type = std::remove_const_t<value_t<I>>;

    auto data_type = std::is_same<std::remove_const_t<type>, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto x_tensor = create_tensor_wrapper(x);
    auto y_tensor = create_tensor_wrapper(y);

    cudnnTensorDescriptor_t b_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&b_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(b_tensor, CUDNN_TENSOR_NCHW, data_type, 1, etl::dim<0>(b), 1, 1));

    // Allocate GPU memory, if necessary

    x.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    y.ensure_gpu_allocated();

    // Copy x -> y

    cudnn_check(cudnnTransformTensor(handle.get(),
        alpha, *x_tensor, x.gpu_memory(),
        beta, *y_tensor, y.gpu_memory()));

    // Add b -> y

    cudnn_check(cudnnAddTensor(handle.get(),
        alpha, b_tensor, b.gpu_memory(),
        alpha, *y_tensor, y.gpu_memory()));

    y.validate_gpu();
    y.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyTensorDescriptor(b_tensor));
}

/*!
 * \brief Compute the bias addition of b into x and store the result in y
 * \param x The a expression
 * \param b The b expression
 * \param y The c expression
 */
template <typename I, typename K, typename C>
void bias_add_2d(I&& x, K&& b, C&& y) {
    using type = std::remove_const_t<value_t<I>>;

    auto data_type = std::is_same<std::remove_const_t<type>, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto x_tensor = create_tensor_wrapper(x);
    auto y_tensor = create_tensor_wrapper(y);

    cudnnTensorDescriptor_t b_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&b_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(b_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, 1, etl::dim<0>(b)));

    // Allocate GPU memory, if necessary

    x.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    y.ensure_gpu_allocated();

    // Copy x -> y

    cudnn_check(cudnnTransformTensor(handle.get(),
        alpha, *x_tensor, x.gpu_memory(),
        beta, *y_tensor, y.gpu_memory()));

    // Add b -> y

    // This is highly retarded stuff :(
    // Unfortunately cudnnAddTensor does not support 2D tensors :(
    // This is solved when EGBLAS is available, since this will be
    // computed with EGBLAS first

    {
        decltype(auto) handle = etl::impl::cublas::start_cublas();

        for (size_t i = 0; i < etl::dim<0>(x); ++i) {
            impl::cublas::cublas_axpy(handle.get(), etl::dim<1>(y), alpha, b.gpu_memory(), 1, y.gpu_memory() + i * etl::dim<1>(y), 1);
        }
    }

    y.validate_gpu();
    y.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyTensorDescriptor(b_tensor));
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Compute the bias addition of b into x and store the result in y
 * \param x The a expression
 * \param b The b expression
 * \param y The c expression
 */
template <typename I, typename K, typename C>
void bias_add_4d(I&& x, K&& b, C&& y) {
    cpp_unused(x);
    cpp_unused(b);
    cpp_unused(y);
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief Compute the bias addition of b into x and store the result in y
 * \param x The a expression
 * \param b The b expression
 * \param y The c expression
 */
template <typename I, typename K, typename C>
void bias_add_2d(I&& x, K&& b, C&& y) {
    cpp_unused(x);
    cpp_unused(b);
    cpp_unused(y);
    cpp_unreachable("CUDNN not available/enabled");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cudnn

} //end of namespace impl

} //end of namespace etl
