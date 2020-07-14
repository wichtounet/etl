//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Convolution implementations with NVidia cuDNN library
 */

#pragma once

#define ETL_TENSOR_CORES

#ifdef ETL_CUDNN_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cudnn/cudnn.hpp"

#endif

namespace etl::impl::cudnn {

/*!
 * \brief Traits indicating if Convolution with CUDNN is
 * possible for the given configuration.
 *
 * \param I The type of the input matrix
 * \param K The type of the kernel matrix
 * \param C The type of the output matrix
 */
template <typename I, typename K, typename C>
constexpr bool conv_possible = cudnn_enabled&& all_homogeneous<I, K, C>&& all_row_major<I, K, C>&& all_dma<I, K, C>;

/*!
 * \brief Traits indicating if Convolution with CUDNN is
 * possible for the given configuration.
 *
 * \param I The type of the input matrix
 * \param K The type of the kernel matrix
 */
template <typename I, typename K>
constexpr bool conv_possible_ = cudnn_enabled&& all_homogeneous<I, K>&& all_row_major<I, K>&& all_dma<I, K>;

#ifdef ETL_CUDNN_MODE

/*!
 * \brief CUDNN implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid_set(I&& input, K&& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, cudnnConvolutionMode_t mode) {
    using type = std::remove_const_t<value_t<I>>;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    auto data_type = std::is_same_v<type, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto input_tensor  = create_tensor_wrapper(input);
    auto output_tensor = create_tensor_wrapper(conv);
    auto filter        = create_filter_wrapper(kernel);

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, p1, p2, s1, s2, 1, 1, mode, data_type));
#ifdef ETL_TENSOR_CORES
    cudnn_check(cudnnSetConvolutionMathType(convolution, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), *input_tensor, *filter, convolution, *output_tensor,
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), *input_tensor, *filter, convolution, *output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if (workspace_size) {
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.ensure_gpu_up_to_date();
    kernel.ensure_gpu_up_to_date();
    conv.ensure_gpu_allocated();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(), alpha, *input_tensor, input.gpu_memory(), *filter, kernel.gpu_memory(), convolution, conv_algo,
                                        workspace.get(), workspace_size, beta, *output_tensor, conv.gpu_memory()));

    conv.validate_gpu();
    conv.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
}

/*!
 * \brief CUDNN implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid([[maybe_unused]] I&& input,
                 [[maybe_unused]] K&& kernel,
                 [[maybe_unused]] C&& conv,
                 [[maybe_unused]] size_t s1,
                 [[maybe_unused]] size_t s2,
                 [[maybe_unused]] size_t p1,
                 [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv2_valid_set(input, kernel, conv, s1, s2, p1, p2, CUDNN_CONVOLUTION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief CUDNN implementation of a 2D 'valid' convolution
 * C = I * K, with flipped kernels.
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid_flipped([[maybe_unused]] I&& input,
                         [[maybe_unused]] K&& kernel,
                         [[maybe_unused]] C&& conv,
                         [[maybe_unused]] size_t s1,
                         [[maybe_unused]] size_t s2,
                         [[maybe_unused]] size_t p1,
                         [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv2_valid_set(input, kernel, conv, s1, s2, p1, p2, CUDNN_CROSS_CORRELATION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief cudnn implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_forward_set(I&& input, K&& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, cudnnConvolutionMode_t mode) {
    using type = value_t<I>;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    auto data_type = std::is_same_v<type, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto input_tensor  = create_tensor_wrapper(input);
    auto output_tensor = create_tensor_wrapper(conv);
    auto filter        = create_filter_wrapper(kernel);

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, p1, p2, s1, s2, 1, 1, mode, data_type));
#ifdef ETL_TENSOR_CORES
    cudnn_check(cudnnSetConvolutionMathType(convolution, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), *input_tensor, *filter, convolution, *output_tensor,
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), *input_tensor, *filter, convolution, *output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if (workspace_size) {
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.ensure_gpu_up_to_date();
    kernel.ensure_gpu_up_to_date();
    conv.ensure_gpu_allocated();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(), alpha, *input_tensor, input.gpu_memory(), *filter, kernel.gpu_memory(), convolution, conv_algo,
                                        workspace.get(), workspace_size, beta, *output_tensor, conv.gpu_memory()));

    conv.validate_gpu();
    conv.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
}

/*!
 * \brief cudnn implementation of a 4D 'valid' convolution C = I * K, with flipped weights
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_forward([[maybe_unused]] I&& input,
                   [[maybe_unused]] K&& kernel,
                   [[maybe_unused]] C&& conv,
                   [[maybe_unused]] size_t s1,
                   [[maybe_unused]] size_t s2,
                   [[maybe_unused]] size_t p1,
                   [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv4_forward_set(input, kernel, conv, s1, s2, p1, p2, CUDNN_CONVOLUTION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief cudnn implementation of a 4D 'valid' convolution C = I * K, with flipped weights
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_forward_flipped([[maybe_unused]] I&& input,
                           [[maybe_unused]] K&& kernel,
                           [[maybe_unused]] C&& conv,
                           [[maybe_unused]] size_t s1,
                           [[maybe_unused]] size_t s2,
                           [[maybe_unused]] size_t p1,
                           [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv4_forward_set(input, kernel, conv, s1, s2, p1, p2, CUDNN_CROSS_CORRELATION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief CUDNN implementation of a 4D 'valid' convolution C = I * K, where the output
 * are considered to be kernels
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_filter_set(I&& input, K&& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, cudnnConvolutionMode_t mode) {
    using type = value_t<I>;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    auto data_type = std::is_same_v<type, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto input_tensor  = create_tensor_wrapper(input);
    auto output_tensor = create_tensor_wrapper(kernel);
    auto filter        = create_filter_wrapper(conv);

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, p1, p2, s1, s2, 1, 1, mode, data_type));
#ifdef ETL_TENSOR_CORES
    cudnn_check(cudnnSetConvolutionMathType(convolution, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif

    // Find the algorithm to use
    cudnnConvolutionBwdFilterAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(handle.get(), *input_tensor, *output_tensor, convolution, *filter,
                                                           CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle.get(), *input_tensor, *output_tensor, convolution, *filter, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if (workspace_size) {
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.ensure_gpu_up_to_date();
    kernel.ensure_gpu_up_to_date();
    conv.ensure_gpu_allocated();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardFilter(handle.get(), alpha, *input_tensor, input.gpu_memory(), *output_tensor, kernel.gpu_memory(), convolution,
                                               conv_algo, workspace.get(), workspace_size, beta, *filter, conv.gpu_memory()));

    conv.validate_gpu();
    conv.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
}

/*!
 * \brief CUDNN implementation of a 4D 'valid' convolution C = I * K, where the output
 * are considered to be kernels, with flipped weights
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_filter([[maybe_unused]] I&& input,
                           [[maybe_unused]] K&& kernel,
                           [[maybe_unused]] C&& conv,
                           [[maybe_unused]] size_t s1,
                           [[maybe_unused]] size_t s2,
                           [[maybe_unused]] size_t p1,
                           [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv4_backward_filter_set(input, kernel, conv, s1, s2, p1, p2, CUDNN_CONVOLUTION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief CUDNN implementation of a 4D 'valid' convolution C = I * K, where the output
 * are considered to be kernels, with flipped weights
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_filter_flipped([[maybe_unused]] I&& input,
                                   [[maybe_unused]] K&& kernel,
                                   [[maybe_unused]] C&& conv,
                                   [[maybe_unused]] size_t s1,
                                   [[maybe_unused]] size_t s2,
                                   [[maybe_unused]] size_t p1,
                                   [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv4_backward_filter_set(input, kernel, conv, s1, s2, p1, p2, CUDNN_CROSS_CORRELATION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief cudnn implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_set(I&& input, K&& kernel, C&& conv, cudnnConvolutionMode_t mode) {
    using type = std::remove_const_t<value_t<I>>;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    auto data_type = std::is_same_v<type, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto input_tensor  = create_tensor_wrapper(input);
    auto output_tensor = create_tensor_wrapper(conv);
    auto filter        = create_filter_wrapper(kernel);

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, mode, data_type));
#ifdef ETL_TENSOR_CORES
    cudnn_check(cudnnSetConvolutionMathType(convolution, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif

    // Find the algorithm to use
    cudnnConvolutionBwdDataAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(), *filter, *input_tensor, convolution, *output_tensor,
                                                         CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(), *filter, *input_tensor, convolution, *output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if (workspace_size) {
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.ensure_gpu_up_to_date();
    kernel.ensure_gpu_up_to_date();
    conv.ensure_gpu_allocated();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardData(handle.get(), alpha, *filter, kernel.gpu_memory(), *input_tensor, input.gpu_memory(), convolution, conv_algo,
                                             workspace.get(), workspace_size, beta, *output_tensor, conv.gpu_memory()));

    conv.validate_gpu();
    conv.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
}

/*!
 * \brief cudnn implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    if constexpr (conv_possible<I, K, C>) {
        conv2_full_set(input, kernel, conv, CUDNN_CROSS_CORRELATION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief cudnn implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_flipped([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    if constexpr (conv_possible<I, K, C>) {
        conv2_full_set(input, kernel, conv, CUDNN_CONVOLUTION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief CUDNN implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi_set(I& input, K&& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, cudnnConvolutionMode_t mode) {
    using type = std::remove_const_t<value_t<I>>;

    auto data_type = std::is_same_v<type, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    decltype(auto) handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, etl::dim<0>(input), etl::dim<1>(input)));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, etl::dim<0>(conv), etl::dim<1>(conv), etl::dim<2>(conv)));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, etl::dim<0>(kernel), 1, etl::dim<1>(kernel), etl::dim<2>(kernel)));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, p1, p2, s1, s2, 1, 1, mode, data_type));
#ifdef ETL_TENSOR_CORES
    cudnn_check(cudnnSetConvolutionMathType(convolution, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), input_tensor, filter, convolution, output_tensor,
                                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), input_tensor, filter, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if (workspace_size) {
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.ensure_gpu_up_to_date();
    kernel.ensure_gpu_up_to_date();
    conv.ensure_gpu_allocated();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(), alpha, input_tensor, input.gpu_memory(), filter, kernel.gpu_memory(), convolution, conv_algo,
                                        workspace.get(), workspace_size, beta, output_tensor, conv.gpu_memory()));

    conv.validate_gpu();
    conv.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi([[maybe_unused]] I&& input,
                       [[maybe_unused]] K&& kernel,
                       [[maybe_unused]] C&& conv,
                       [[maybe_unused]] size_t s1,
                       [[maybe_unused]] size_t s2,
                       [[maybe_unused]] size_t p1,
                       [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv2_valid_multi_set(input, kernel, conv, s1, s2, p1, p2, CUDNN_CONVOLUTION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi_flipped([[maybe_unused]] I&& input,
                               [[maybe_unused]] K&& kernel,
                               [[maybe_unused]] C&& conv,
                               [[maybe_unused]] size_t s1,
                               [[maybe_unused]] size_t s2,
                               [[maybe_unused]] size_t p1,
                               [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv2_valid_multi_set(input, kernel, conv, s1, s2, p1, p2, CUDNN_CROSS_CORRELATION);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief cudnn implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data_set(I&& input, K&& kernel, C&& conv, cudnnConvolutionMode_t mode, size_t s1, size_t s2, size_t p1, size_t p2) {
    using type = value_t<I>;

    type alpha[] = {1.0f};
    type beta[]  = {0.0f};

    auto data_type = std::is_same_v<type, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    decltype(auto) handle = start_cudnn();

    // Prepare the tensors
    auto input_tensor  = create_tensor_wrapper(input);
    auto output_tensor = create_tensor_wrapper(conv);
    auto filter        = create_filter_wrapper(kernel);

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, p1, p2, s1, s2, 1, 1, mode, data_type));
#ifdef ETL_TENSOR_CORES
    cudnn_check(cudnnSetConvolutionMathType(convolution, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
#endif

    // Find the algorithm to use
    cudnnConvolutionBwdDataAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(), *filter, *input_tensor, convolution, *output_tensor,
                                                         CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(), *filter, *input_tensor, convolution, *output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if (workspace_size) {
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.ensure_gpu_up_to_date();
    kernel.ensure_gpu_up_to_date();
    conv.ensure_gpu_allocated();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardData(handle.get(), alpha, *filter, kernel.gpu_memory(), *input_tensor, input.gpu_memory(), convolution, conv_algo,
                                             workspace.get(), workspace_size, beta, *output_tensor, conv.gpu_memory()));

    conv.validate_gpu();
    conv.invalidate_cpu();

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
}

/*!
 * \brief cudnn implementation of a 4D 'valid' backward convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data([[maybe_unused]] I&& input,
                         [[maybe_unused]] K&& kernel,
                         [[maybe_unused]] C&& conv,
                         [[maybe_unused]] size_t s1,
                         [[maybe_unused]] size_t s2,
                         [[maybe_unused]] size_t p1,
                         [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv4_backward_data_set(input, kernel, conv, CUDNN_CROSS_CORRELATION, s1, s2, p1, p2);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief cudnn implementation of a 2D 'valid' backward convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data_flipped([[maybe_unused]] I&& input,
                                 [[maybe_unused]] K&& kernel,
                                 [[maybe_unused]] C&& conv,
                                 [[maybe_unused]] size_t s1,
                                 [[maybe_unused]] size_t s2,
                                 [[maybe_unused]] size_t p1,
                                 [[maybe_unused]] size_t p2) {
    if constexpr (conv_possible<I, K, C>) {
        conv4_backward_data_set(input, kernel, conv, CUDNN_CONVOLUTION, s1, s2, p1, p2);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief cudnn implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data_full([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    if constexpr (conv_possible<I, K, C>) {
        conv4_backward_data_set(input, kernel, conv, CUDNN_CROSS_CORRELATION, 1, 1, 0, 0);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

/*!
 * \brief cudnn implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data_full_flipped([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    if constexpr (conv_possible<I, K, C>) {
        conv4_backward_data_set(input, kernel, conv, CUDNN_CONVOLUTION, 1, 1, 0, 0);
    } else {
        cpp_unreachable("CUDNN not available/enabled");
    }
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief CUDNN implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid([[maybe_unused]] I&& input,
                 [[maybe_unused]] K&& kernel,
                 [[maybe_unused]] C&& conv,
                 [[maybe_unused]] size_t s1,
                 [[maybe_unused]] size_t s2,
                 [[maybe_unused]] size_t p1,
                 [[maybe_unused]] size_t p2) {
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief CUDNN implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid_flipped([[maybe_unused]] I&& input,
                         [[maybe_unused]] K&& kernel,
                         [[maybe_unused]] C&& conv,
                         [[maybe_unused]] size_t s1,
                         [[maybe_unused]] size_t s2,
                         [[maybe_unused]] size_t p1,
                         [[maybe_unused]] size_t p2) {
    cpp_unreachable("CUDNN not available/enabled");
}

/*!
 * \brief cudnn implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_forward([[maybe_unused]] I&& input,
                   [[maybe_unused]] K&& kernel,
                   [[maybe_unused]] C&& conv,
                   [[maybe_unused]] size_t s1,
                   [[maybe_unused]] size_t s2,
                   [[maybe_unused]] size_t p1,
                   [[maybe_unused]] size_t p2) {
    cpp_unreachable("Unsupported feature called: cudnn conv4_valid");
}

/*!
 * \brief cudnn implementation of a 4D 'valid' convolution C = I * K, with flipped weights
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_forward_flipped([[maybe_unused]] I&& input,
                           [[maybe_unused]] K&& kernel,
                           [[maybe_unused]] C&& conv,
                           [[maybe_unused]] size_t s1,
                           [[maybe_unused]] size_t s2,
                           [[maybe_unused]] size_t p1,
                           [[maybe_unused]] size_t p2) {
    cpp_unreachable("Unsupported feature called: cudnn conv4_valid_flipped");
}

/*!
 * \brief CUDNN implementation of a 4D 'valid' convolution C = I * K, where the output
 * are considered to be kernels
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_filter([[maybe_unused]] I&& input,
                           [[maybe_unused]] K&& kernel,
                           [[maybe_unused]] C&& conv,
                           [[maybe_unused]] size_t s1,
                           [[maybe_unused]] size_t s2,
                           [[maybe_unused]] size_t p1,
                           [[maybe_unused]] size_t p2) {
    cpp_unreachable("Unsupported feature called: cudnn conv4_valid_filter");
}

/*!
 * \brief CUDNN implementation of a 4D 'valid' convolution C = I * K, where the output
 * are considered to be kernels, with flipped weights
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_filter_flipped([[maybe_unused]] I&& input,
                                   [[maybe_unused]] K&& kernel,
                                   [[maybe_unused]] C&& conv,
                                   [[maybe_unused]] size_t s1,
                                   [[maybe_unused]] size_t s2,
                                   [[maybe_unused]] size_t p1,
                                   [[maybe_unused]] size_t p2) {
    cpp_unreachable("Unsupported feature called: cudnn conv4_backward_filter_flipped");
}

/*!
 * \brief cudnn implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    cpp_unreachable("Unsupported feature called: cudnn conv2_full");
}

/*!
 * \brief cudnn implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_flipped([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    cpp_unreachable("Unsupported feature called: cudnn conv2_full_flipped");
}

/*!
 * \brief cudnn implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data_full([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    cpp_unreachable("Unsupported feature called: cudnn conv4_full");
}

/*!
 * \brief cudnn implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data_full_flipped([[maybe_unused]] I&& input, [[maybe_unused]] K&& kernel, [[maybe_unused]] C&& conv) {
    cpp_unreachable("Unsupported feature called: cudnn conv4_ful_flippedl");
}

/*!
 * \brief CUDNN implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi([[maybe_unused]] I&& input,
                       [[maybe_unused]] K&& kernel,
                       [[maybe_unused]] C&& conv,
                       [[maybe_unused]] size_t s1,
                       [[maybe_unused]] size_t s2,
                       [[maybe_unused]] size_t p1,
                       [[maybe_unused]] size_t p2) {
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid_multi");
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi_flipped([[maybe_unused]] I&& input,
                               [[maybe_unused]] K&& kernel,
                               [[maybe_unused]] C&& conv,
                               [[maybe_unused]] size_t s1,
                               [[maybe_unused]] size_t s2,
                               [[maybe_unused]] size_t p1,
                               [[maybe_unused]] size_t p2) {
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid_multi_flipped");
}

/*!
 * \brief cudnn implementation of a 4D 'valid' backward convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data([[maybe_unused]] I&& input,
                         [[maybe_unused]] K&& kernel,
                         [[maybe_unused]] C&& conv,
                         [[maybe_unused]] size_t s1,
                         [[maybe_unused]] size_t s2,
                         [[maybe_unused]] size_t p1,
                         [[maybe_unused]] size_t p2) {
    cpp_unreachable("Unsupported feature called: cudnn conv4_backward_data");
}

/*!
 * \brief cudnn implementation of a 2D 'valid' backward convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_backward_data_flipped([[maybe_unused]] I&& input,
                                 [[maybe_unused]] K&& kernel,
                                 [[maybe_unused]] C&& conv,
                                 [[maybe_unused]] size_t s1,
                                 [[maybe_unused]] size_t s2,
                                 [[maybe_unused]] size_t p1,
                                 [[maybe_unused]] size_t p2) {
    cpp_unreachable("Unsupported feature called: cudnn conv4_backward_data_flipped");
}

    //COVERAGE_EXCLUDE_END

#endif

} //end of namespace etl::impl::cudnn
