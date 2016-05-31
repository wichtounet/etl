//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
#include "etl/impl/cudnn/cudnn.hpp"

#endif

namespace etl {

namespace impl {

namespace cudnn {

#ifdef ETL_CUDNN_MODE

#define cudnn_check(call)                                                                                 \
    {                                                                                                     \
        cudnnStatus_t status = call;                                                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                                                             \
            std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                             \
        }                                                                                                 \
    }

template <typename T>
void conv2_valid(const opaque_memory<T,2>& input, const opaque_memory<T,2>& kernel, const opaque_memory<T,2>& conv) {
    using type = std::remove_const_t<T>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, input.template dim<0>(), input.template dim<1>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, conv.template dim<0>(), conv.template dim<1>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, 1, 1, kernel.template dim<0>(), kernel.template dim<1>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), input_tensor, filter, convolution,
        output_tensor, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), input_tensor, filter, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(),
        alpha, input_tensor, input.gpu_memory(),
        filter, kernel.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv2_valid_flipped(const opaque_memory<T,2>& input, const opaque_memory<T,2>& kernel, const opaque_memory<T,2>& conv) {
    using type = std::remove_const_t<T>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, input.template dim<0>(), input.template dim<1>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, conv.template dim<0>(), conv.template dim<1>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, 1, 1, kernel.template dim<0>(), kernel.template dim<1>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), input_tensor, filter, convolution,
        output_tensor, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), input_tensor, filter, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(),
        alpha, input_tensor, input.gpu_memory(),
        filter, kernel.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv4_valid(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    using type = T;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type,
        input.template dim<0>(), input.template dim<1>(), input.template dim<2>(), input.template dim<3>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type,
        conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>(), conv.template dim<3>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW,
        kernel.template dim<0>(), kernel.template dim<1>(), kernel.template dim<2>(), kernel.template dim<3>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), input_tensor, filter, convolution,
        output_tensor, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), input_tensor, filter, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(),
        alpha, input_tensor, input.gpu_memory(),
        filter, kernel.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv4_valid_flipped(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    using type = T;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type,
        input.template dim<0>(), input.template dim<1>(), input.template dim<2>(), input.template dim<3>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type,
        conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>(), conv.template dim<3>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW,
        kernel.template dim<0>(), kernel.template dim<1>(), kernel.template dim<2>(), kernel.template dim<3>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), input_tensor, filter, convolution,
        output_tensor, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), input_tensor, filter, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(),
        alpha, input_tensor, input.gpu_memory(),
        filter, kernel.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv4_valid_filter(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    using type = T;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type,
        input.template dim<0>(), input.template dim<1>(), input.template dim<2>(), input.template dim<3>()));

    // Prepare the input kernel tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type,
        kernel.template dim<0>(), kernel.template dim<1>(), kernel.template dim<2>(), kernel.template dim<3>()));

    // Prepare the output filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW,
        conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>(), conv.template dim<3>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    // Find the algorithm to use
    cudnnConvolutionBwdFilterAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(handle.get(), input_tensor, output_tensor, convolution,
        filter, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle.get(), input_tensor, output_tensor, convolution, filter, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardFilter(handle.get(),
        alpha, input_tensor, input.gpu_memory(),
        output_tensor, kernel.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, filter, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv4_valid_filter_flipped(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    using type = T;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type,
        input.template dim<0>(), input.template dim<1>(), input.template dim<2>(), input.template dim<3>()));

    // Prepare the input kernel tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type,
        kernel.template dim<0>(), kernel.template dim<1>(), kernel.template dim<2>(), kernel.template dim<3>()));

    // Prepare the output filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW,
        conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>(), conv.template dim<3>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    // Find the algorithm to use
    cudnnConvolutionBwdFilterAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(handle.get(), input_tensor, output_tensor, convolution,
        filter, CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle.get(), input_tensor, output_tensor, convolution, filter, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardFilter(handle.get(),
        alpha, input_tensor, input.gpu_memory(),
        output_tensor, kernel.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, filter, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}


template <typename T>
void conv2_full(const opaque_memory<T,2>& input, const opaque_memory<T,2>& kernel, const opaque_memory<T,2>& conv) {
    using type = std::remove_const_t<T>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, input.template dim<0>(), input.template dim<1>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, conv.template dim<0>(), conv.template dim<1>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, 1, 1, kernel.template dim<0>(), kernel.template dim<1>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    // Find the algorithm to use
    cudnnConvolutionBwdDataAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(), filter, input_tensor, convolution,
        output_tensor, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(), filter, input_tensor, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardData(handle.get(),
        alpha, filter, kernel.gpu_memory(),
        input_tensor, input.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv2_full_flipped(const opaque_memory<T,2>& input, const opaque_memory<T,2>& kernel, const opaque_memory<T,2>& conv) {
    using type = std::remove_const_t<T>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, input.template dim<0>(), input.template dim<1>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, conv.template dim<0>(), conv.template dim<1>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, 1, 1, kernel.template dim<0>(), kernel.template dim<1>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    // Find the algorithm to use
    cudnnConvolutionBwdDataAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(), filter, input_tensor, convolution,
        output_tensor, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(), filter, input_tensor, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardData(handle.get(),
        alpha, filter, kernel.gpu_memory(),
        input_tensor, input.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv4_full(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    using type = T;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type,
        input.template dim<0>(), input.template dim<1>(), input.template dim<2>(), input.template dim<3>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type,
        conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>(), conv.template dim<3>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW,
        kernel.template dim<0>(), kernel.template dim<1>(), kernel.template dim<2>(), kernel.template dim<3>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    // Find the algorithm to use
    cudnnConvolutionBwdDataAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(), filter, input_tensor, convolution,
        output_tensor, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(), filter, input_tensor, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardData(handle.get(),
        alpha, filter, kernel.gpu_memory(),
        input_tensor, input.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv4_full_flipped(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    using type = T;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type,
        input.template dim<0>(), input.template dim<1>(), input.template dim<2>(), input.template dim<3>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type,
        conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>(), conv.template dim<3>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW,
        kernel.template dim<0>(), kernel.template dim<1>(), kernel.template dim<2>(), kernel.template dim<3>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    // Find the algorithm to use
    cudnnConvolutionBwdDataAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(), filter, input_tensor, convolution,
        output_tensor, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(), filter, input_tensor, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardData(handle.get(),
        alpha, filter, kernel.gpu_memory(),
        input_tensor, input.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv2_valid_multi(const opaque_memory<T,2>& input, const opaque_memory<T,3>& kernel, const opaque_memory<T,3>& conv) {
    using type = std::remove_const_t<T>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, input.template dim<0>(), input.template dim<1>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, kernel.template dim<0>(), 1, kernel.template dim<1>(), kernel.template dim<2>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION));

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), input_tensor, filter, convolution,
        output_tensor, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), input_tensor, filter, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(),
        alpha, input_tensor, input.gpu_memory(),
        filter, kernel.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename T>
void conv2_valid_multi_flipped(const opaque_memory<T,2>& input, const opaque_memory<T,3>& kernel, const opaque_memory<T,3>& conv) {
    using type = std::remove_const_t<T>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, input.template dim<0>(), input.template dim<1>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, kernel.template dim<0>(), 1, kernel.template dim<1>(), kernel.template dim<2>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    // Find the algorithm to use
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionForwardAlgorithm(handle.get(), input_tensor, filter, convolution,
        output_tensor, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(handle.get(), input_tensor, filter, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    input.gpu_allocate_copy_if_necessary();
    kernel.gpu_allocate_copy_if_necessary();
    conv.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionForward(handle.get(),
        alpha, input_tensor, input.gpu_memory(),
        filter, kernel.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

//TODO Make real cudnn calls for the following two functions
//For some reason, it is not possible for backward convolution to expand the
//number of feature maps, therefore, it is necessary to make many calls to sub
//routines, which is highly inefficient and will result in many GPU allocations

template <typename I, typename K, typename C>
void conv2_full_multi(const I& input, const K& kernel, C&& conv) {
    auto input_gpu = input.direct();

    for(std::size_t i = 0; i < kernel.template dim<0>(); ++i){
        decltype(auto) result = conv(i);
        auto result_gpu = result.direct();

        conv2_full(input_gpu, kernel(i).direct(), result_gpu);
        result_gpu.gpu_copy_from();
        result_gpu.gpu_evict();
    }
}

template <typename I, typename K, typename C>
void conv2_full_multi_flipped(const I& input, const K& kernel, C&& conv) {
    auto input_gpu = input.direct();

    for(std::size_t i = 0; i < kernel.template dim<0>(); ++i){
        decltype(auto) result = conv(i);
        auto result_gpu = result.direct();

        conv2_full_flipped(input_gpu, kernel(i).direct(), result_gpu);
        result_gpu.gpu_copy_from();
        result_gpu.gpu_evict();
    }
}

//COVERAGE_EXCLUDE_END

template <typename I, typename K, typename C>
void conv2_full_multi_real(const I& input, const K& kernel, C&& conv) {
    using type = value_t<I>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, input.template dim<0>(), input.template dim<1>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, kernel.template dim<0>(), 1, kernel.template dim<1>(), kernel.template dim<2>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    // Find the algorithm to use
    cudnnConvolutionBwdDataAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(), filter, input_tensor, convolution,
        output_tensor, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(), filter, input_tensor, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    auto input_gpu = input.direct();
    auto kernel_gpu = kernel.direct();
    auto conv_gpu = conv.direct();

    input_gpu.gpu_allocate_copy_if_necessary();
    kernel_gpu.gpu_allocate_copy_if_necessary();
    conv_gpu.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardData(handle.get(),
        alpha, filter, kernel_gpu.gpu_memory(),
        input_tensor, input_gpu.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv_gpu.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

template <typename I, typename K, typename C>
void conv2_full_multi_flipped_real(const I& input, const K& kernel, C&& conv) {
    using type = value_t<I>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, input.template dim<0>(), input.template dim<1>()));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, conv.template dim<0>(), conv.template dim<1>(), conv.template dim<2>()));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, kernel.template dim<0>(), 1, kernel.template dim<1>(), kernel.template dim<2>()));

    // Prepare the convolution
    cudnnConvolutionDescriptor_t convolution;
    cudnn_check(cudnnCreateConvolutionDescriptor(&convolution));
    cudnn_check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));

    // Find the algorithm to use
    cudnnConvolutionBwdDataAlgo_t conv_algo;
    cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(), filter, input_tensor, convolution,
        output_tensor, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, cudnn_max_workspace, &conv_algo));

    // Prepare the workspace
    std::size_t workspace_size = 0;
    cudnn_check(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(), filter, input_tensor, convolution, output_tensor, conv_algo, &workspace_size));

    impl::cuda::cuda_memory<type> workspace;

    if(workspace_size){
        workspace = impl::cuda::cuda_allocate_only<type>(workspace_size);
    }

    // Allocate GPU memory, if necessary

    auto input_gpu = input.direct();
    auto kernel_gpu = kernel.direct();
    auto conv_gpu = conv.direct();

    input_gpu.gpu_allocate_copy_if_necessary();
    kernel_gpu.gpu_allocate_copy_if_necessary();
    conv_gpu.gpu_allocate_if_necessary();

    // Perform the convolution

    cudnn_check(cudnnConvolutionBackwardData(handle.get(),
        alpha, filter, kernel_gpu.gpu_memory(),
        input_tensor, input_gpu.gpu_memory(),
        convolution, conv_algo, workspace.get(), workspace_size,
        beta, output_tensor, conv_gpu.gpu_memory()));

    // Release the resources
    cudnn_check(cudnnDestroyConvolutionDescriptor(convolution));
    cudnn_check(cudnnDestroyFilterDescriptor(filter));
    cudnn_check(cudnnDestroyTensorDescriptor(output_tensor));
    cudnn_check(cudnnDestroyTensorDescriptor(input_tensor));
}

//COVERAGE_EXCLUDE_END

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief cudnn implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv2_valid(const opaque_memory<T,2>& input, const opaque_memory<T,2>& kernel, const opaque_memory<T,2>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid");
}

/*!
 * \brief cudnn implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv2_valid_flipped(const opaque_memory<T,2>& input, const opaque_memory<T,2>& kernel, const opaque_memory<T,2>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid_flipped");
}


/*!
 * \brief cudnn implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_valid(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv4_valid");
}

template <typename T>
void conv4_valid_flipped(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv4_valid_flipped");
}

template <typename T>
void conv4_valid_filter(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv4_valid_filter");
}

template <typename T>
void conv4_valid_filter_flipped(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv4_valid_filter_flipped");
}

/*!
 * \brief cudnn implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv2_full(const opaque_memory<T,2>& input, const opaque_memory<T,2>& kernel, const opaque_memory<T,2>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_full");
}

/*!
 * \brief cudnn implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv2_full_flipped(const opaque_memory<T,2>& input, const opaque_memory<T,2>& kernel, const opaque_memory<T,2>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_full_flipped");
}

/*!
 * \brief cudnn implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_full(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv4_full");
}

/*!
 * \brief cudnn implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_full_flipped(const opaque_memory<T,4>& input, const opaque_memory<T,4>& kernel, const opaque_memory<T,4>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv4_ful_flippedl");
}

template <typename T>
void conv2_valid_multi(const opaque_memory<T,2>& input, const opaque_memory<T,3>& kernel, const opaque_memory<T,3>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid_multi");
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv2_valid_multi_flipped(const opaque_memory<T,2>& input, const opaque_memory<T,3>& kernel, const opaque_memory<T,3>& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid_multi_flipped");
}

/*!
 * \brief Standard implementation of a 2D 'full' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi(const I& input, const K& kernel, C&& conv){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_full_multi");
}

/*!
 * \brief Standard implementation of a 2D 'full' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi_flipped(const I& input, const K& kernel, C&& conv){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_full_multi_flipped");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cudnn

} //end of namespace impl

} //end of namespace etl
