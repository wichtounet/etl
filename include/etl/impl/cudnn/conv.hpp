//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUDNN_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cudnn/cudnn.hpp"

#endif

namespace etl {

namespace impl {

namespace cudnn {

#ifdef ETL_CUDNN_MODE

#define cudnn_check(call)                                                                                  \
    {                                                                                                      \
        cudnnStatus_t status = call;                                                                       \
        if (status != CUDNN_STATUS_SUCCESS) {                                                              \
            std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << " from " << #call << std::endl; \
        }                                                                                                  \
    }

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value))>
void conv2_valid(const I& input, const K& kernel, C&& conv) {
    using type = value_t<I>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, etl::dim<0>(input), etl::dim<1>(input)));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, etl::dim<0>(conv), etl::dim<1>(conv)));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, 1, 1, etl::dim<0>(kernel), etl::dim<1>(kernel)));

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

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value))>
void conv4_valid(const I& input, const K& kernel, C&& conv) {
    using type = value_t<I>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type,
        etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input)));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type,
        etl::dim<0>(conv), etl::dim<1>(conv), etl::dim<2>(conv), etl::dim<3>(conv)));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW,
        etl::dim<0>(kernel), etl::dim<1>(kernel), etl::dim<2>(kernel), etl::dim<3>(kernel)));

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

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value))>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    using type = value_t<I>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, etl::dim<0>(input), etl::dim<1>(input)));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, etl::dim<0>(conv), etl::dim<1>(conv)));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, 1, 1, etl::dim<0>(kernel), etl::dim<1>(kernel)));

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

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value))>
void conv4_full(const I& input, const K& kernel, C&& conv) {
    using type = value_t<I>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

    // Prepare the input tensor
    cudnnTensorDescriptor_t input_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&input_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, data_type,
        etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input)));

    // Prepare the output tensor
    cudnnTensorDescriptor_t output_tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&output_tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, data_type,
        etl::dim<0>(conv), etl::dim<1>(conv), etl::dim<2>(conv), etl::dim<3>(conv)));

    // Prepare the filter
    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW,
        etl::dim<0>(kernel), etl::dim<1>(kernel), etl::dim<2>(kernel), etl::dim<3>(kernel)));

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

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value))>
void conv2_valid_multi(const I& input, const K& kernel, C&& conv) {
    using type = value_t<I>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

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

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value))>
void conv2_valid_multi_flipped(const I& input, const K& kernel, C&& conv) {
    using type = value_t<I>;

    auto data_type = std::is_same<type, float>::value ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    type alpha[] = {1.0f};
    type beta[] = {0.0f};

    cudnn_handle handle = start_cudnn();

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

template <typename I, typename K, typename C, cpp_enable_if((!all_dma<I, K, C>::value))>
void conv2_valid(const I& input, const K& kernel, C&& conv);

template <typename I, typename K, typename C, cpp_enable_if((!all_dma<I, K, C>::value))>
void conv4_valid(const I& input, const K& kernel, C&& conv);

template <typename I, typename K, typename C, cpp_enable_if((!all_dma<I, K, C>::value))>
void conv2_full(const I& input, const K& kernel, C&& conv);

template <typename I, typename K, typename C, cpp_enable_if((!all_dma<I, K, C>::value))>
void conv4_full(const I& input, const K& kernel, C&& conv);

template <typename I, typename K, typename C, cpp_enable_if((!all_dma<I, K, C>::value))>
void conv2_valid_multi(const I& input, const K& kernel, C&& conv);

template <typename I, typename K, typename C, cpp_enable_if((!all_dma<I, K, C>::value))>
void conv2_valid_multi_flipped(const I& input, const K& kernel, C&& conv);

#else

//COVERAGE_EXCLUDE_BEGIN

template <typename I, typename K, typename C>
void conv2_valid(const I& input, const K& kernel, C&& conv);
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid");
}

template <typename I, typename K, typename C>
void conv4_valid(const I& input, const K& kernel, C&& conv);
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv4_valid");
}

template <typename I, typename K, typename C>
void conv2_full(const I& input, const K& kernel, C&& conv);
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_full");
}

template <typename I, typename K, typename C>
void conv4_full(const I& input, const K& kernel, C&& conv);
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv4_full");
}

template <typename I, typename K, typename C>
void conv2_valid_multi(const I& input, const K& kernel, C&& conv);
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid_multi");
}

template <typename I, typename K, typename C>
void conv2_valid_multi_flipped(const I& input, const K& kernel, C&& conv);
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("Unsupported feature called: cudnn conv2_valid_multi_flipped");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cudnn

} //end of namespace impl

} //end of namespace etl
