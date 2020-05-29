//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Utility functions for cudnn
 */

#pragma once

#include "cudnn.h"

#define cudnn_check(call)                                                                                 \
    {                                                                                                     \
        cudnnStatus_t status = call;                                                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                                                             \
            std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                             \
        }                                                                                                 \
    }

namespace etl::impl::cudnn {

/*!
 * \brief RTTI helper to manage CUDNN handle
 */
struct cudnn_handle {
    cudnnHandle_t handle; ///< The raw cudnn handle

    /*!
     * \brief Construct the helper and create the handle directly
     */
    cudnn_handle() {
        cudnn_check(cudnnCreate(&handle));
    }

    /*!
     * \brief Construct the helper from the raw handle
     * \param handle The raw cudnn handle
     */
    explicit cudnn_handle(cudnnHandle_t handle) : handle(handle) {}

    cudnn_handle(const cudnn_handle& rhs) = delete;
    cudnn_handle& operator=(const cudnn_handle& rhs) = delete;

    cudnn_handle(cudnn_handle&& rhs) noexcept = default;
    cudnn_handle& operator=(cudnn_handle&& rhs) noexcept = default;

    /*!
     * \brief Get the cudnn handle
     * \return the raw cudnn handle
     */
    cudnnHandle_t get() {
        return handle;
    }

    /*!
     * \brief Destruct the helper and release the raw cudnn handle
     */
    ~cudnn_handle() {
        cudnn_check(cudnnDestroy(handle));
    }
};

#ifndef ETL_CUDNN_LOCAL_HANDLE

/*!
 * \brief Start cudnn and return a RTTI helper over a raw cudnn handle
 * \return RTTI helper over a raw cudnn handle
 */
inline cudnn_handle& start_cudnn() {
    static cudnn_handle handle;
    return handle;
}

#else

/*!
 * \brief Start cudnn and return a RTTI helper over a raw cudnn handle
 * \return RTTI helper over a raw cudnn handle
 */
inline cudnn_handle start_cudnn() {
    return {};
}

#endif

/*!
 * \brief Wrapper for CUDNN tensor.
 *
 * Needs to be specialized for the destructor.
 */
template <typename T>
struct cudnn_wrapper {
    T tensor; ///< The CUDNN tensor

    /*!
     * \brief Create a new cudnn_wrapper for the given tensor.
     */
    explicit cudnn_wrapper(T tensor) : tensor(tensor) {}

    /*!
     * \brief Move construct a cudnn_wrapper.
     *
     * This will remove the tensor from the RHS.
     *
     * \param rhs The right hand side cudnn_wrapper
     */
    cudnn_wrapper(cudnn_wrapper&& rhs) {
        tensor     = rhs.tensor;
        rhs.tensor = nullptr;
    }

    cudnn_wrapper(const cudnn_wrapper& rhs) = delete;
    cudnn_wrapper& operator=(const cudnn_wrapper& rhs) = delete;

    cudnn_wrapper& operator=(cudnn_wrapper&& rhs) = delete;

    /*!
     * \brief Convert back to the CUDNN tensor type.
     * \return The CUDNN tensor.
     */
    T operator*() {
        return tensor;
    }

    /*!
     * \brief Delete the cudnn_wrapper, releasing the tensor.
     */
    ~cudnn_wrapper();
};

/*!
 * \copydoc cudnn_wraper::~cudnn_wrapper()
 */
template <>
inline cudnn_wrapper<cudnnTensorDescriptor_t>::~cudnn_wrapper() {
    if (tensor) {
        cudnn_check(cudnnDestroyTensorDescriptor(tensor));
    }
}

/*!
 * \copydoc cudnn_wraper::~cudnn_wrapper()
 */
template <>
inline cudnn_wrapper<cudnnFilterDescriptor_t>::~cudnn_wrapper() {
    if (tensor) {
        cudnn_check(cudnnDestroyFilterDescriptor(tensor));
    }
}

/*!
 * \copydoc cudnn_wraper::~cudnn_wrapper()
 */
template <>
inline cudnn_wrapper<cudnnPoolingDescriptor_t>::~cudnn_wrapper() {
    if (tensor) {
        cudnn_check(cudnnDestroyPoolingDescriptor(tensor));
    }
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_1d<I>)>
cudnnTensorDescriptor_t create_tensor(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, data_type, 1, etl::dim<0>(input), 1, 1));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_2d<I>)>
cudnnTensorDescriptor_t create_tensor(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, etl::dim<0>(input), etl::dim<1>(input)));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_3d<I>)>
cudnnTensorDescriptor_t create_tensor(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, data_type, 1, etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input)));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_4d<I>)>
cudnnTensorDescriptor_t create_tensor(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(
        cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, data_type, etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input)));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return a cudnn_wrapper around a created CUDNN tensor
 */
template <typename I>
cudnn_wrapper<cudnnTensorDescriptor_t> create_tensor_wrapper(I&& input) {
    return cudnn_wrapper<cudnnTensorDescriptor_t>{create_tensor(input)};
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_3d<I>)>
cudnnTensorDescriptor_t create_tensor_5d(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    const int D1 = etl::dim<0>(input);
    const int D2 = etl::dim<1>(input);
    const int D3 = etl::dim<2>(input);

    int dims[]    = {1, 1, D1, D2, D3};
    int strides[] = {D1 * D2 * D3, D1 * D2 * D3, D2 * D3, D3, 1};

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(cudnnSetTensorNdDescriptor(tensor, data_type, 5, dims, strides));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_4d<I>)>
cudnnTensorDescriptor_t create_tensor_5d(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    const int D1 = etl::dim<0>(input);
    const int D2 = etl::dim<1>(input);
    const int D3 = etl::dim<2>(input);
    const int D4 = etl::dim<3>(input);

    int dims[]    = {1, D1, D2, D3, D4};
    int strides[] = {D1 * D2 * D3 * D4, D2 * D3 * D4, D3 * D4, D4, 1};

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(cudnnSetTensorNdDescriptor(tensor, data_type, 5, dims, strides));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return a cudnn_wrapper around a created CUDNN tensor
 */
template <typename I>
cudnn_wrapper<cudnnTensorDescriptor_t> create_tensor_wrapper_5d(I&& input) {
    return cudnn_wrapper<cudnnTensorDescriptor_t>{create_tensor_5d(input)};
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I>
cudnnTensorDescriptor_t create_tensor_flat(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    // Surprisingly, CUDNN does not do any optimization for flat vectors
    // It means that the position of the dimension is very important
    // Putting at the first position (N) is generally the slowest case
    // But putting it at the last (W) seems better

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, data_type, 1, 1, 1, etl::size(input)));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return a cudnn_wrapper around a created CUDNN tensor
 */
template <typename I>
cudnn_wrapper<cudnnTensorDescriptor_t> create_tensor_flat_wrapper(I&& input) {
    return cudnn_wrapper<cudnnTensorDescriptor_t>{create_tensor_flat(input)};
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_1d<I>)>
cudnnTensorDescriptor_t create_tensor_front(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, data_type, etl::dim<0>(input), 1, 1, 1));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_2d<I>)>
cudnnTensorDescriptor_t create_tensor_front(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, data_type, etl::dim<0>(input), etl::dim<1>(input), 1, 1));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN tensor
 */
template <typename I, cpp_enable_iff(is_4d<I>)>
cudnnTensorDescriptor_t create_tensor_front(I&& input) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnTensorDescriptor_t tensor;
    cudnn_check(cudnnCreateTensorDescriptor(&tensor));
    cudnn_check(
        cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, data_type, etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input)));

    return tensor;
}

/*!
 * \brief Create a CUDNN tensor for the given input matrix
 * \param input The input matrix
 * \return a cudnn_wrapper around a created CUDNN tensor
 */
template <typename I>
cudnn_wrapper<cudnnTensorDescriptor_t> create_tensor_front_wrapper(I&& input) {
    return cudnn_wrapper<cudnnTensorDescriptor_t>{create_tensor_front(input)};
}

/*!
 * \brief Create a CUDNN filter tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN filter tensor
 */
template <typename I, cpp_enable_iff(is_2d<I>)>
cudnnFilterDescriptor_t create_filter(I&& kernel) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, 1, 1, etl::dim<0>(kernel), etl::dim<1>(kernel)));

    return filter;
}

/*!
 * \brief Create a CUDNN filter tensor for the given input matrix
 * \param input The input matrix
 * \return the created CUDNN filter tensor
 */
template <typename I, cpp_enable_iff(is_4d<I>)>
cudnnFilterDescriptor_t create_filter(I&& kernel) {
    using T = value_t<I>;

    auto data_type = std::is_same_v<std::remove_const_t<T>, float> ? CUDNN_DATA_FLOAT : CUDNN_DATA_DOUBLE;

    cudnnFilterDescriptor_t filter;
    cudnn_check(cudnnCreateFilterDescriptor(&filter));
    cudnn_check(
        cudnnSetFilter4dDescriptor(filter, data_type, CUDNN_TENSOR_NCHW, etl::dim<0>(kernel), etl::dim<1>(kernel), etl::dim<2>(kernel), etl::dim<3>(kernel)));

    return filter;
}

/*!
 * \brief Create a CUDNN filter tensor for the given input matrix
 * \param input The input matrix
 * \return a cudnn_wrapper around a created CUDNN filter tensor
 */
template <typename I>
cudnn_wrapper<cudnnFilterDescriptor_t> create_filter_wrapper(I&& kernel) {
    return cudnn_wrapper<cudnnFilterDescriptor_t>{create_filter(kernel)};
}

/*!
 * \brief Create a CUDNN pooling descriptor for the given input matrix
 * \return the created CUDNN filter tensor
 */
inline cudnnPoolingDescriptor_t create_pooling_desc(cudnnPoolingMode_t mode, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
    cudnnPoolingDescriptor_t pooling_desc;
    cudnn_check(cudnnCreatePoolingDescriptor(&pooling_desc));
    cudnn_check(cudnnSetPooling2dDescriptor(pooling_desc, mode, CUDNN_PROPAGATE_NAN, c1, c2, p1, p2, s1, s2));

    return pooling_desc;
}

/*!
 * \brief Create a CUDNN pooling descriptor for the given input matrix
 * \return a cudnn_wrapper around a created CUDNN filter tensor
 */
inline cudnn_wrapper<cudnnPoolingDescriptor_t> create_pooling_desc_wrapper(
    cudnnPoolingMode_t mode, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
    return cudnn_wrapper<cudnnPoolingDescriptor_t>{create_pooling_desc(mode, c1, c2, s1, s2, p1, p2)};
}

/*!
 * \brief Create a CUDNN pooling descriptor for the given input matrix
 * \return the created CUDNN filter tensor
 */
inline cudnnPoolingDescriptor_t create_pooling_desc(
    cudnnPoolingMode_t mode, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
    int c[] = {int(c1), int(c2), int(c3)};
    int s[] = {int(s1), int(s2), int(s3)};
    int p[] = {int(p1), int(p2), int(p3)};

    cudnnPoolingDescriptor_t pooling_desc;
    cudnn_check(cudnnCreatePoolingDescriptor(&pooling_desc));
    cudnn_check(cudnnSetPoolingNdDescriptor(pooling_desc, mode, CUDNN_PROPAGATE_NAN, 3, c, p, s));

    return pooling_desc;
}

/*!
 * \brief Create a CUDNN pooling descriptor for the given input matrix
 * \return a cudnn_wrapper around a created CUDNN filter tensor
 */
inline cudnn_wrapper<cudnnPoolingDescriptor_t> create_pooling_desc_wrapper(
    cudnnPoolingMode_t mode, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
    return cudnn_wrapper<cudnnPoolingDescriptor_t>{create_pooling_desc(mode, c1, c2, c3, s1, s2, s3, p1, p2, p3)};
}

template <typename Value>
struct cudnn_desc_cache_value {
    bool found;

#ifdef ETL_CUDNN_DESC_CACHE
    Value & value;
#else
    Value value;
#endif

    operator bool() const {
        return found;
    }

    Value * operator->() {
        return &value;
    }
};

template <typename Key, typename Value>
struct cudnn_desc_cache {
#ifdef ETL_CUDNN_DESC_CACHE
    std::vector<std::pair<Key, std::unique_ptr<Value>>> cache;

    cudnn_desc_cache_value<Value> operator[]([[maybe_unused]] const Key& key) {
        for (auto & [k, value] : cache) {
            if (k == key) {
                return {true, *value};
            }
        }

        cache.emplace_back(key, std::make_unique<Value>());

        return {false, *cache.back().second};
    }
#else
    cudnn_desc_cache_value<Value> operator[]([[maybe_unused]] const Key& key) {
        return {false, Value{}};
    }
#endif
};

} //end of namespace etl::impl::cudnn
