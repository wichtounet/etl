//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Utilities to cache the descriptors of CUDNN operations.
 *
 * Unfortunately, this does not provide enough speedup to justify the increased
 * complexity in code. It does provide about 5% speedup for very small
 * operations, but the speedup is less than 1% for real-size operations.
 */

#pragma once

#ifdef ETL_CUDNN_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cudnn/cudnn.hpp"

#endif

namespace etl::impl::cudnn {

#ifdef ETL_CUDNN_MODE

/*!
 * \brief Compare the dimensions of the two given matrices
 */
template <typename M, cpp_enable_iff(is_1d<M>)>
bool fast_compare(M& lhs, M& rhs) {
    return etl::dim<0>(lhs) == etl::dim<0>(rhs);
}

/*!
 * \brief Compare the dimensions of the two given matrices
 */
template <typename M, cpp_enable_iff(is_2d<M>)>
bool fast_compare(M& lhs, M& rhs) {
    return etl::dim<0>(lhs) == etl::dim<0>(rhs) && etl::dim<1>(lhs) == etl::dim<1>(rhs);
}

/*!
 * \brief Compare the dimensions of the two given matrices
 */
template <typename M, cpp_enable_iff(is_3d<M>)>
bool fast_compare(M& lhs, M& rhs) {
    return etl::dim<0>(lhs) == etl::dim<0>(rhs) && etl::dim<1>(lhs) == etl::dim<1>(rhs) && etl::dim<2>(lhs) == etl::dim<2>(rhs);
}

/*!
 * \brief Compare the dimensions of the two given matrices
 */
template <typename M, cpp_enable_iff(is_4d<M>)>
bool fast_compare(M& lhs, M& rhs) {
    return etl::dim<0>(lhs) == etl::dim<0>(rhs) && etl::dim<1>(lhs) == etl::dim<1>(rhs) && etl::dim<2>(lhs) == etl::dim<2>(rhs)
           && etl::dim<3>(lhs) == etl::dim<3>(rhs);
}

/*!
 * \brief A key representing a matrix
 */
template <typename M, bool F, size_t D>
struct mat_cache_key_impl;

/*!
 * \copydoc mat_cache_key_impl
 */
template <typename M>
struct mat_cache_key_impl<M, false, 1> {
    size_t a; ///< The first dimension

    /*!
     * \brief Construct an empty (undefined values) mat_cache_key_impl
     */
    mat_cache_key_impl() {
        // Nothing else to init
    }

    /*!
     * \brief Construct a mat_cache_key_impl for the given matrix
     * \param mat The matrix to use as key
     */
    explicit mat_cache_key_impl(M& mat) : a(etl::dim<0>(mat)) {
        // Nothing else to init
    }

    /*!
     * \brief Compare the key with the given matrix
     * \param rhs The matrix to be compared against
     * \return true if the key is equivalent with the given matrix
     */
    bool operator==(M& rhs) {
        return a == etl::dim<0>(rhs);
    }
};

/*!
 * \copydoc mat_cache_key_impl
 */
template <typename M>
struct mat_cache_key_impl<M, false, 2> {
    size_t a; ///< The first dimension
    size_t b; ///< The second dimension

    /*!
     * \brief Construct an empty (undefined values) mat_cache_key_impl
     */
    mat_cache_key_impl() {
        // Nothing else to init
    }

    /*!
     * \brief Construct a mat_cache_key_impl for the given matrix
     * \param mat The matrix to use as key
     */
    explicit mat_cache_key_impl(M& mat) : a(etl::dim<0>(mat)), b(etl::dim<1>(mat)) {
        // Nothing else to init
    }

    /*!
     * \brief Compare the key with the given matrix
     * \param rhs The matrix to be compared against
     * \return true if the key is equivalent with the given matrix
     */
    bool operator==(M& rhs) {
        return a == etl::dim<0>(a) && b == etl::dim<1>(rhs);
    }
};

/*!
 * \copydoc mat_cache_key_impl
 */
template <typename M>
struct mat_cache_key_impl<M, false, 3> {
    size_t a; ///< The first dimension
    size_t b; ///< The second dimension
    size_t c; ///< The third dimension

    /*!
     * \brief Construct an empty (undefined values) mat_cache_key_impl
     */
    mat_cache_key_impl() {
        // Nothing else to init
    }

    /*!
     * \brief Construct a mat_cache_key_impl for the given matrix
     * \param mat The matrix to use as key
     */
    explicit mat_cache_key_impl(M& mat) : a(etl::dim<0>(mat)), b(etl::dim<1>(mat)), c(etl::dim<2>(mat)) {
        // Nothing else to init
    }

    /*!
     * \brief Compare the key with the given matrix
     * \param rhs The matrix to be compared against
     * \return true if the key is equivalent with the given matrix
     */
    bool operator==(M& rhs) {
        return a == etl::dim<0>(rhs) && b == etl::dim<1>(rhs) && c == etl::dim<2>(rhs);
    }
};

/*!
 * \copydoc mat_cache_key_impl
 */
template <typename M>
struct mat_cache_key_impl<M, false, 4> {
    size_t a; ///< The first dimension
    size_t b; ///< The second dimension
    size_t c; ///< The third dimension
    size_t d; ///< The fourth dimension

    /*!
     * \brief Construct an empty (undefined values) mat_cache_key_impl
     */
    mat_cache_key_impl() {
        // Nothing else to init
    }

    /*!
     * \brief Construct a mat_cache_key_impl for the given matrix
     * \param mat The matrix to use as key
     */
    explicit mat_cache_key_impl(M& mat) : a(etl::dim<0>(mat)), b(etl::dim<1>(mat)), c(etl::dim<2>(mat)), d(etl::dim<3>(mat)) {
        // Nothing else to init
    }

    /*!
     * \brief Compare the key with the given matrix
     * \param rhs The matrix to be compared against
     * \return true if the key is equivalent with the given matrix
     */
    bool operator==(M& rhs) {
        return a == etl::dim<0>(rhs) && b == etl::dim<1>(rhs) && c == etl::dim<2>(rhs) && d == etl::dim<3>(rhs);
    }
};

/*!
 * \brief A key representing a matrix
 */
template <typename M>
using mat_cache_key = mat_cache_key_impl<M, is_fast<M>, decay_traits<M>::dimensions()>;

/*!
 * \brief A key representing the union of the keys of three matrices
 */
template <typename A, typename B, typename C>
struct ternary_cache_key {
    mat_cache_key<A> key_a; ///< The key for the first matrix
    mat_cache_key<B> key_b; ///< The key for the second matrix
    mat_cache_key<C> key_c; ///< The key for the third matrix

    /*!
     * \brief Construct an empty key (undefined values)
     */
    ternary_cache_key() {
        // Nothing else to init
    }

    /*!
     * \brief Construct a key from the three given matrices
     * \param a The first matrix
     * \param b The second matrix
     * \param c The third matrix
     */
    ternary_cache_key(A& a, B& b, C& c) : key_a(a), key_b(b), key_c(c) {
        // Nothing else to init
    }

    /*!
     * \brief Test if the key is equivalent to the union of the three given
     * matrices.
     * \param a The first matrix
     * \param b The second matrix
     * \param c The third matrix
     * \return true if the key is equivalent, false otherwise
     */
    bool equals(A& a, B& b, C& c) {
        return key_a == a && key_b == b && key_c == c;
    }
};

/*!
 * \brief A static cache for ternary keys
 */
template <typename K, typename V, size_t L = 16>
struct ternary_static_cache {
    std::array<K, L> keys;   ///< The keys of the cache
    std::array<V, L> values; ///< The values of the cache

    size_t size = 0; ///< The current size of the cache

    static constexpr size_t last = L; ///< The past-the-end position

    /*!
     * \brief Return the position inside the cache of the given key
     * \param a The first matrix
     * \param b The second matrix
     * \param c The third matrix
     * \return the position of the key if found, otherwise, return last
     */
    template <typename A, typename B, typename C>
    size_t find(A& a, B& b, C& c) {
        for (size_t i = 0; i < size; ++i) {
            if (keys[i].equals(a, b, c)) {
                return i;
            }
        }

        return last;
    }

    /*!
     * \brief Insert a new element into the cache and returns its position
     * \param a The first matrix
     * \param b The second matrix
     * \param c The third matrix
     * \return the position of the inserted key if possible, otherwise,
     * if the cache is full, return last
     */
    template <typename A, typename B, typename C>
    size_t insert(A& a, B& b, C& c) {
        if (size == last - 1) {
            return last;
        }

        ++size;

        new (&keys[size - 1]) K(a, b, c);

        return size - 1;
    }

    /*!
     * \brief Return the value at the given position
     * \param i The index to get the value from
     * \return A reference to the value at the given position
     */
    auto& operator[](size_t i) {
        return values[i];
    }
};

/*!
 * \brief Descriptors for a 4D convolution.
 */
struct conv4_descriptor {
    cudnnTensorDescriptor_t input_tensor;     ///< The input tensor
    cudnnTensorDescriptor_t output_tensor;    ///< The output tensor
    cudnnFilterDescriptor_t filter;           ///< The filter descriptor
    cudnnConvolutionDescriptor_t convolution; ///< The convolution descriptor
    cudnnConvolutionFwdAlgo_t conv_algo;      ///< The convolution algorithm

    size_t workspace_size = 0; ///< The necessary size of the workspace
};

#endif

} //end of namespace etl::impl::cudnn
