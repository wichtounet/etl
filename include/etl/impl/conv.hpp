//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_IMPL_CONVOLUTION_HPP
#define ETL_IMPL_CONVOLUTION_HPP

#include <algorithm>

//Include the implementations
#include "std/conv.hpp"
#include "sse/conv.hpp"
#include "avx/conv.hpp"

namespace etl {

namespace detail {

template<typename I, typename K, typename C>
struct c_is_single_precision : cpp::bool_constant_c<cpp::and_c<is_single_precision<I>, is_single_precision<K>, is_single_precision<C>>> {};

template<typename I, typename K, typename C>
struct c_is_double_precision : cpp::bool_constant_c<cpp::and_c<is_double_precision<I>, is_double_precision<K>, is_double_precision<C>>> {};

template<typename I, typename K, typename C>
struct c_is_dma : cpp::bool_constant_c<cpp::and_c<has_direct_access<I>, has_direct_access<K>, has_direct_access<C>>> {};

template<typename I, typename K, typename C, typename Enable = void>
struct conv1_full_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::standard::conv1_full(input, kernel, conv);
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv1_same_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::standard::conv1_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv1_valid_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::standard::conv1_valid(input, kernel, conv);
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv2_full_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::standard::conv2_full(input, kernel, conv);
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv2_same_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::standard::conv2_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv2_valid_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::standard::conv2_valid(input, kernel, conv);
    }
};

#ifdef ETL_VECTORIZE

#ifdef __AVX__

template<typename I, typename K, typename C>
struct conv1_full_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::dconv1_full(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_same_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::dconv1_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_valid_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::dconv1_valid(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_full_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::sconv1_full(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_same_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::sconv1_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_valid_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::sconv1_valid(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_full_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::dconv2_full(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_same_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::dconv2_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_valid_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::dconv2_valid(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_same_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::sconv2_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_valid_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::avx::sconv2_valid(input, kernel, conv);
    }
};

#elif defined(__SSE3__)

template<typename I, typename K, typename C>
struct conv1_full_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::dconv1_full(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_same_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::dconv1_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_valid_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::dconv1_valid(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_full_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::dconv2_full(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_same_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::dconv2_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_valid_impl<I, K, C, std::enable_if_t<c_is_double_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::dconv2_valid(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_full_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::sconv1_full(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_same_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::sconv1_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv1_valid_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::sconv1_valid(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_valid_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::sconv2_valid(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_same_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::sconv2_same(input, kernel, conv);
    }
};

template<typename I, typename K, typename C>
struct conv2_full_impl<I, K, C, std::enable_if_t<c_is_single_precision<I,K,C>::value && c_is_dma<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        impl::sse::sconv2_full(input, kernel, conv);
    }
};

#endif //__SSE3__

#endif //ETL_VECTORIZE

} //end of namespace detail

} //end of namespace etl

#endif
