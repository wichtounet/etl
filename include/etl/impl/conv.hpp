//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the convolution implementations.
 *
 * The functions are responsible for selecting the most efficient
 * implementation for each case, based on what is available. The selection of
 * parallel versus serial is also done at this level. The implementation
 * functions should never be used directly, only functions of this header can
 * be used directly.
 *
 * Ideas for improvements:
 *  * Parallel dispatching for SSE/AVX implementation is not perfect, it should be done inside the micro kernel main loop
 */

#pragma once

//Include the implementations
#include "etl/impl/std/conv.hpp"
#include "etl/impl/sse/conv.hpp"
#include "etl/impl/avx/conv.hpp"
#include "etl/impl/reduc/conv_mmul.hpp"

namespace etl {

/*!
 * \brief Enumeration describing the different types of convolution
 */
enum class conv_type {
    VALID, ///< Valid convolution
    SAME,  ///< Same convolution
    FULL   ///< Full convolution
};

namespace detail {

/*!
 * \brief Enumeration describing the different convolution implementations
 */
enum class conv_impl {
    STD, ///< Standard implementation
    SSE, ///< Vectorized SSE implementation
    AVX  ///< Vectorized AVX implementation
};

template <typename I, typename K, typename C>
inline conv_impl select_conv_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr const order input_order  = decay_traits<I>::storage_order;
    static constexpr const order kernel_order = decay_traits<K>::storage_order;
    static constexpr const order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return conv_impl::STD;
    }

    static constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

    if (avx) {
        return conv_impl::AVX;
    } else if (sse) {
        return conv_impl::SSE;
    } else {
        return conv_impl::STD;
    }
}

template <typename I, typename K, typename C>
inline bool select_parallel(const I& /*input*/, const K& kernel, C&& conv) {
    if(parallel){
        return size(conv) >= conv1_parallel_threshold_conv && size(kernel) >= conv1_parallel_threshold_kernel;
    } else {
        return false;
    }
}

template <typename I, typename K, typename C, typename Enable = void>
struct conv1_full_impl {
    static void apply(const I& input, const K& kernel, C&& conv) {
        conv_impl impl = select_conv_impl<I, K, C>();
        selected_apply(input, kernel, std::forward<C>(conv), impl);
    }

    static void selected_apply(const I& input, const K& kernel, C&& conv, conv_impl impl) {
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::avx::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::sse::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::standard::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
        }
    }
};

template <typename I, typename K, typename C, typename Enable = void>
struct conv1_same_impl {
    static void apply(const I& input, const K& kernel, C&& conv) {
        conv_impl impl = select_conv_impl<I, K, C>();
        selected_apply(input, kernel, std::forward<C>(conv), impl);
    }

    static void selected_apply(const I& input, const K& kernel, C&& conv, conv_impl impl) {
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::avx::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::sse::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::standard::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
        }
    }
};

template <typename I, typename K, typename C, typename Enable = void>
struct conv1_valid_impl {
    static void apply(const I& input, const K& kernel, C&& conv) {
        conv_impl impl = select_conv_impl<I, K, C>();
        selected_apply(input, kernel, std::forward<C>(conv), impl);
    }

    static void selected_apply(const I& input, const K& kernel, C&& conv, conv_impl impl) {
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::avx::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::sse::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::standard::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        }
    }
};

//Should only be used by the benchmark
template <typename I, typename K, typename C>
void conv1_full_direct(const I& input, const K& kernel, C&& conv, conv_impl impl){
    conv1_full_impl<I, K, C>::selected_apply(input, kernel, std::forward<C>(conv), impl);
}

//Should only be used by the benchmark
template <typename I, typename K, typename C>
void conv1_same_direct(const I& input, const K& kernel, C&& conv, conv_impl impl){
    conv1_same_impl<I, K, C>::selected_apply(input, kernel, std::forward<C>(conv), impl);
}

//Should only be used by the benchmark
template <typename I, typename K, typename C>
void conv1_valid_direct(const I& input, const K& kernel, C&& conv, conv_impl impl){
    conv1_valid_impl<I, K, C>::selected_apply(input, kernel, std::forward<C>(conv), impl);
}

template <typename I, typename K, typename C, typename Enable = void>
struct conv2_full_impl {
    static void apply(const I& input, const K& kernel, C&& conv) {
        conv_impl impl = select_conv_impl<I, K, C>();

        if (impl == conv_impl::AVX) {
            impl::avx::conv2_full(input, kernel, conv);
        } else if (impl == conv_impl::SSE) {
            impl::sse::conv2_full(input, kernel, conv);
        } else if (impl == conv_impl::STD) {
            impl::standard::conv2_full(input, kernel, conv);
        }
    }
};

template <typename I, typename K, typename C, typename Enable = void>
struct conv2_same_impl {
    static void apply(const I& input, const K& kernel, C&& conv) {
        conv_impl impl = select_conv_impl<I, K, C>();

        if (impl == conv_impl::AVX) {
            impl::avx::conv2_same(input, kernel, conv);
        } else if (impl == conv_impl::SSE) {
            impl::sse::conv2_same(input, kernel, conv);
        } else if (impl == conv_impl::STD) {
            impl::standard::conv2_same(input, kernel, conv);
        }
    }
};

template <typename I, typename K, typename C, typename Enable = void>
struct conv2_valid_impl {
    static void apply(const I& input, const K& kernel, C&& conv) {
        conv_impl impl = select_conv_impl<I, K, C>();

        if (impl == conv_impl::AVX) {
            impl::avx::conv2_valid(input, kernel, conv);
        } else if (impl == conv_impl::SSE) {
            impl::sse::conv2_valid(input, kernel, conv);
        } else if (impl == conv_impl::STD) {
            impl::standard::conv2_valid(input, kernel, conv);
        }
    }
};

template <conv_type TT, typename I, typename K, typename C, typename Enable = void>
struct conv_deep_impl {
    template <conv_type TT2 = TT, typename I2 = I, cpp_enable_if(decay_traits<I2>::dimensions() == 3, TT2 == conv_type::FULL)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            conv(i) = conv_2d_full(input(i), kernel(i));
        }
    }

    template <conv_type TT2 = TT, typename I2 = I, cpp_enable_if(decay_traits<I2>::dimensions() == 3, TT2 == conv_type::SAME)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            conv(i) = conv_2d_same(input(i), kernel(i));
        }
    }

    template <conv_type TT2 = TT, typename I2 = I, cpp_enable_if(decay_traits<I2>::dimensions() == 3, TT2 == conv_type::VALID)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            conv(i) = conv_2d_valid(input(i), kernel(i));
        }
    }

    template <typename I2 = I, cpp_enable_if((decay_traits<I2>::dimensions() > 3))>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            conv_deep_impl<TT, decltype(input(i)), decltype(kernel(i)), decltype(conv(i))>::apply(input(i), kernel(i), conv(i));
        }
    }
};

template <typename I, typename K, typename C, typename = void>
using conv_deep_valid_impl = conv_deep_impl<conv_type::VALID, I, K, C>;

template <typename I, typename K, typename C, typename = void>
using conv_deep_same_impl = conv_deep_impl<conv_type::SAME, I, K, C>;

template <typename I, typename K, typename C, typename = void>
using conv_deep_full_impl = conv_deep_impl<conv_type::FULL, I, K, C>;

//The following partial specializations are here to ensure compilation
//(and avoid using static_if/SFINAE at higher level)

template <typename I, typename K, typename C>
struct conv1_full_impl<I, K, C, std::enable_if_t<!all_dma<I, K, C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
        cpp_unreachable("Should never be reached");
    }
};

template <typename I, typename K, typename C>
struct conv1_valid_impl<I, K, C, std::enable_if_t<!all_dma<I, K, C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
        cpp_unreachable("Should never be reached");
    }
};

template <typename I, typename K, typename C>
struct conv1_same_impl<I, K, C, std::enable_if_t<!all_dma<I, K, C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
        cpp_unreachable("Should never be reached");
    }
};

template <typename I, typename K, typename C>
struct conv2_full_impl<I, K, C, std::enable_if_t<!all_dma<I, K, C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
        cpp_unreachable("Should never be reached");
    }
};

template <typename I, typename K, typename C>
struct conv2_valid_impl<I, K, C, std::enable_if_t<!all_dma<I, K, C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
        cpp_unreachable("Should never be reached");
    }
};

template <typename I, typename K, typename C>
struct conv2_same_impl<I, K, C, std::enable_if_t<!all_dma<I, K, C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
        cpp_unreachable("Should never be reached");
    }
};

template <conv_type TT, typename I, typename K, typename C>
struct conv_deep_impl<TT, I, K, C, std::enable_if_t<!all_dma<I, K, C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
        cpp_unreachable("Should never be reached");
    }
};

} //end of namespace detail

} //end of namespace etl
