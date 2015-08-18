//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

//Include the implementations
#include "etl/impl/std/conv.hpp"
#include "etl/impl/sse/conv.hpp"
#include "etl/impl/avx/conv.hpp"
#include "etl/impl/reduc/conv_mmul.hpp"

namespace etl {

enum class conv_type {
    VALID,
    SAME,
    FULL
};

namespace detail {

enum class conv_impl {
    STD,
    SSE,
    AVX
};

template<typename T>
inline cpp14_constexpr conv_impl select_conv_impl(){
    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

    if(avx){
        return conv_impl::AVX;
    } else if(sse){
        return conv_impl::SSE;
    } else {
        return conv_impl::STD;
    }
}

template<typename I, typename K, typename C, typename Enable = void>
struct conv1_full_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        auto impl = select_conv_impl<value_t<I>>();

        if(impl == conv_impl::AVX){
            impl::avx::conv1_full(input, kernel, conv);
        } else if(impl == conv_impl::SSE){
            impl::sse::conv1_full(input, kernel, conv);
        } else if(impl == conv_impl::STD){
            impl::standard::conv1_full(input, kernel, conv);
        }
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv1_same_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        auto impl = select_conv_impl<value_t<I>>();

        if(impl == conv_impl::AVX){
            impl::avx::conv1_same(input, kernel, conv);
        } else if(impl == conv_impl::SSE){
            impl::sse::conv1_same(input, kernel, conv);
        } else if(impl == conv_impl::STD){
            impl::standard::conv1_same(input, kernel, conv);
        }
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv1_valid_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        auto impl = select_conv_impl<value_t<I>>();

        if(impl == conv_impl::AVX){
            impl::avx::conv1_valid(input, kernel, conv);
        } else if(impl == conv_impl::SSE){
            impl::sse::conv1_valid(input, kernel, conv);
        } else if(impl == conv_impl::STD){
            impl::standard::conv1_valid(input, kernel, conv);
        }
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv2_full_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        auto impl = select_conv_impl<value_t<I>>();

        if(impl == conv_impl::AVX){
            impl::avx::conv2_full(input, kernel, conv);
        } else if(impl == conv_impl::SSE){
            impl::sse::conv2_full(input, kernel, conv);
        } else if(impl == conv_impl::STD){
            impl::standard::conv2_full(input, kernel, conv);
        }
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv2_same_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        auto impl = select_conv_impl<value_t<I>>();

        if(impl == conv_impl::AVX){
            impl::avx::conv2_same(input, kernel, conv);
        } else if(impl == conv_impl::SSE){
            impl::sse::conv2_same(input, kernel, conv);
        } else if(impl == conv_impl::STD){
            impl::standard::conv2_same(input, kernel, conv);
        }
    }
};

template<typename I, typename K, typename C, typename Enable = void>
struct conv2_valid_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        auto impl = select_conv_impl<value_t<I>>();

        if(impl == conv_impl::AVX){
            impl::avx::conv2_valid(input, kernel, conv);
        } else if(impl == conv_impl::SSE){
            impl::sse::conv2_valid(input, kernel, conv);
        } else if(impl == conv_impl::STD){
            impl::standard::conv2_valid(input, kernel, conv);
        }
    }
};

template<conv_type TT, typename I, typename K, typename C, typename Enable = void>
struct conv_deep_impl {
    template<conv_type TT2 = TT, typename I2 = I, cpp_enable_if(decay_traits<I2>::dimensions() == 3 && TT2 == conv_type::FULL)>
    static void apply(const I& input, const K& kernel, C&& conv){
        for(std::size_t i = 0; i < dim<0>(input); ++i){
            conv(i) = conv_2d_full(input(i), kernel(i));
        }
    }

    template<conv_type TT2 = TT, typename I2 = I, cpp_enable_if(decay_traits<I2>::dimensions() == 3 && TT2 == conv_type::SAME)>
    static void apply(const I& input, const K& kernel, C&& conv){
        for(std::size_t i = 0; i < dim<0>(input); ++i){
            conv(i) = conv_2d_same(input(i), kernel(i));
        }
    }

    template<conv_type TT2 = TT, typename I2 = I, cpp_enable_if(decay_traits<I2>::dimensions() == 3 && TT2 == conv_type::VALID)>
    static void apply(const I& input, const K& kernel, C&& conv){
        for(std::size_t i = 0; i < dim<0>(input); ++i){
            conv(i) = conv_2d_valid(input(i), kernel(i));
        }
    }

    template<typename I2 = I, cpp_enable_if((decay_traits<I2>::dimensions() > 3))>
    static void apply(const I& input, const K& kernel, C&& conv){
        for(std::size_t i = 0; i < dim<0>(input); ++i){
            conv_deep_impl<TT, decltype(input(i)), decltype(kernel(i)), decltype(conv(i))>::apply(input(i), kernel(i), conv(i));
        }
    }
};

template<typename I, typename K, typename C>
using conv_deep_valid_impl = conv_deep_impl<conv_type::VALID, I, K, C>;

template<typename I, typename K, typename C>
using conv_deep_same_impl = conv_deep_impl<conv_type::SAME, I, K, C>;

template<typename I, typename K, typename C>
using conv_deep_full_impl = conv_deep_impl<conv_type::FULL, I, K, C>;

//The following partial specializations are here to ensure compilation
//(and avoid using static_if/SFINAE at higher level)

template<typename I, typename K, typename C>
struct conv1_full_impl<I, K, C, std::enable_if_t<!all_dma<I,K,C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/){
        cpp_unreachable("Should never be reached");
    }
};

template<typename I, typename K, typename C>
struct conv1_valid_impl<I, K, C, std::enable_if_t<!all_dma<I,K,C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/){
        cpp_unreachable("Should never be reached");
    }
};

template<typename I, typename K, typename C>
struct conv1_same_impl<I, K, C, std::enable_if_t<!all_dma<I,K,C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/){
        cpp_unreachable("Should never be reached");
    }
};

template<typename I, typename K, typename C>
struct conv2_full_impl<I, K, C, std::enable_if_t<!all_dma<I,K,C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/){
        cpp_unreachable("Should never be reached");
    }
};

template<typename I, typename K, typename C>
struct conv2_valid_impl<I, K, C, std::enable_if_t<!all_dma<I,K,C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/){
        cpp_unreachable("Should never be reached");
    }
};

template<typename I, typename K, typename C>
struct conv2_same_impl<I, K, C, std::enable_if_t<!all_dma<I,K,C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/){
        cpp_unreachable("Should never be reached");
    }
};

template<conv_type TT, typename I, typename K, typename C>
struct conv_deep_impl<TT, I, K, C, std::enable_if_t<!all_dma<I,K,C>::value>> {
    static void apply(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/){
        cpp_unreachable("Should never be reached");
    }
};

} //end of namespace detail

} //end of namespace etl
