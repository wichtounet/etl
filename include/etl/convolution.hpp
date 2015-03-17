//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONVOLUTION_HPP
#define ETL_CONVOLUTION_HPP

#include <algorithm>

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

//Get the implementations
#include "impl/conv.hpp"

namespace etl {

namespace detail {

template<std::size_t T, typename I, typename K, typename C, cpp::disable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_1d_sizes(const I& input, const K& kernel, const C& conv){
    static_assert(etl_traits<I>::dimensions() == 1 && etl_traits<K>::dimensions() == 1 && etl_traits<C>::dimensions() == 1, "Invalid dimensions for 1D convolution");

    if(T == 1){
        cpp_assert(dim<0>(conv) == dim<0>(input) + dim<0>(kernel) - 1, "Invalid sizes for 'full' convolution");
    } else if(T == 2){
        cpp_assert(dim<0>(conv) == dim<0>(input), "Invalid sizes for 'same' convolution");
    } else if(T == 3){
        cpp_assert(dim<0>(conv) == dim<0>(input) - dim<0>(kernel) + 1, "Invalid sizes for 'valid' convolution");
    }

    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
}

template<std::size_t T, typename I, typename K, typename C, cpp::enable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_1d_sizes(const I&, const K&, const C&){
    static_assert(etl_traits<I>::dimensions() == 1 && etl_traits<K>::dimensions() == 1 && etl_traits<C>::dimensions() == 1, "Invalid dimensions for 1D convolution");

    static_assert(
        T != 1 || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() + etl_traits<K>::template dim<0>() - 1,
        "Invalid sizes for 'full'convolution");
    static_assert(
        T != 2 || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>(),
        "Invalid sizes for 'same'convolution");
    static_assert(
        T != 3 || etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() - etl_traits<K>::template dim<0>() + 1,
        "Invalid sizes for 'valid'convolution");
}

template<std::size_t T, typename I, typename K, typename C, cpp::disable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_2d_sizes(const I& input, const K& kernel, const C& conv){
    static_assert(etl_traits<I>::dimensions() == 2 && etl_traits<K>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Invalid dimensions for 2D convolution");

    if(T == 1){
        cpp_assert(
                dim<0>(conv) == dim<0>(input) + dim<0>(kernel) - 1
            &&  dim<1>(conv) == dim<1>(input) + dim<1>(kernel) - 1,
            "Invalid sizes for 'valid' convolution");
    } else if(T == 2){
        cpp_assert(dim<0>(conv) == dim<0>(input) && dim<1>(conv) == dim<1>(input), "Invalid sizes for 'same' convolution");
    } else if(T == 3){
        cpp_assert(
                dim<0>(conv) == dim<0>(input) - dim<0>(kernel) + 1
            &&  dim<1>(conv) == dim<1>(input) - dim<1>(kernel) + 1,
            "Invalid sizes for 'valid' convolution");
    }

    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
}

template<std::size_t T, typename I, typename K, typename C, cpp::enable_if_all_u<etl_traits<I>::is_fast, etl_traits<K>::is_fast, etl_traits<C>::is_fast> = cpp::detail::dummy>
void check_conv_2d_sizes(const I&, const K&, const C&){
    static_assert(etl_traits<I>::dimensions() == 2 && etl_traits<K>::dimensions() == 2 && etl_traits<C>::dimensions() == 2, "Invalid dimensions for 2D convolution");

    static_assert(
        T != 1 || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() + etl_traits<K>::template dim<0>() - 1
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>() + etl_traits<K>::template dim<1>() - 1),
        "Invalid sizes for 'full'convolution");
    static_assert(
        T != 2 || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>()
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>()),
        "Invalid sizes for 'same'convolution");
    static_assert(
        T != 3 || (
                etl_traits<C>::template dim<0>() == etl_traits<I>::template dim<0>() - etl_traits<K>::template dim<0>() + 1
            &&  etl_traits<C>::template dim<1>() == etl_traits<I>::template dim<1>() - etl_traits<K>::template dim<1>() + 1),
        "Invalid sizes for 'valid'convolution");
}

} //end of namespace detail

//1D Convolution

template<typename I, typename K, typename C>
C& convolve_1d_full(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    detail::check_conv_1d_sizes<1>(input, kernel, conv);

    detail::conv1_full_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

template<typename I, typename K, typename C>
C& convolve_1d_same(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    detail::check_conv_1d_sizes<2>(input, kernel, conv);

    detail::conv1_same_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

template<typename I, typename K, typename C>
C& convolve_1d_valid(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    detail::check_conv_1d_sizes<3>(input, kernel, conv);

    detail::conv1_valid_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

//2D convolutions

template<typename I, typename K, typename C>
C& convolve_2d_full(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    detail::check_conv_2d_sizes<1>(input, kernel, conv);

    detail::conv2_full_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

template<typename I, typename K, typename C>
C& convolve_2d_same(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    detail::check_conv_2d_sizes<2>(input, kernel, conv);

    detail::conv2_same_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

template<typename I, typename K, typename C>
C& convolve_2d_valid(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    detail::check_conv_2d_sizes<3>(input, kernel, conv);

    detail::conv2_valid_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

//Deep convolutions

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_full(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_2d_full(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_full(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_full(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_same(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_2d_same(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_same(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_same(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<etl_traits<I>::dimensions() == 3> = cpp::detail::dummy>
C& convolve_deep_valid(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_2d_valid(input(i), kernel(i), conv(i));
    }

    return conv;
}

template<typename I, typename K, typename C, cpp::enable_if_u<(etl_traits<I>::dimensions() > 3)> = cpp::detail::dummy>
C& convolve_deep_valid(const I& input, const K& kernel, C&& conv){
    static_assert(dimensions<I>() == dimensions<K>() && dimensions<I>()== dimensions<C>(), "Deep convolution parameters need to have the same number of dimensions");
    static_assert(dim<0, I>() == dim<0, K>() && dim<0, I>() == dim<0, C>(), "Deep convolution parameters need to have the same first dimension");

    for(std::size_t i = 0; i < dim<0>(input); ++i){
        convolve_deep_valid(input(i), kernel(i), conv(i));
    }

    return conv;
}

} //end of namespace etl

#endif
