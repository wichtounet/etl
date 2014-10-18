//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONVOLUTION_HPP
#define ETL_CONVOLUTION_HPP

#include <algorithm>

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

namespace etl {

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

template<typename I, typename K, typename C>
static C& convolve_1d_full(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    check_conv_1d_sizes<1>(input, kernel, conv);

    for(std::size_t i = 0; i < size(conv); ++i) {
        const auto lo = i >= size(kernel) - 1 ? i - (size(kernel) - 1) : 0;
        const auto hi = i < size(input) - 1 ? i : size(input) - 1;

        double temp = 0.0;

        for(std::size_t j = lo; j <= hi; ++j) {
            temp += input[j] * kernel[i - j];
        }

        conv[i] = temp;
    }

    return conv;
}

template<typename I, typename K, typename C>
static C& convolve_1d_same(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    check_conv_1d_sizes<2>(input, kernel, conv);

    for(std::size_t j = 0 ; j < size(conv) ; ++j){
        int l_lo = std::max<int>(0, j - (size(kernel) - 1) / 2);
        int l_hi = std::min<int>(size(input)- 1, j + size(kernel) / 2);

        double temp = 0.0;

        for(std::size_t l = l_lo ; l <= static_cast<std::size_t>(l_hi); ++l){
            temp += input(l) * kernel(j - l + size(kernel) / 2);
        }

        conv(0 + j) = temp;
    }

    return conv;
}

template<typename I, typename K, typename C>
static C& convolve_1d_valid(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    check_conv_1d_sizes<3>(input, kernel, conv);

    for(std::size_t j = 0 ; j < size(conv) ; ++j){
        double temp = 0.0;

        for(std::size_t l = j ; l <= j + size(kernel) - 1; ++l){
            temp += input[l] * kernel[j + size(kernel) - 1 - l];
        }

        conv[j] = temp;
    }

    return conv;
}

template<typename I, typename K, typename C>
static C& convolve_2d_full(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    check_conv_2d_sizes<1>(input, kernel, conv);

    for(std::size_t i = 0 ; i < rows(conv) ; ++i){
        auto k_lo = std::max<int>(0, i - rows(kernel) + 1);
        auto k_hi = std::min(rows(input) - 1, i);

        for(std::size_t j = 0 ; j < columns(conv) ; ++j){
            auto l_lo = std::max<int>(0, j - columns(kernel) + 1);
            auto l_hi = std::min(columns(input) - 1 ,j);

            double temp = 0.0;

            for(std::size_t k = k_lo ; k <= k_hi ; ++k){
                for(std::size_t l = l_lo ; l <= l_hi ; ++l){
                    temp += input(k,l) * kernel(i - k, j - l);
                }
            }

            conv(i, j) = temp;
        }
    }

    return conv;
}

template<typename I, typename K, typename C>
static C& convolve_2d_same(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    check_conv_2d_sizes<2>(input, kernel, conv);

    for(std::size_t i = 0 ; i < rows(conv); ++i){
        auto k_lo = std::max<int>(0, i - (rows(kernel)-1)/2);
        auto k_hi = std::min<int>(rows(input) - 1, i + rows(kernel)/2);

        for(std::size_t j = 0 ; j < columns(conv); ++j){
            auto l_lo = std::max<int>(0, j - (columns(kernel)-1)/2);
            auto l_hi = std::min<int>(columns(input) - 1, j + columns(kernel)/2);

            double temp = 0.0;

            for(int k = k_lo ; k <= k_hi ; ++k){
                for(std::size_t l = l_lo ; l <= static_cast<std::size_t>(l_hi); ++l){
                    temp += input(k, l) * kernel(i-k+rows(kernel)/2, j-l+columns(kernel)/2);
                }
            }

            conv(i, j) = temp;
        }
    }

    return conv;
}

template<typename I, typename K, typename C>
static C& convolve_2d_valid(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    check_conv_2d_sizes<3>(input, kernel, conv);

    for(std::size_t i = 0 ; i < rows(conv) ; ++i){
        for(std::size_t j = 0 ; j < columns(conv) ; ++j){
            double temp = 0.0;

            for(std::size_t k = i ; k <= i + rows(kernel)-1; ++k){
                for(std::size_t l = j ; l <= j + columns(kernel)-1 ; ++l){
                    temp += input(k,l) * kernel((i+rows(kernel)-1-k), (j+columns(kernel)-1-l));
                }
            }

            conv(i,j) = temp;
        }
    }

    return conv;
}

} //end of namespace etl

#endif
