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

#include <immintrin.h>

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
C& convolve_1d_full(const I& input, const K& kernel, C&& conv){
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
C& convolve_1d_same(const I& input, const K& kernel, C&& conv){
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

namespace detail {

template<typename I, typename K, typename C, typename Enable = void>
struct conv1_valid_impl {
    static void apply(const I& input, const K& kernel, C&& conv){
        for(std::size_t j = 0 ; j < size(conv) ; ++j){
            double temp = 0.0;

            for(std::size_t l = j ; l <= j + size(kernel) - 1; ++l){
                temp += input[l] * kernel[j + size(kernel) - 1 - l];
            }

            conv[j] = temp;
        }
    }
};

#ifdef ETL_VECTORIZE

#ifdef __SSE3__

template<typename I, typename K, typename C>
struct is_fast_dconv : cpp::bool_constant_c<cpp::and_c<
          is_double_precision<I>, is_double_precision<K>, is_double_precision<C>
        , has_direct_access<I>, has_direct_access<K>, has_direct_access<C>
    >> {};

template<typename I, typename K, typename C>
struct is_fast_sconv : cpp::bool_constant_c<cpp::and_c<
          is_single_precision<I>, is_single_precision<K>, is_single_precision<C>
        , has_direct_access<I>, has_direct_access<K>, has_direct_access<C>
    >> {};

template<typename I, typename K, typename C>
struct conv1_valid_impl<I, K, C, std::enable_if_t<is_fast_dconv<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        __m128* kernel_reverse = new __m128[kernel.size()];

        auto out = conv.memory_start();
        auto in = input.memory_start();
        auto k = kernel.memory_start();

        for(int i=0; i< kernel.size(); i++){
            kernel_reverse[i] = _mm_load1_pd(k + kernel.size() - i - 1);
        }

        register __m128 tmp1;
        register __m128 tmp2;
        register __m128 res;

        for(std::size_t i=0; i<input.size()-kernel.size(); i+=2){
            res = _mm_setzero_pd();

            for(std::size_t k=0; k<kernel.size(); k++){
                tmp1 = _mm_loadu_pd(in + i + k);
                tmp2 = _mm_mul_pd(kernel_reverse[k], tmp1);
                res = _mm_add_pd(res, tmp2);
            }

            _mm_storeu_pd(out+i, res);
        }

        auto i = input.size() - kernel.size();
        conv[i] = 0.0;
        for(int k=0; k<kernel.size(); k++){
            conv[i] += input[i+k] * kernel[kernel.size() - k - 1];
        }

        delete[] kernel_reverse;
    }
};

template<typename I, typename K, typename C>
struct conv1_valid_impl<I, K, C, std::enable_if_t<is_fast_sconv<I,K,C>::value>> {
    static void apply(const I& input, const K& kernel, C&& conv){
        __m128* kernel_reverse = new __m128[kernel.size()];

        auto out = conv.memory_start();
        auto in = input.memory_start();
        auto k = kernel.memory_start();

        for(std::size_t i=0; i< kernel.size(); i++){
            kernel_reverse[i] = _mm_load1_ps(k + kernel.size() - i - 1);
        }

        register __m128 tmp1;
        register __m128 tmp2;
        register __m128 res;

        for(std::size_t i=0; i<input.size()-kernel.size(); i+=4){
            res = _mm_setzero_ps();

            for(std::size_t k=0; k<kernel.size(); k++){
                tmp1 = _mm_loadu_ps(in + i + k);
                tmp2 = _mm_mul_ps(kernel_reverse[k], tmp1);
                res = _mm_add_ps(res, tmp2);
            }

            _mm_storeu_ps(out+i, res);
        }

        auto i = input.size() - kernel.size();
        conv[i] = 0.0;
        for(int k=0; k<kernel.size(); k++){
            conv[i] += input[i+k] * kernel[kernel.size() - k - 1];
        }

        delete[] kernel_reverse;
    }
};

#endif //__SSE3__

#endif //ETL_VECTORIZE

} //end of namespace detail

template<typename I, typename K, typename C>
C& convolve_1d_valid(const I& input, const K& kernel, C&& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    check_conv_1d_sizes<3>(input, kernel, conv);

    detail::conv1_valid_impl<I,K,C>::apply(input, kernel, std::forward<C>(conv));

    return conv;
}

template<typename I, typename K, typename C>
C& convolve_2d_full(const I& input, const K& kernel, C&& conv){
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
C& convolve_2d_same(const I& input, const K& kernel, C&& conv){
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
C& convolve_2d_valid(const I& input, const K& kernel, C&& conv){
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

} //end of namespace etl

#endif
