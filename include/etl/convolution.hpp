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

#include "tmp.hpp"

namespace etl {

template<typename I, typename K, typename C>
static C& convolve_1d_full(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    //static_assert(C::etl_size == I::etl_size + K::etl_size - 1, "Invalid output vector size");
    cpp_assert(size(conv) == size(input) + size(kernel) - 1, "Invalid output vector size");

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
static C& convolve_1d_same(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    //static_assert(C::etl_size == I::etl_size, "Invalid output vector size");

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
static C& convolve_1d_valid(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    //static_assert(C::etl_size == I::etl_size - K::etl_size + 1, "Invalid output vector size");

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
static C& convolve_2d_full(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

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
static C& convolve_2d_same(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

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
static C& convolve_2d_valid(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

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
