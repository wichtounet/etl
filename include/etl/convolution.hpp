//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_CONVOLUTION_HPP
#define ETL_CONVOLUTION_HPP

#include <algorithm>

#include "assert.hpp"
#include "tmp.hpp"

namespace etl {

template<typename I, typename K, typename C>
static void convolve_1d_full(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    static_assert(C::etl_size == I::etl_size + K::etl_size - 1, "Invalid output vector size");

    for(std::size_t i = 0; i < C::etl_size; ++i) {
        const auto lo = i >= K::etl_size - 1 ? i - (K::etl_size - 1) : 0;
        const auto hi = i <  I::etl_size - 1 ? i                     : I::etl_size - 1;

        double temp = 0.0;

        for(std::size_t j = lo; j <= hi; ++j) {
            temp += input[j] * kernel[i - j];
        }

        conv[i] = temp;
    }
}

template<typename I, typename K, typename C>
static void convolve_1d_same(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    static_assert(C::etl_size == I::etl_size, "Invalid output vector size");

    for(std::size_t j = 0 ; j < C::etl_size ; ++j){
        int l_lo = std::max<int>(0, j - (K::etl_size - 1) / 2);
        int l_hi = std::min<int>(I::etl_size- 1, j + K::etl_size / 2);

        double temp = 0.0;

        for(std::size_t l = l_lo ; l <= static_cast<std::size_t>(l_hi); ++l){
            temp += input(l) * kernel(j - l + K::etl_size / 2);
        }

        conv(0 + j) = temp;
    }
}

template<typename I, typename K, typename C>
static void convolve_1d_valid(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");
    static_assert(C::etl_size == I::etl_size - K::etl_size + 1, "Invalid output vector size");

    for(std::size_t j = 0 ; j < C::etl_size ; ++j){
        double temp = 0.0;

        for(std::size_t l = j ; l <= j + K::etl_size - 1; ++l){
            temp += input[l] * kernel[j + K::etl_size - 1 - l];
        }

        conv[j] = temp;
    }
}

template<typename I, typename K, typename C>
static void convolve_2d_full(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    for(std::size_t i = 0 ; i < C::rows ; ++i){
        auto k_lo = std::max<int>(0, i - K::rows + 1);
        auto k_hi = std::min(I::rows - 1, i);

        for(std::size_t j = 0 ; j < C::columns ; ++j){
            auto l_lo = std::max<int>(0, j - K::columns + 1);
            auto l_hi = std::min(I::columns - 1 ,j);

            double temp = 0.0;

            for(std::size_t k = k_lo ; k <= k_hi ; ++k){
                for(std::size_t l = l_lo ; l <= l_hi ; ++l){
                    temp += input(k,l) * kernel(i - k, j - l);
                }
            }

            conv(i, j) = temp;
        }
    }
}

template<typename I, typename K, typename C>
static void convolve_2d_same(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    for(std::size_t i = 0 ; i < C::rows; ++i){
        auto k_lo = std::max<int>(0, i - (K::rows-1)/2);
        auto k_hi = std::min<int>(I::rows - 1, i + K::rows/2);

        for(std::size_t j = 0 ; j < C::columns; ++j){
            auto l_lo = std::max<int>(0, j - (K::columns-1)/2);
            auto l_hi = std::min<int>(I::columns - 1, j + K::columns/2);

            double temp = 0.0;

            for(int k = k_lo ; k <= k_hi ; ++k){
                for(std::size_t l = l_lo ; l <= static_cast<std::size_t>(l_hi); ++l){
                    temp += input(k, l) * kernel(i-k+K::rows/2, j-l+K::columns/2);
                }
            }

            conv(i, j) = temp;
        }
    }
}

template<typename I, typename K, typename C>
static void convolve_2d_valid(const I& input, const K& kernel, C& conv){
    static_assert(is_etl_expr<I>::value && is_etl_expr<K>::value && is_etl_expr<C>::value, "Convolution only supported for ETL expressions");

    for(std::size_t i = 0 ; i < C::rows ; ++i){
        for(std::size_t j = 0 ; j < C::columns ; ++j){
            double temp = 0.0;

            for(std::size_t k = i ; k <= i + K::rows-1; ++k){
                for(std::size_t l = j ; l <= j + K::columns-1 ; ++l){
                    temp += input(k,l) * kernel((i+K::rows-1-k), (j+K::columns-1-l));
                }
            }

            conv(i,j) = temp;
        }
    }
}

} //end of namespace etl

#endif
