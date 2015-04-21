//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_IMPL_TRANSPOSE_HPP
#define ETL_IMPL_TRANSPOSE_HPP

#include "../config.hpp"
#include "../traits_lite.hpp"

#ifdef ETL_MKL_MODE
#include "mkl_trans.h"
#endif

/**
 * Implementations of matrix transposition.
 *    1. Simple implementation using for loop
 *    2. Implementations using MKL
 *
 * Square and rectangular implementation are separated. 
 */

namespace etl {

namespace detail {

template<typename C, typename Enable = void>
struct inplace_square_transpose {
    template<typename CC>
    static void apply(CC&& c){
        using std::swap;

        const auto N = etl::dim<0>(c);

        for(std::size_t i = 0; i < N - 1; ++i){
            for(std::size_t j = i + 1; j < N; ++j){
                swap(c(i, j), c(j, i));
            }
        }
    }
};

//TODO This implementation is really too slow

template<typename C, typename Enable = void>
struct inplace_rectangular_transpose {
    template<typename CC>
    static void apply(CC&& mat){
        using std::swap;

        const auto N = etl::dim<0>(mat);
        const auto M = etl::dim<1>(mat);

        auto data = mat.memory_start();

        for(std::size_t k = 0; k < N*M; k++) {
            auto idx = k;
            do {
                idx = (idx % N) * M + (idx / N);
            } while(idx < k);
            std::swap(data[k], data[idx]);
        }
    }
};

#ifdef ETL_MKL_MODE

template<typename C>
struct inplace_square_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_single_precision<C>::value>> {
    template<typename CC>
    static void apply(CC&& c){
        mkl_simatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<1>(c), etl::dim<1>(c));
    }
};

template<typename C>
struct inplace_square_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_double_precision<C>::value>> {
    template<typename CC>
    static void apply(CC&& c){
        mkl_dimatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<1>(c), etl::dim<1>(c));
    }
};

template<typename C>
struct inplace_rectangular_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_single_precision<C>::value>> {
    template<typename CC>
    static void apply(CC&& c){
        mkl_simatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<1>(c), etl::dim<1>(c));
    }
};

template<typename C>
struct inplace_rectangular_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_double_precision<C>::value>> {
    template<typename CC>
    static void apply(CC&& c){
        mkl_dimatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<1>(c), etl::dim<1>(c));
    }
};

#endif

} //end of namespace detail

} //end of namespace etl

#endif
