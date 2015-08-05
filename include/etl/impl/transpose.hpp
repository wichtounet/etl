//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/value_fwd.hpp"
#include "etl/temporary.hpp"
#include "etl/config.hpp"
#include "etl/traits_lite.hpp"

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

template<typename C, typename Enable = void>
struct inplace_rectangular_transpose {
    template<typename CC>
    static void apply(CC&& mat){
        auto copy = force_temporary(mat);

        auto data = mat.memory_start();

        //Dimensions prior to transposition
        const auto N = etl::dim<0>(mat);
        const auto M = etl::dim<1>(mat);

        for(std::size_t i = 0; i < N; ++i){
            for(std::size_t j = 0; j < M; ++j){
                data[j * N + i] = copy(i, j);
            }
        }
    }

    //This implementation is really slow but has O(1) space
    template<typename CC>
    static void real_inplace(CC&& mat){
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
        mkl_simatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<1>(c), etl::dim<0>(c));
    }
};

template<typename C>
struct inplace_square_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_double_precision<C>::value>> {
    template<typename CC>
    static void apply(CC&& c){
        mkl_dimatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, c.memory_start(), etl::dim<1>(c), etl::dim<0>(c));
    }
};

template<typename C>
struct inplace_rectangular_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_single_precision<C>::value>> {
    template<typename CC>
    static void apply(CC&& c){
        auto copy = force_temporary(c);
        mkl_somatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, copy.memory_start(), etl::dim<1>(c), c.memory_start(), etl::dim<0>(c));
    }
};

template<typename C>
struct inplace_rectangular_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_double_precision<C>::value>> {
    template<typename CC>
    static void apply(CC&& c){
        auto copy = force_temporary(c);
        mkl_domatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, copy.memory_start(), etl::dim<1>(c), c.memory_start(), etl::dim<0>(c));
    }
};

#endif

} //end of namespace detail

} //end of namespace etl
