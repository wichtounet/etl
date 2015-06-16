#pragma once
//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "../config.hpp"
#include "../traits_lite.hpp"

/**
 * Implementations of the outer product:
 *    1. Simple implementation using for loop
 *    2. Implementations using BLAS SGET and DGER
 */

namespace etl {

namespace detail {

template<typename A, typename B, typename C, typename Enable = void>
struct outer_product_impl {
    static void apply(const A& a, const B& b, C&& c){
        for(std::size_t i = 0; i < etl::dim<0>(c); ++i){
            for(std::size_t j = 0; j < etl::dim<1>(c); ++j){
                c(i,j) = a(i) * b(j);
            }
        }
    }
};

#ifdef ETL_BLAS_MODE

template<typename A, typename B, typename C>
struct outer_product_impl <A, B, C, std::enable_if_t<all_single_precision<A, B, C>::value && all_dma<A, B, C>::value>> {
    static void apply(const A& a, const B& b, C&& c){
        c = 0;

        cblas_sger(
            CblasRowMajor,
            etl::dim<0>(a), etl::dim<0>(b),
            1.0,
            a.memory_start(), 1,
            b.memory_start(), 1,
            c.memory_start(), etl::dim<0>(b)
        );
    }
};

template<typename A, typename B, typename C>
struct outer_product_impl <A, B, C, std::enable_if_t<all_double_precision<A, B, C>::value && all_dma<A, B, C>::value>> {
    static void apply(const A& a, const B& b, C&& c){
        c = 0;

        cblas_dger(
            CblasRowMajor,
            etl::dim<0>(a), etl::dim<0>(b),
            1.0,
            a.memory_start(), 1,
            b.memory_start(), 1,
            c.memory_start(), etl::dim<0>(b)
        );
    }
};

#endif

} //end of namespace detail

} //end of namespace etl
