//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file transpose.hpp
 * \brief Implementations of inplace matrix transposition
 *
 * Implementations of inplace matrix transposition.
 *    1. Simple implementation using for loop
 *    2. Implementations using MKL
 *
 * Square and rectangular implementation are separated.
 */

#pragma once

#include "etl/temporary.hpp"

#ifdef ETL_MKL_MODE
#include "mkl_trans.h"
#endif

namespace etl {

namespace detail {

template <typename C, typename Enable = void>
struct inplace_square_transpose {
    template <typename CC>
    static void apply(CC&& c) {
        using std::swap;

        const std::size_t N = etl::dim<0>(c);

        for (std::size_t i = 0; i < N - 1; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                swap(c(i, j), c(j, i));
            }
        }
    }
};

template <typename C, typename Enable = void>
struct inplace_rectangular_transpose {
    template <typename CC>
    static void apply(CC&& mat) {
        auto copy = force_temporary(mat);

        auto data = mat.memory_start();

        //Dimensions prior to transposition
        const std::size_t N = etl::dim<0>(mat);
        const std::size_t M = etl::dim<1>(mat);

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                data[j * N + i] = copy(i, j);
            }
        }
    }

    //This implementation is really slow but has O(1) space
    template <typename CC>
    static void real_inplace(CC&& mat) {
        using std::swap;

        const std::size_t N = etl::dim<0>(mat);
        const std::size_t M = etl::dim<1>(mat);

        auto data = mat.memory_start();

        for (std::size_t k = 0; k < N * M; k++) {
            auto idx = k;
            do {
                idx = (idx % N) * M + (idx / N);
            } while (idx < k);
            std::swap(data[k], data[idx]);
        }
    }
};

template <typename A, typename C, typename Enable = void>
struct transpose {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        auto mem_c = c.memory_start();
        auto mem_a = a.memory_start();

        // Delegate aliasing transpose to inplace algorithm
        if (mem_c == mem_a) {
            if (etl::dim<0>(a) == etl::dim<1>(a)) {
                inplace_square_transpose<C>::apply(c);
            } else {
                inplace_rectangular_transpose<C>::apply(c);
            }
        } else {
            if (decay_traits<A>::storage_order == order::RowMajor) {
                for (std::size_t i = 0; i < etl::dim<0>(a); ++i) {
                    for (std::size_t j = 0; j < etl::dim<1>(a); ++j) {
                        mem_c[j * etl::dim<1>(c) + i] = mem_a[i * etl::dim<1>(a) + j];
                    }
                }
            } else {
                for (std::size_t j = 0; j < etl::dim<1>(a); ++j) {
                    for (std::size_t i = 0; i < etl::dim<0>(a); ++i) {
                        mem_c[i * etl::dim<0>(c) + j] = mem_a[j * etl::dim<0>(a) + i];
                    }
                }
            }
        }
    }
};

#ifdef ETL_MKL_MODE

//Helpers for MKL

template <typename A, typename C, cpp_enable_if(all_single_precision<A, C>::value)>
void mkl_otrans(A&& a, C&& c){
    auto mem_c = c.memory_start();
    auto mem_a = a.memory_start();

    if (decay_traits<A>::storage_order == order::RowMajor) {
        mkl_somatcopy('R', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0f, mem_a, etl::dim<1>(a), mem_c, etl::dim<0>(a));
    } else {
        mkl_somatcopy('C', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0f, mem_a, etl::dim<0>(a), mem_c, etl::dim<1>(a));
    }
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A, C>::value)>
void mkl_otrans(A&& a, C&& c){
    auto mem_c = c.memory_start();
    auto mem_a = a.memory_start();

    if (decay_traits<A>::storage_order == order::RowMajor) {
        mkl_domatcopy('R', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0, mem_a, etl::dim<1>(a), mem_c, etl::dim<0>(a));
    } else {
        mkl_domatcopy('C', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0, mem_a, etl::dim<0>(a), mem_c, etl::dim<1>(a));
    }
}

template <typename C, cpp_enable_if(all_single_precision<C>::value)>
void mkl_itrans(C&& c){
    if (decay_traits<C>::storage_order == order::RowMajor) {
        mkl_simatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<1>(c), etl::dim<0>(c));
    } else {
        mkl_simatcopy('C', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<0>(c), etl::dim<1>(c));
    }
}

template <typename C, cpp_enable_if(all_double_precision<C>::value)>
void mkl_itrans(C&& c){
    if (decay_traits<C>::storage_order == order::RowMajor) {
        mkl_dimatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, c.memory_start(), etl::dim<1>(c), etl::dim<0>(c));
    } else {
        mkl_dimatcopy('C', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, c.memory_start(), etl::dim<0>(c), etl::dim<1>(c));
    }
}

template <typename C>
struct inplace_square_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_single_precision<C>::value>> {
    template <typename CC>
    static void apply(CC&& c) {
        mkl_itrans(c);
    }
};

template <typename C>
struct inplace_square_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_double_precision<C>::value>> {
    template <typename CC>
    static void apply(CC&& c) {
        mkl_itrans(c);
    }
};

template <typename C>
struct inplace_rectangular_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_single_precision<C>::value>> {
    template <typename CC>
    static void apply(CC&& c) {
        mkl_otrans(force_temporary(c), c);
    }
};

template <typename C>
struct inplace_rectangular_transpose<C, std::enable_if_t<has_direct_access<C>::value && is_double_precision<C>::value>> {
    template <typename CC>
    static void apply(CC&& c) {
        mkl_otrans(force_temporary(c), c);
    }
};

template <typename A, typename C>
struct transpose<A, C, std::enable_if_t<all_dma<A, C>::value && all_floating<A, C>::value>> {
    template <typename AA, typename CC>
    static void apply(AA&& a, CC&& c) {
        auto mem_c = c.memory_start();
        auto mem_a = a.memory_start();

        // Delegate aliasing transpose to inplace algorithm
        if (mem_c == mem_a) {
            if (etl::dim<0>(a) == etl::dim<1>(a)) {
                mkl_itrans(c);
            } else {
                mkl_otrans(force_temporary(c), c);
            }
        } else {
            mkl_otrans(a, c);
        }
    }
};

#endif

} //end of namespace detail

} //end of namespace etl
