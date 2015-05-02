//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_IMPL_SCALAR_OP_HPP
#define ETL_IMPL_SCALAR_OP_HPP

#include "../config.hpp"

/**
 * Implementations of vector/matrix scalar operations.
 */

namespace etl {

namespace detail {

template<typename T, typename Enable = void>
struct scalar_add {
    template<typename TT>
    static void apply(TT&& lhs, value_t<TT> rhs){
        auto m = lhs.memory_start();

        for(std::size_t i = 0; i < size(lhs); ++i){
            m[i] += rhs;
        }
    }
};

template<typename T, typename Enable = void>
struct scalar_sub {
    template<typename TT>
    static void apply(TT&& lhs, value_t<TT> rhs){
        auto m = lhs.memory_start();

        for(std::size_t i = 0; i < size(lhs); ++i){
            m[i] -= rhs;
        }
    }
};

template<typename T, typename Enable = void>
struct scalar_mul {
    template<typename TT>
    static void apply(TT&& lhs, value_t<TT> rhs){
        auto m = lhs.memory_start();

        for(std::size_t i = 0; i < size(lhs); ++i){
            m[i] *= rhs;
        }
    }
};

template<typename T, typename Enable = void>
struct scalar_div {
    template<typename TT>
    static void apply(TT&& lhs, value_t<TT> rhs){
        auto m = lhs.memory_start();

        for(std::size_t i = 0; i < size(lhs); ++i){
            m[i] /= rhs;
        }
    }
};

template<typename T, typename Enable = void>
struct scalar_mod {
    template<typename TT>
    static void apply(TT&& lhs, value_t<TT> rhs){
        auto m = lhs.memory_start();

        for(std::size_t i = 0; i < size(lhs); ++i){
            m[i] %= rhs;
        }
    }
};

#ifdef ETL_BLAS_MODE

template<typename T>
struct scalar_mul<T, std::enable_if_t<is_single_precision<T>::value && has_direct_access<T>::value>> {
    template<typename TT>
    static void apply(TT&& lhs, value_t<TT> rhs){
        cblas_sscal(size(lhs), rhs, lhs.memory_start(), 1);
    }
};

template<typename T>
struct scalar_mul<T, std::enable_if_t<is_double_precision<T>::value && has_direct_access<T>::value>> {
    template<typename TT>
    static void apply(TT&& lhs, value_t<TT> rhs){
        cblas_dscal(size(lhs), rhs, lhs.memory_start(), 1);
    }
};

#endif

} //end of namespace detail

} //end of namespace etl

#endif
