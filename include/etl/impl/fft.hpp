//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_IMPL_FFT_HPP
#define ETL_IMPL_FFT_HPP

#include <algorithm>

#include "../config.hpp"
#include "../traits_lite.hpp"

#include "blas/fft.hpp"

namespace etl {

namespace detail {

template<typename A, typename C, typename Enable = void>
struct fft_impl;

template<typename A, typename C>
struct is_blas_dfft : cpp::bool_constant_c<cpp::and_c<is_mkl_enabled, is_double_precision<A>, is_dma_2<A, C>>> {};

template<typename A, typename C>
struct is_blas_sfft : cpp::bool_constant_c<cpp::and_c<is_mkl_enabled, is_single_precision<A>, is_dma_2<A, C>>> {};

template<typename A, typename C>
struct fft_impl<A, C, std::enable_if_t<is_blas_dfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::dfft(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct fft_impl<A, C, std::enable_if_t<is_blas_sfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::sfft(std::forward<A>(a), std::forward<C>(c));
    }
};

} //end of namespace detail

} //end of namespace etl

#endif
