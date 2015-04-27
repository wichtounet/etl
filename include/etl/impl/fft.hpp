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
struct fft1_impl;

template<typename A, typename C, typename Enable = void>
struct fft2_impl;

template<typename A, typename C, typename Enable = void>
struct ifft1_impl;

template<typename A, typename C, typename Enable = void>
struct ifft1_real_impl;

template<typename A, typename C, typename Enable = void>
struct ifft2_impl;

template<typename A, typename B, typename C, typename Enable = void>
struct fft_conv1_full_impl;

template<typename A, typename C>
struct is_blas_dfft : cpp::and_c<is_mkl_enabled, is_double_precision<A>, is_dma_2<A, C>> {};

template<typename A, typename C>
struct is_blas_sfft : cpp::and_c<is_mkl_enabled, is_single_precision<A>, is_dma_2<A, C>> {};

template<typename A, typename C>
struct is_blas_cfft : cpp::and_c<is_mkl_enabled, is_complex_single_precision<A>, is_dma_2<A, C>> {};

template<typename A, typename C>
struct is_blas_zfft : cpp::and_c<is_mkl_enabled, is_complex_double_precision<A>, is_dma_2<A, C>> {};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_blas_dfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::dfft1(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_blas_sfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::sfft1(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::cfft1(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::zfft1(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct ifft1_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::cifft1(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct ifft1_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::zifft1(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct ifft1_real_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::cifft1_real(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct ifft1_real_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::zifft1_real(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct is_blas_sfft_convolve : cpp::and_c<is_mkl_enabled, is_single_precision_3<A,B,C>, is_dma_3<A, B, C>> {};

template<typename A, typename B, typename C>
struct is_blas_dfft_convolve : cpp::and_c<is_mkl_enabled, is_double_precision_3<A,B,C>, is_dma_3<A, B, C>> {};

template<typename A, typename B, typename C>
struct fft_conv1_full_impl<A, B, C, std::enable_if_t<is_blas_sfft_convolve<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::sfft1_convolve(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct fft_conv1_full_impl<A, B, C, std::enable_if_t<is_blas_dfft_convolve<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::dfft1_convolve(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_blas_dfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::dfft2(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_blas_sfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::sfft2(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::cfft2(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::zfft2(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct ifft2_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::cifft2(std::forward<A>(a), std::forward<C>(c));
    }
};

template<typename A, typename C>
struct ifft2_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    static void apply(A&& a, C&& c){
        etl::impl::blas::zifft2(std::forward<A>(a), std::forward<C>(c));
    }
};

} //end of namespace detail

} //end of namespace etl

#endif
