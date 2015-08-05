//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

#include "etl/config.hpp"
#include "etl/traits_lite.hpp"

#include "etl/impl/std/fft.hpp"
#include "etl/impl/blas/fft.hpp"

namespace etl {

namespace detail {

template<typename A, typename C, typename Enable = void>
struct fft1_impl {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::standard::fft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C, typename Enable = void>
struct fft2_impl {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::standard::fft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C, typename Enable = void>
struct ifft1_impl {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::standard::ifft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C, typename Enable = void>
struct ifft1_real_impl {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::standard::ifft1_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C, typename Enable = void>
struct ifft2_impl {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::standard::ifft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C, typename Enable = void>
struct ifft2_real_impl {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::standard::ifft2_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C, typename Enable = void>
struct fft_conv1_full_impl {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::standard::fft1_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C, typename Enable = void>
struct fft_conv2_full_impl {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::standard::fft2_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct is_blas_dfft : cpp::and_c<is_mkl_enabled, is_double_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_blas_sfft : cpp::and_c<is_mkl_enabled, is_single_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_blas_cfft : cpp::and_c<is_mkl_enabled, is_complex_single_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_blas_zfft : cpp::and_c<is_mkl_enabled, is_complex_double_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_blas_dfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::dfft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_blas_sfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::sfft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::cfft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::zfft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft1_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::cifft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft1_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::zifft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft1_real_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::cifft1_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft1_real_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::zifft1_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C>
struct is_blas_sfft_convolve : cpp::and_c<is_mkl_enabled, all_single_precision<A,B,C>, all_dma<A, B, C>> {};

template<typename A, typename B, typename C>
struct is_blas_dfft_convolve : cpp::and_c<is_mkl_enabled, all_double_precision<A,B,C>, all_dma<A, B, C>> {};

template<typename A, typename B, typename C>
struct fft_conv1_full_impl<A, B, C, std::enable_if_t<is_blas_sfft_convolve<A,B,C>::value>> {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::blas::sfft1_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C>
struct fft_conv1_full_impl<A, B, C, std::enable_if_t<is_blas_dfft_convolve<A,B,C>::value>> {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::blas::dfft1_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_blas_dfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::dfft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_blas_sfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::sfft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::cfft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::zfft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft2_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::cifft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft2_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::zifft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft2_real_impl<A, C, std::enable_if_t<is_blas_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::cifft2_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft2_real_impl<A, C, std::enable_if_t<is_blas_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::blas::zifft2_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C>
struct fft_conv2_full_impl<A, B, C, std::enable_if_t<is_blas_sfft_convolve<A,B,C>::value>> {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::blas::sfft2_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C>
struct fft_conv2_full_impl<A, B, C, std::enable_if_t<is_blas_dfft_convolve<A,B,C>::value>> {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::blas::dfft2_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

} //end of namespace detail

} //end of namespace etl
