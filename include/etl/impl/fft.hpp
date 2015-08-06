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
#include "etl/impl/cufft/fft.hpp"

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
struct is_blas_dfft : cpp::and_c<is_mkl_enabled, cpp::not_c<is_cufft_enabled>, is_double_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_blas_sfft : cpp::and_c<is_mkl_enabled, cpp::not_c<is_cufft_enabled>, is_single_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_blas_cfft : cpp::and_c<is_mkl_enabled, cpp::not_c<is_cufft_enabled>, is_complex_single_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_blas_zfft : cpp::and_c<is_mkl_enabled, cpp::not_c<is_cufft_enabled>, is_complex_double_precision<A>, all_dma<A, C>> {};

inline bool is_power_of_two(long n){
    return (n & (n - 1)) == 0;
}

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
struct is_blas_sfft_convolve : cpp::and_c<is_mkl_enabled, cpp::not_c<is_cufft_enabled>, all_single_precision<A,B,C>, all_dma<A, B, C>> {};

template<typename A, typename B, typename C>
struct is_blas_dfft_convolve : cpp::and_c<is_mkl_enabled, cpp::not_c<is_cufft_enabled>, all_double_precision<A,B,C>, all_dma<A, B, C>> {};

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

template<typename A, typename C>
struct is_cufft_dfft : cpp::and_c<is_cufft_enabled, is_double_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_cufft_sfft : cpp::and_c<is_cufft_enabled, is_single_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_cufft_cfft : cpp::and_c<is_cufft_enabled, is_complex_single_precision<A>, all_dma<A, C>> {};

template<typename A, typename C>
struct is_cufft_zfft : cpp::and_c<is_cufft_enabled, is_complex_double_precision<A>, all_dma<A, C>> {};

enum class fft_impl {
    STD,
    MKL,
    CUFFT
};

enum class precision {
    S,
    D,
    C,
    Z
};

template<bool DMA>
inline fft_impl select_fft1_impl(const std::size_t n){
    //Only std implementation is able to handle non-dma expressions
    if(!DMA){
        return fft_impl::STD;
    }

    //Note since these boolean will be known at compile time, the conditions will be a lot simplified
    static constexpr const bool mkl = is_mkl_enabled::value;
    static constexpr const bool cufft = is_cufft_enabled::value;

    if(cufft){
        if(is_power_of_two(n)){
            if(n <= 64){
                return fft_impl::STD;
            } else if(n <= 1024){
                if(mkl){
                    return fft_impl::MKL;
                } else {
                    return fft_impl::STD;
                }
            } else if(n <= 65536 && mkl){
                return fft_impl::MKL;
            }

            return fft_impl::CUFFT;
        }

        if(n <= 250000 && mkl){
            return fft_impl::MKL;
        }

        return fft_impl::CUFFT;
    } else if(mkl) {
        if(is_power_of_two(n) && n <= 64){
            return fft_impl::STD;
        }

        return fft_impl::MKL;
    } else {
        return fft_impl::STD;
    }
}

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_cufft_sfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        auto impl = select_fft1_impl<all_dma<A,C>::value>(etl::size(c));

        if(impl == fft_impl::STD){
            etl::impl::standard::fft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if(impl == fft_impl::MKL){
            etl::impl::blas::sfft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if(impl == fft_impl::CUFFT){
            etl::impl::cufft::sfft1(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_cufft_dfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        auto impl = select_fft1_impl<all_dma<A,C>::value>(etl::size(c));

        if(impl == fft_impl::STD){
            etl::impl::standard::fft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if(impl == fft_impl::MKL){
            etl::impl::blas::dfft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if(impl == fft_impl::CUFFT){
            etl::impl::cufft::dfft1(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_cufft_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        auto impl = select_fft1_impl<all_dma<A,C>::value>(etl::size(c));

        if(impl == fft_impl::STD){
            etl::impl::standard::fft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if(impl == fft_impl::MKL){
            etl::impl::blas::cfft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if(impl == fft_impl::CUFFT){
            etl::impl::cufft::cfft1(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template<typename A, typename C>
struct fft1_impl<A, C, std::enable_if_t<is_cufft_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        auto impl = select_fft1_impl<all_dma<A,C>::value>(etl::size(c));

        if(impl == fft_impl::STD){
            etl::impl::standard::fft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if(impl == fft_impl::MKL){
            etl::impl::blas::zfft1(std::forward<AA>(a), std::forward<CC>(c));
        } else if(impl == fft_impl::CUFFT){
            etl::impl::cufft::zfft1(std::forward<AA>(a), std::forward<CC>(c));
        }
    }
};

template<typename A, typename C>
struct ifft1_impl<A, C, std::enable_if_t<is_cufft_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::cifft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft1_impl<A, C, std::enable_if_t<is_cufft_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::zifft1(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft1_real_impl<A, C, std::enable_if_t<is_cufft_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::cifft1_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft1_real_impl<A, C, std::enable_if_t<is_cufft_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::zifft1_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C>
struct is_cufft_sfft_convolve : cpp::and_c<is_cufft_enabled, all_single_precision<A,B,C>, all_dma<A, B, C>> {};

template<typename A, typename B, typename C>
struct is_cufft_dfft_convolve : cpp::and_c<is_cufft_enabled, all_double_precision<A,B,C>, all_dma<A, B, C>> {};

template<typename A, typename B, typename C>
struct fft_conv1_full_impl<A, B, C, std::enable_if_t<is_cufft_sfft_convolve<A,B,C>::value>> {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::cufft::sfft1_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C>
struct fft_conv1_full_impl<A, B, C, std::enable_if_t<is_cufft_dfft_convolve<A,B,C>::value>> {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::cufft::dfft1_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_cufft_dfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::dfft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_cufft_sfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::sfft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_cufft_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::cfft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct fft2_impl<A, C, std::enable_if_t<is_cufft_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::zfft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft2_impl<A, C, std::enable_if_t<is_cufft_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::cifft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft2_impl<A, C, std::enable_if_t<is_cufft_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::zifft2(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft2_real_impl<A, C, std::enable_if_t<is_cufft_cfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::cifft2_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename C>
struct ifft2_real_impl<A, C, std::enable_if_t<is_cufft_zfft<A,C>::value>> {
    template<typename AA, typename CC>
    static void apply(AA&& a, CC&& c){
        etl::impl::cufft::zifft2_real(std::forward<AA>(a), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C>
struct fft_conv2_full_impl<A, B, C, std::enable_if_t<is_cufft_sfft_convolve<A,B,C>::value>> {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::cufft::sfft2_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

template<typename A, typename B, typename C>
struct fft_conv2_full_impl<A, B, C, std::enable_if_t<is_cufft_dfft_convolve<A,B,C>::value>> {
    template<typename AA, typename BB, typename CC>
    static void apply(AA&& a, BB&& b, CC&& c){
        etl::impl::cufft::dfft2_convolve(std::forward<AA>(a), std::forward<BB>(b), std::forward<CC>(c));
    }
};

} //end of namespace detail

} //end of namespace etl
