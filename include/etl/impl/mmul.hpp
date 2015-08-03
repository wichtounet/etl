//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

#include "../config.hpp"
#include "../threshold.hpp"
#include "../traits_lite.hpp"

#include "std/mmul.hpp"
#include "std/strassen_mmul.hpp"
#include "blas/gemm.hpp"
#include "eblas/gemm.hpp"
#include "cublas/gemm.hpp"

namespace etl {

namespace detail {

template<typename A, typename B, typename C, typename Enable = void>
struct mm_mul_impl {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
using is_fast_dgemm = cpp::and_c<cpp::not_c<is_cblas_enabled>, cpp::not_c<is_cublas_enabled>, all_double_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
using is_fast_sgemm = cpp::and_c<cpp::not_c<is_cblas_enabled>, cpp::not_c<is_cublas_enabled>, all_single_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_fast_sgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        if(etl::size(c) < sgemm_eblas_min){
            etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else {
            etl::impl::eblas::fast_sgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_fast_dgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        if(etl::size(c) < dgemm_eblas_min){
            etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        } else {
            etl::impl::eblas::fast_dgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template<typename A, typename B, typename C>
using is_blas_dgemm = cpp::and_c<is_cblas_enabled, cpp::not_c<is_cublas_enabled>, all_double_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
using is_blas_sgemm = cpp::and_c<is_cblas_enabled, cpp::not_c<is_cublas_enabled>, all_single_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
using is_blas_cgemm = cpp::and_c<is_cblas_enabled, cpp::not_c<is_cublas_enabled>, all_complex_single_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
using is_blas_zgemm = cpp::and_c<is_cblas_enabled, cpp::not_c<is_cublas_enabled>, all_complex_double_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_blas_dgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::dgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_blas_sgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::sgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_blas_cgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::cgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_blas_zgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::zgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C, typename Enable = void>
struct vm_mul_impl {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::standard::vm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct vm_mul_impl<A, B, C, std::enable_if_t<is_blas_dgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::dgevm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct vm_mul_impl<A, B, C, std::enable_if_t<is_blas_sgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::sgevm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct vm_mul_impl<A, B, C, std::enable_if_t<is_blas_cgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::cgevm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct vm_mul_impl<A, B, C, std::enable_if_t<is_blas_zgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::zgevm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C, typename Enable = void>
struct mv_mul_impl {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::standard::mv_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct mv_mul_impl<A, B, C, std::enable_if_t<is_blas_dgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::dgemv(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct mv_mul_impl<A, B, C, std::enable_if_t<is_blas_sgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::sgemv(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct mv_mul_impl<A, B, C, std::enable_if_t<is_blas_cgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::cgemv(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct mv_mul_impl<A, B, C, std::enable_if_t<is_blas_zgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::blas::zgemv(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
using is_cublas_dgemm = cpp::and_c<is_cublas_enabled, all_double_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
using is_cublas_sgemm = cpp::and_c<is_cublas_enabled, all_single_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
using is_cublas_cgemm = cpp::and_c<is_cublas_enabled, all_complex_single_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
using is_cublas_zgemm = cpp::and_c<is_cublas_enabled, all_complex_double_precision<A, B, C>, all_dma<A, B, C>>;

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_cublas_sgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        if(etl::size(c) < sgemm_cublas_min){
            if(is_cblas_enabled::value){
                etl::impl::blas::sgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
            } else {
                if(etl::size(c) < sgemm_std_max){
                    etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
                } else {
                    etl::impl::cublas::sgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
                }
            }
        } else {
            etl::impl::cublas::sgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_cublas_dgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        if(etl::size(c) < dgemm_cublas_min){
            if(is_cblas_enabled::value){
                etl::impl::blas::dgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
            } else {
                if(etl::size(c) < dgemm_std_max){
                    etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
                } else {
                    etl::impl::cublas::dgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
                }
            }
        } else {
            etl::impl::cublas::dgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_cublas_cgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        if(etl::size(c) < cgemm_cublas_min){
            if(is_cblas_enabled::value){
                etl::impl::blas::cgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
            } else {
                if(etl::size(c) < cgemm_std_max){
                    etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
                } else {
                    etl::impl::cublas::cgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
                }
            }
        } else {
            etl::impl::cublas::cgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_cublas_zgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        if(etl::size(c) < zgemm_cublas_min){
            if(is_cblas_enabled::value){
                etl::impl::blas::zgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
            } else {
                if(etl::size(c) < zgemm_std_max){
                    etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
                } else {
                    etl::impl::cublas::zgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
                }
            }
        } else {
            etl::impl::cublas::cgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
        }
    }
};

template<typename A, typename B, typename C, typename Enable = void>
struct strassen_mm_mul_impl {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::standard::strassen_mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

} //end of namespace detail

} //end of namespace etl
