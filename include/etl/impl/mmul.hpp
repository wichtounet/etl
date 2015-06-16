//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

#include "../config.hpp"
#include "../traits_lite.hpp"

#include "std/mmul.hpp"
#include "std/strassen_mmul.hpp"
#include "blas/gemm.hpp"
#include "eblas/gemm.hpp"

namespace etl {

namespace detail {

template<typename A, typename B, typename C, typename Enable = void>
struct mm_mul_impl {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::standard::mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct is_fast_dgemm : cpp::and_c<cpp::not_c<is_cblas_enabled>, all_double_precision<A, B, C>, all_dma<A, B, C>> {};

template<typename A, typename B, typename C>
struct is_fast_sgemm : cpp::and_c<cpp::not_c<is_cblas_enabled>, all_single_precision<A, B, C>, all_dma<A, B, C>> {};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_fast_dgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::eblas::fast_dgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct mm_mul_impl<A, B, C, std::enable_if_t<is_fast_sgemm<A,B,C>::value>> {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::eblas::fast_sgemm(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }
};

template<typename A, typename B, typename C>
struct is_blas_dgemm : cpp::and_c<is_cblas_enabled, all_double_precision<A, B, C>, all_dma<A, B, C>> {};

template<typename A, typename B, typename C>
struct is_blas_sgemm : cpp::and_c<is_cblas_enabled, all_single_precision<A, B, C>, all_dma<A, B, C>> {};

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

template<typename A, typename B, typename C, typename Enable = void>
struct strassen_mm_mul_impl {
    static void apply(A&& a, B&& b, C&& c){
        etl::impl::standard::strassen_mm_mul(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
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

} //end of namespace detail

} //end of namespace etl
