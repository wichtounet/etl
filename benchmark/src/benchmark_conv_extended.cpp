//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"
#include "benchmark_conv.hpp"

#ifdef ETL_EXTENDED_BENCH
#define sdm_t1 etl::dyn_vector
#define fdm_t1 etl::fast_dyn_matrix

// Version with some padding (image and kernel are not multiple of 8)
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid_multi_2 [conv][conv2]", conv_2d_multi_policy_pad,
    FLOPS([](size_t d1, size_t d2, size_t d3){ return 2 * d1 * d1 * d2 * d2 * d3; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2, size_t d3){ return std::make_tuple(smat(d1,d1), smat3(d3,d2,d2), smat3(d3,d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat3& b, smat3& r){ r = etl::conv_2d_valid_multi(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::STD, etl::conv_2d_valid_multi(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::VEC, etl::conv_2d_valid_multi(a, b)); })
    MKL_SECTION_FUNCTOR("fft", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::VALID_FFT_MKL, etl::conv_2d_valid_multi(a, b)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::BLAS_VEC, etl::conv_2d_valid_multi(a, b)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::BLAS_MKL, etl::conv_2d_valid_multi(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::CUDNN, etl::conv_2d_valid_multi(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv1_valid_dyn_1 [conv][conv1]", fast_policy,
    FLOPS([](size_t){ return 2 * 10000 * 5000; }),
    CPM_SECTION_INIT([](size_t){ return std::make_tuple(sdm_t1<float>(10000), sdm_t1<float>(5000), sdm_t1<float>(5001)); }),
    CPM_SECTION_FUNCTOR("default", [](auto& a, auto& b, auto& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](auto& a, auto& b, auto& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_valid(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv1_valid_fast_1 [conv][conv1]", fast_policy,
    FLOPS([](size_t){ return 2 * 10000 * 5000; }),
    CPM_SECTION_INIT([](size_t){ return std::make_tuple(fdm_t1<float,10000>(), fdm_t1<float,5000>(), fdm_t1<float,5001>()); }),
    CPM_SECTION_FUNCTOR("default", [](auto& a, auto& b, auto& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](auto& a, auto& b, auto& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_valid(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv1_same [conv][conv1]", conv_1d_large_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_same(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::VEC, etl::conv_1d_same(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv1_same [conv][conv1]", conv_1d_large_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_same(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::VEC, etl::conv_1d_same(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_same [conv][conv2]", conv_2d_large_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_same(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::VEC, etl::conv_2d_same(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_same [conv][conv2]", conv_2d_large_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_same(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::VEC, etl::conv_2d_same(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_full [conv][conv2]", conv_2d_large_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_full(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::VEC, etl::conv_2d_full(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_full(a, b)); })
    ,CPM_SECTION_FUNCTOR("fft_std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::FFT_STD, etl::conv_2d_full(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_2d_full(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_2d_full(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid_multi_flipped [conv][conv2]", conv_2d_multi_policy,
    FLOPS([](size_t d1, size_t d2, size_t d3){ return 2 * d1 * d1 * d2 * d2 * d3; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2, size_t d3){ return std::make_tuple(smat(d1,d1), smat3(d3,d2,d2), smat3(d3,d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat3& b, smat3& r){ r = etl::conv_2d_valid_multi_flipped(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::STD, etl::conv_2d_valid_multi_flipped(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::VEC, etl::conv_2d_valid_multi_flipped(a, b)); })
    MKL_SECTION_FUNCTOR("fft", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::VALID_FFT_MKL, etl::conv_2d_valid_multi_flipped(a, b)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::BLAS_VEC, etl::conv_2d_valid_multi_flipped(a, b)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::BLAS_MKL, etl::conv_2d_valid_multi_flipped(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::CUDNN, etl::conv_2d_valid_multi_flipped(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid_multi_multi_flipped [conv][conv2]", conv_2d_multi_multi_policy,
    FLOPS([](size_t d1, size_t d2, size_t d3, size_t d4){ return 2 * d1 * d1 * d2 * d2 * d3 * d4; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2, size_t d3, size_t d4){
        return std::make_tuple(smat3(d3, d1, d1), smat3(d4, d2, d2), smat4(d4, d3, d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat3& a, smat3& b, smat4& r){ r = etl::conv_2d_valid_multi_multi_flipped(a, b); })
    VEC_SECTION_FUNCTOR("vec", [](smat3& a, smat3& b, smat4& r){ r = selected_helper(etl::conv_multi_impl::VEC, etl::conv_2d_valid_multi_multi_flipped(a, b)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat3& a, smat3& b, smat4& r){ r = selected_helper(etl::conv_multi_impl::BLAS_VEC, etl::conv_2d_valid_multi_multi_flipped(a, b)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat3& a, smat3& b, smat4& r){ r = selected_helper(etl::conv_multi_impl::BLAS_MKL, etl::conv_2d_valid_multi_multi_flipped(a, b)); })
    MKL_SECTION_FUNCTOR("fft", [](smat3& a, smat3& b, smat4& r){ r = selected_helper(etl::conv_multi_impl::VALID_FFT_MKL, etl::conv_2d_valid_multi_multi_flipped(a, b)); })
)

// Note: STD is way too slow to benchmark

CONV4_BENCH("sconv4_valid_1 [conv][conv4][conv4_chain]", conv_4d_valid_policy_1, conv_4d_valid)
CONV4_BENCH("sconv4_valid_2 [conv][conv4][conv4_chain]", conv_4d_valid_policy_2, conv_4d_valid)
CONV4_BENCH("sconv4_valid_3 [conv][conv4][conv4_chain]", conv_4d_valid_policy_3, conv_4d_valid)
CONV4_BENCH("sconv4_valid_4 [conv][conv4][conv4_chain]", conv_4d_valid_policy_4, conv_4d_valid)
CONV4_BENCH("sconv4_valid_5 [conv][conv4][conv4_chain]", conv_4d_valid_policy_5, conv_4d_valid)
CONV4_BENCH("sconv4_valid_6 [conv][conv4][conv4_chain]", conv_4d_valid_policy_6, conv_4d_valid)
CONV4_BENCH("sconv4_valid_7 [conv][conv4][conv4_chain]", conv_4d_valid_policy_7, conv_4d_valid)
CONV4_BENCH("sconv4_valid_8 [conv][conv4][conv4_chain]", conv_4d_valid_policy_8, conv_4d_valid)
CONV4_BENCH("sconv4_valid_9 [conv][conv4][conv4_chain]", conv_4d_valid_policy_9, conv_4d_valid)
CONV4_BENCH("sconv4_valid_10 [conv][conv4][conv4_chain]", conv_4d_valid_policy_10, conv_4d_valid)
CONV4_BENCH("sconv4_valid_11 [conv][conv4][conv4_chain]", conv_4d_valid_policy_11, conv_4d_valid)
CONV4_BENCH("sconv4_valid_12 [conv][conv4][conv4_chain]", conv_4d_valid_policy_12, conv_4d_valid)
CONV4_BENCH("sconv4_valid_13 [conv][conv4][conv4_chain]", conv_4d_valid_policy_13, conv_4d_valid)
CONV4_BENCH("sconv4_valid_14 [conv][conv4][conv4_chain]", conv_4d_valid_policy_14, conv_4d_valid)
CONV4_BENCH("sconv4_valid_15 [conv][conv4][conv4_chain]", conv_4d_valid_policy_15, conv_4d_valid)
CONV4_BENCH("sconv4_valid_16 [conv][conv4][conv4_chain]", conv_4d_valid_policy_16, conv_4d_valid)
CONV4_BENCH("sconv4_valid_17 [conv][conv4][conv4_chain]", conv_4d_valid_policy_17, conv_4d_valid)
CONV4_BENCH("sconv4_valid_18 [conv][conv4][conv4_chain]", conv_4d_valid_policy_18, conv_4d_valid)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid_same [conv][conv4]", conv_4d_valid_policy_3,
    FLOPS([](size_t n, size_t k, size_t c, size_t i, size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t k, size_t c, size_t i, size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(k, c, w, w), smat4(n, k, i, i)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid(a, b, 1, 1, 1, 1); }),
    CPM_SECTION_FUNCTOR("std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::STD, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid_same_large [conv][conv4]", conv_4d_valid_policy_18,
    FLOPS([](size_t n, size_t k, size_t c, size_t i, size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t k, size_t c, size_t i, size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(k, c, w, w), smat4(n, k, i, i)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid(a, b, 1, 1, 1, 1); }),
    CPM_SECTION_FUNCTOR("std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::STD, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid(a, b, 1, 1, 1, 1)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid_back_same [conv][conv4]", conv_4d_valid_policy_3,
    FLOPS([](size_t n, size_t c, size_t k, size_t i, size_t w){ return 2 * n * c * k * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t c, size_t k, size_t i, size_t w){
        return std::make_tuple(smat4(n, k, i, i), smat4(k, c, w, w), smat4(n, c, i, i)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_back<1,1,1,1>(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::STD, (etl::conv_4d_valid_back<1,1,1,1>(a, b))); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, (etl::conv_4d_valid_back<1,1,1,1>(a, b))); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, (etl::conv_4d_valid_back<1,1,1,1>(a, b))); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, (etl::conv_4d_valid_back<1,1,1,1>(a, b))); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid_filter [conv][conv4]", conv_4d_valid_policy,
    FLOPS([](size_t n, size_t k, size_t c, size_t i, size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t k, size_t c, size_t i, size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(n, k, w, w), smat4(k, c, i - w + 1, i - w + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_filter(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::STD, etl::conv_4d_valid_filter(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_valid_filter(a, b)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, etl::conv_4d_valid_filter(a, b)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, etl::conv_4d_valid_filter(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid_filter_flipped [conv][conv4]", conv_4d_valid_policy,
    FLOPS([](size_t n, size_t k, size_t c, size_t i, size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t k, size_t c, size_t i, size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(n, k, w, w), smat4(k, c, i - w + 1, i - w + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_filter_flipped(a, b); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_valid_filter_flipped(a, b)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, etl::conv_4d_valid_filter_flipped(a, b)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, etl::conv_4d_valid_filter_flipped(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_full [conv][conv4]", conv_4d_full_policy_1,
    FLOPS([](size_t n, size_t c, size_t k, size_t i, size_t w){ return 2 * n * c * k * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t c, size_t k, size_t i, size_t w){
        return std::make_tuple(smat4(n, k, i, i), smat4(k, c, w, w), smat4(n, c, i + w - 1, i + w - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_full(a, b); }),
    CPM_SECTION_FUNCTOR("fft_std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_STD, etl::conv_4d_full(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_full(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_MKL, etl::conv_4d_full(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_CUFFT, etl::conv_4d_full(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_full(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_full_flipped_1 [conv][conv4]", conv_4d_full_policy_1,
    FLOPS([](size_t n, size_t c, size_t k, size_t i, size_t w){ return 2 * n * c * k * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t c, size_t k, size_t i, size_t w){
        return std::make_tuple(smat4(n, k, i, i), smat4(k, c, w, w), smat4(n, c, i + w - 1, i + w - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_full_flipped(a, b); }),
    CPM_SECTION_FUNCTOR("fft_std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_STD, etl::conv_4d_full_flipped(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_full_flipped(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_MKL, etl::conv_4d_full_flipped(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_CUFFT, etl::conv_4d_full_flipped(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_full_flipped(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_full_flipped_2 [conv][conv4]", conv_4d_full_policy_2,
    FLOPS([](size_t n, size_t c, size_t k, size_t i, size_t w){ return 2 * n * c * k * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t c, size_t k, size_t i, size_t w){
        return std::make_tuple(smat4(n, k, i, i), smat4(k, c, w, w), smat4(n, c, i + w - 1, i + w - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_full_flipped(a, b); }),
    CPM_SECTION_FUNCTOR("fft_std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_STD, etl::conv_4d_full_flipped(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_full_flipped(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_MKL, etl::conv_4d_full_flipped(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_CUFFT, etl::conv_4d_full_flipped(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_full_flipped(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_full_flipped_3 [conv][conv4]", conv_4d_full_policy_3,
    FLOPS([](size_t n, size_t c, size_t k, size_t i, size_t w){ return 2 * n * c * k * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t c, size_t k, size_t i, size_t w){
        return std::make_tuple(smat4(n, k, i, i), smat4(k, c, w, w), smat4(n, c, i + w - 1, i + w - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_full_flipped(a, b); }),
    CPM_SECTION_FUNCTOR("fft_std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_STD, etl::conv_4d_full_flipped(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_full_flipped(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_MKL, etl::conv_4d_full_flipped(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_CUFFT, etl::conv_4d_full_flipped(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_full_flipped(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_full_flipped_4 [conv][conv4]", conv_4d_full_policy_4,
    FLOPS([](size_t n, size_t c, size_t k, size_t i, size_t w){ return 2 * n * c * k * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t c, size_t k, size_t i, size_t w){
        return std::make_tuple(smat4(n, k, i, i), smat4(k, c, w, w), smat4(n, c, i + w - 1, i + w - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_full_flipped(a, b); }),
    CPM_SECTION_FUNCTOR("fft_std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_STD, etl::conv_4d_full_flipped(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_full_flipped(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_MKL, etl::conv_4d_full_flipped(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_CUFFT, etl::conv_4d_full_flipped(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_full_flipped(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("imagenet_forward [conv][conv4]", imagenet_forward32_policy,
    FLOPS([](size_t n, size_t k, size_t c, size_t i, size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t k, size_t c, size_t i, size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(k, c, w, w), smat4(n, k, i, i)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("imagenet_forward128 [conv][conv4]", imagenet_forward128_policy,
    FLOPS([](size_t n, size_t k, size_t c, size_t i, size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t k, size_t c, size_t i, size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(k, c, w, w), smat4(n, k, i, i)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1)); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1)); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid_flipped(a, b, 1, 1, 1, 1)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("imagenet_backward [conv][conv4]", imagenet_backward_policy,
    FLOPS([](size_t n, size_t c, size_t k, size_t i, size_t w){ return 2 * n * c * k * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t c, size_t k, size_t i, size_t w){
        return std::make_tuple(smat4(n, k, i, i), smat4(k, c, w, w), smat4(n, c, i, i)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_back<1,1,1,1>(a, b); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, (etl::conv_4d_valid_back<1,1,1,1>(a, b))); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, (etl::conv_4d_valid_back<1,1,1,1>(a, b))); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, (etl::conv_4d_valid_back<1,1,1,1>(a, b))); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("imagenet_gradients [conv][conv4]", imagenet_gradients_policy,
    FLOPS([](size_t n, size_t k, size_t c, size_t i, size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](size_t n, size_t k, size_t c, size_t i, size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(n, k, w, w), smat4(k, c, i - w + 1 + 2, i - w + 1 + 2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_filter_flipped<1,1,1,1>(a, b); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::VEC, (etl::conv_4d_valid_filter_flipped<1,1,1,1>(a, b))); })
    VEC_SECTION_FUNCTOR("blas_vec", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_VEC, (etl::conv_4d_valid_filter_flipped<1,1,1,1>(a, b))); })
    BLAS_SECTION_FUNCTOR("blas_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS_MKL, (etl::conv_4d_valid_filter_flipped<1,1,1,1>(a, b))); })
)

#endif
