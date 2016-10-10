//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

//1D-Convolution benchmarks with large-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        large_kernel_policy,
        "r = conv_1d_full(a,b)(large) [conv][conv1]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); },
        [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }
        );

    CPM_TWO_PASS_NS_P(
        large_kernel_policy,
        "r = conv_1d_same(a,b)(large) [conv][conv1]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); },
        [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }
        );

    CPM_TWO_PASS_NS_P(
        large_kernel_policy,
        "r = conv_1d_valid(a,b)(large) [conv][conv1]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); },
        [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }
        );
}

//1D-Convolution benchmarks with small-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        small_kernel_policy,
        "r = conv_1d_full(a,b)(small) [conv][conv1]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }
        );

    CPM_TWO_PASS_NS_P(
        small_kernel_policy,
        "r = conv_1d_same(a,b)(small) [conv][conv1]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }
        );

    CPM_TWO_PASS_NS_P(
        small_kernel_policy,
        "r = conv_1d_valid(a,b)(small) [conv][conv1]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }
        );
}

//2D-Convolution benchmarks with large-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        large_kernel_policy_2d,
        "r = conv_2d_full(a,b)(large) [conv][conv2]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }
        );

    CPM_TWO_PASS_NS_P(
        large_kernel_policy_2d,
        "r = conv_2d_same(a,b)(large) [conv][conv2]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1, d1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }
        );

    CPM_TWO_PASS_NS_P(
        large_kernel_policy_2d,
        "r = conv_2d_valid(a,b)(large) [conv][conv2]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }
        );
}

//2D-Convolution benchmarks with small-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        small_kernel_policy_2d,
        "r = conv_2d_full(a,b)(small) [conv][conv2]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }
        );

    CPM_TWO_PASS_NS_P(
        small_kernel_policy_2d,
        "r = conv_2d_same(a,b)(small) [conv][conv2]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1, d1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }
        );

    CPM_TWO_PASS_NS_P(
        small_kernel_policy_2d,
        "r = conv_2d_valid(a,b)(small) [conv][conv2]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }
        );
}

CPM_DIRECT_BENCH_TWO_PASS_P(
    NARY_POLICY(VALUES_POLICY(16, 16, 32, 32, 64, 64), VALUES_POLICY(4, 8, 8, 16, 16, 24)),
    "convmtx2 [conv][convmtx]",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat((d1 + d2 - 1)*(d1 + d2 - 1), d2 * d2)); },
    [](std::size_t /*d1*/, std::size_t d2, dmat& a, dmat& b){ b = etl::convmtx2(a, d2, d2); }
)

CPM_DIRECT_BENCH_TWO_PASS_P(
    NARY_POLICY(VALUES_POLICY(16, 16, 32, 32, 64, 64, 128), VALUES_POLICY(4, 8, 8, 16, 16, 32, 32)),
    "convmtx2_t [conv][convmtx]",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat((d1 + d2 - 1)*(d1 + d2 - 1), d2 * d2)); },
    [](std::size_t /*d1*/, std::size_t d2, dmat& a, dmat& b){ etl::convmtx2_direct_t(b, a, d2, d2); }
)

CPM_DIRECT_BENCH_TWO_PASS_P(
    NARY_POLICY(VALUES_POLICY(16, 16, 32, 32, 64, 64, 128), VALUES_POLICY(4, 8, 8, 16, 16, 32, 32)),
    "im2col_direct [conv][im2col]",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2 * d2, (d1 - d2 + 1)*(d1 - d2 + 1))); },
    [](std::size_t /*d1*/, std::size_t d2, dmat& a, dmat& b){ etl::im2col_direct(b, a, d2, d2); }
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv1_valid [conv][conv1]", conv_1d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_valid(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_valid(a, b)); })
)

#ifdef ETL_EXTENDED_BENCH
#define sdm_t1 etl::dyn_vector
#define fdm_t1 etl::fast_dyn_matrix

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv1_valid_dyn_1 [conv][conv1]", fast_policy,
    FLOPS([](std::size_t){ return 2 * 10000 * 5000; }),
    CPM_SECTION_INIT([](std::size_t){ return std::make_tuple(sdm_t1<float>(10000), sdm_t1<float>(5000), sdm_t1<float>(5001)); }),
    CPM_SECTION_FUNCTOR("default", [](auto& a, auto& b, auto& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](auto& a, auto& b, auto& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_valid(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](auto& a, auto& b, auto& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](auto& a, auto& b, auto& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_valid(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv1_valid_fast_1 [conv][conv1]", fast_policy,
    FLOPS([](std::size_t){ return 2 * 10000 * 5000; }),
    CPM_SECTION_INIT([](std::size_t){ return std::make_tuple(fdm_t1<float,10000>(), fdm_t1<float,5000>(), fdm_t1<float,5001>()); }),
    CPM_SECTION_FUNCTOR("default", [](auto& a, auto& b, auto& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](auto& a, auto& b, auto& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_valid(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](auto& a, auto& b, auto& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](auto& a, auto& b, auto& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_valid(a, b)); })
)

#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv1_valid [conv][conv1]", conv_1d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_valid(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_valid(a, b)); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv1_same [conv][conv1]", conv_1d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_same(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_same(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_same(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv1_same [conv][conv1]", conv_1d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_same(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_same(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_same(a, b)); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv1_full [conv][conv1]", conv_1d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_full(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_full(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_full(a, b)); })
    ,CPM_SECTION_FUNCTOR("fft_std", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::FFT_STD, etl::conv_1d_full(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_1d_full(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](svec& a, svec& b, svec& r){ r = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_1d_full(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv1_full [conv][conv1]", conv_1d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_full(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_full(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_full(a, b)); })
    ,CPM_SECTION_FUNCTOR("fft_std", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::FFT_STD, etl::conv_1d_full(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_1d_full(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_1d_full(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_valid(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_valid(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_valid(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid_pad [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat((d1 - d2 + 2 * 3) + 1, (d1 - d2 + 2 * 3) + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = (etl::conv_2d_valid<1, 1, 3, 3>(a, b)); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, (etl::conv_2d_valid<1, 1, 3, 3>(a, b))); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::SSE, (etl::conv_2d_valid<1, 1, 3, 3>(a, b))); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::AVX, (etl::conv_2d_valid<1, 1, 3, 3>(a, b))); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::CUDNN, (etl::conv_2d_valid<1, 1, 3, 3>(a, b))); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_valid [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_valid(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_valid(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_valid(a, b)); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_same [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_same(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_same(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_same(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_same [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_same(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_same(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_same(a, b)); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_full [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_full(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_full(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_full(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_full(a, b)); })
    ,CPM_SECTION_FUNCTOR("fft_std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::FFT_STD, etl::conv_2d_full(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_2d_full(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_2d_full(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_full_small [conv][conv2]", conv_2d_small_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_full(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_full(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_full(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_full(a, b)); })
    ,CPM_SECTION_FUNCTOR("fft_std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::FFT_STD, etl::conv_2d_full(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_2d_full(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_2d_full(a, b)); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_full [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_full(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_full(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_full(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::CUDNN, etl::conv_2d_full(a, b)); })
    ,CPM_SECTION_FUNCTOR("fft_std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::FFT_STD, etl::conv_2d_full(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_2d_full(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_2d_full(a, b)); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid_multi [conv][conv2]", conv_2d_multi_policy,
    FLOPS([](std::size_t d1, std::size_t d2, std::size_t d3){ return 2 * d1 * d1 * d2 * d2 * d3; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2, std::size_t d3){ return std::make_tuple(smat(d1,d1), smat3(d3,d2,d2), smat3(d3,d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat3& b, smat3& r){ r = etl::conv_2d_valid_multi(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::STD, etl::conv_2d_valid_multi(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::SSE, etl::conv_2d_valid_multi(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::AVX, etl::conv_2d_valid_multi(a, b)); }),
    CPM_SECTION_FUNCTOR("fft", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::FFT, etl::conv_2d_valid_multi(a, b)); }),
    CPM_SECTION_FUNCTOR("blas", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::BLAS, etl::conv_2d_valid_multi(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::CUDNN, etl::conv_2d_valid_multi(a, b)); })
)

#ifdef ETL_EXTENDED_BENCH

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid_multi_flipped [conv][conv2]", conv_2d_multi_policy,
    FLOPS([](std::size_t d1, std::size_t d2, std::size_t d3){ return 2 * d1 * d1 * d2 * d2 * d3; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2, std::size_t d3){ return std::make_tuple(smat(d1,d1), smat3(d3,d2,d2), smat3(d3,d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat3& b, smat3& r){ r = etl::conv_2d_valid_multi_flipped(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::STD, etl::conv_2d_valid_multi_flipped(a, b)); }),
    CPM_SECTION_FUNCTOR("fft", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::FFT, etl::conv_2d_valid_multi_flipped(a, b)); }),
    CPM_SECTION_FUNCTOR("blas", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::BLAS, etl::conv_2d_valid_multi_flipped(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat& a, smat3& b, smat3& r){ r = selected_helper(etl::conv_multi_impl::CUDNN, etl::conv_2d_valid_multi_flipped(a, b)); })
)

// Note: STD is way too slow to benchmark

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid [conv][conv4]", conv_4d_valid_policy,
    FLOPS([](std::size_t n, std::size_t k, std::size_t c, std::size_t i, std::size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](std::size_t n, std::size_t k, std::size_t c, std::size_t i, std::size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(k, c, w, w), smat4(n, k, i - w + 1, i - w + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid(a, b); })
    SSE_SECTION_FUNCTOR("sse", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::SSE, etl::conv_4d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::AVX, etl::conv_4d_valid(a, b)); }),
    CPM_SECTION_FUNCTOR("blas", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS, etl::conv_4d_valid(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid_flipped [conv][conv4]", conv_4d_valid_policy,
    FLOPS([](std::size_t n, std::size_t k, std::size_t c, std::size_t i, std::size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](std::size_t n, std::size_t k, std::size_t c, std::size_t i, std::size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(k, c, w, w), smat4(n, k, i - w + 1, i - w + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_flipped(a, b); })
    SSE_SECTION_FUNCTOR("sse", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::SSE, etl::conv_4d_valid_flipped(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::AVX, etl::conv_4d_valid_flipped(a, b)); }),
    CPM_SECTION_FUNCTOR("blas", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS, etl::conv_4d_valid(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid_flipped(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid_filter [conv][conv4]", conv_4d_valid_policy,
    FLOPS([](std::size_t n, std::size_t k, std::size_t c, std::size_t i, std::size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](std::size_t n, std::size_t k, std::size_t c, std::size_t i, std::size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(n, k, w, w), smat4(k, c, i - w + 1, i - w + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_filter(a, b); })
    SSE_SECTION_FUNCTOR("sse", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::SSE, etl::conv_4d_valid_filter(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::AVX, etl::conv_4d_valid_filter(a, b)); }),
    CPM_SECTION_FUNCTOR("blas", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS, etl::conv_4d_valid_filter(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid_filter(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_valid_filter_flipped [conv][conv4]", conv_4d_valid_policy,
    FLOPS([](std::size_t n, std::size_t k, std::size_t c, std::size_t i, std::size_t w){ return 2 * n * k * c * i * i * w * w; }),
    CPM_SECTION_INIT([](std::size_t n, std::size_t k, std::size_t c, std::size_t i, std::size_t w){
        return std::make_tuple(smat4(n, c, i, i), smat4(n, k, w, w), smat4(k, c, i - w + 1, i - w + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_valid_filter_flipped(a, b); })
    SSE_SECTION_FUNCTOR("sse", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::SSE, etl::conv_4d_valid_filter_flipped(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::AVX, etl::conv_4d_valid_filter_flipped(a, b)); }),
    CPM_SECTION_FUNCTOR("blas", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::BLAS, etl::conv_4d_valid_filter_flipped(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_valid_filter_flipped(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv4_full [conv][conv4]", conv_4d_full_policy,
    FLOPS([](std::size_t n, std::size_t c, std::size_t k, std::size_t i, std::size_t w){ return 2 * n * c * k * i * i * w * w; }),
    CPM_SECTION_INIT([](std::size_t n, std::size_t c, std::size_t k, std::size_t i, std::size_t w){
        return std::make_tuple(smat4(n, k, i, i), smat4(k, c, w, w), smat4(n, c, i + w - 1, i + w - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, smat4& b, smat4& r){ r = etl::conv_4d_full(a, b); }),
    CPM_SECTION_FUNCTOR("fft_std", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::FFT_STD, etl::conv_4d_full(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::SSE, etl::conv_4d_full(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::AVX, etl::conv_4d_full(a, b)); })
    MKL_SECTION_FUNCTOR("fft_mkl", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv_impl::FFT_MKL, etl::conv_4d_full(a, b)); })
    CUFFT_SECTION_FUNCTOR("fft_cufft", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv_impl::FFT_CUFFT, etl::conv_4d_full(a, b)); })
    CUDNN_SECTION_FUNCTOR("cudnn", [](smat4& a, smat4& b, smat4& r){ r = selected_helper(etl::conv4_impl::CUDNN, etl::conv_4d_full(a, b)); })
)

#endif
