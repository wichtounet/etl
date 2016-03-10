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
    ,CPM_SECTION_FUNCTOR("fft", [](svec& a, svec& b, svec& r){ r = etl::fft_conv_1d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](svec& a, svec& b, svec& r){ etl::impl::reduc::conv1_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv1_full [conv][conv1]", conv_1d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_1d_full(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_1d_full(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_1d_full(a, b)); })
    ,CPM_SECTION_FUNCTOR("fft", [](dvec& a, dvec& b, dvec& r){ r = etl::fft_conv_1d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](dvec& a, dvec& b, dvec& r){ etl::impl::reduc::conv1_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sconv2_valid [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_valid(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_valid(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_valid [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_valid(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_valid(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_valid(a, b)); })
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
    ,CPM_SECTION_FUNCTOR("fft", [](smat& a, smat& b, smat& r){ r = etl::fft_conv_2d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](smat& a, smat& b, smat& r){ etl::impl::reduc::conv2_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("dconv2_full [conv][conv2]", conv_2d_large_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::STD, etl::conv_2d_full(a, b)); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::SSE, etl::conv_2d_full(a, b)); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ r = selected_helper(etl::conv_impl::AVX, etl::conv_2d_full(a, b)); })
    ,CPM_SECTION_FUNCTOR("fft", [](dmat& a, dmat& b, dmat& r){ r = etl::fft_conv_2d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](dmat& a, dmat& b, dmat& r){ etl::impl::reduc::conv2_full(a, b, r); })
)
