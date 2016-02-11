//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_BENCHMARK "Tests Benchmarks"
#include "benchmark.hpp"

namespace {

float float_ref = 0.0;
double double_ref = 0.0;

} //end of anonymous namespace

//Bench assignment
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a (s) [std][assign][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (d) [std][assign][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (c) [std][assign][c]",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d)); },
        [](cvec& a, cvec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (z) [std][assign][z]",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d)); },
        [](zvec& a, zvec& r){ r = a; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = A (d) [std][assign][d]",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& R){ R = A; }
        );
}

//Bench addition
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a + b [std][add][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b + c [std][add][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& c, dvec& r){ r = a + b + c; },
        [](std::size_t d){ return 2 * d; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = A + B [std][add][d]",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& R){ R = A + B; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R += A [std][add][d]",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& R){ R += A; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R += A + B [std][add][d]",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& R){ R += A + B; },
        [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (c) [std][add][complex][c]",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a + b; },
        [](std::size_t d){ return 2 * d; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (z) [std][add][complex][z]",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a + b; },
        [](std::size_t d){ return 2 * d; }
        );
}

//Bench subtraction
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a - b [std][sub][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a - b; }
        );
}

//Bench multiplication
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a >> b (s) [std][mul][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& a, svec& b, svec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (d) [std][mul][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (c) [std][mul][complex][c]",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (z) [std][mul][complex][z]",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a >> b; },
        [](std::size_t d){ return 6 * d; }
        );
}

//Bench division
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a / b (s) [std][div][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& a, svec& b, svec& r){ r = a / b; }
        );

    CPM_TWO_PASS_NS(
        "r = a / b (d) [std][div][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a / b; }
        );

    CPM_TWO_PASS_NS(
        "r = a / b (c) [std][div][complex][c]",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a / b; },
        [](std::size_t d){ return 6 * d; }
        );

    CPM_TWO_PASS_NS(
        "r = a / b (z) [std][div][complex][z]",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a / b; },
        [](std::size_t d){ return 6 * d; }
        );
}

//Bench saxpy
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "saxpy [std][axpy][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a * 2.3 + r; },
        [](std::size_t d){ return 2 * d; }
        );

    CPM_TWO_PASS_NS(
        "daxpy [std][axpy][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = a * 2.3 + r; },
        [](std::size_t d){ return 2 * d; }
        );
}

CPM_DIRECT_SECTION_TWO_PASS_NS("ssum [std][sum][s]",
    CPM_SECTION_INIT([](std::size_t d1){ return std::make_tuple(svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a){ float_ref = etl::sum(a); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a){ float_ref = etl::detail::sum_direct(a, etl::detail::sum_imple::STD); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a){ float_ref = etl::detail::sum_direct(a, etl::detail::sum_imple::SSE); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a){ float_ref = etl::detail::sum_direct(a, etl::detail::sum_imple::AVX); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS("dsum [std][sum][d]",
    CPM_SECTION_INIT([](std::size_t d1){ return std::make_tuple(dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a){ double_ref = etl::sum(a); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a){ double_ref = etl::detail::sum_direct(a, etl::detail::sum_imple::STD); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a){ double_ref = etl::detail::sum_direct(a, etl::detail::sum_imple::SSE); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a){ double_ref = etl::detail::sum_direct(a, etl::detail::sum_imple::AVX); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_F("ssum_expr1 [std][sum][s]",
    FLOPS([](std::size_t d){ return 4 * d; }),
    CPM_SECTION_INIT([](std::size_t d1){ return std::make_tuple(svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a){ float_ref = etl::sum((a >> a) + (a >> a)); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a){ float_ref = etl::detail::sum_direct((a >> a) + (a >> a), etl::detail::sum_imple::STD); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a){ float_ref = etl::detail::sum_direct((a >> a) + (a >> a), etl::detail::sum_imple::SSE); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a){ float_ref = etl::detail::sum_direct((a >> a) + (a >> a), etl::detail::sum_imple::AVX); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_F("ssum_expr2 [std][sum][s]",
    FLOPS([](std::size_t d){ return 4 * d; }),
    CPM_SECTION_INIT([](std::size_t d1){ return std::make_tuple(svec(d1), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b){ float_ref = etl::sum((a >> a) - (b >> b)); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b){ float_ref = etl::detail::sum_direct((a >> a) - (b >> b), etl::detail::sum_imple::STD); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b){ float_ref = etl::detail::sum_direct((a >> a) - (b >> b), etl::detail::sum_imple::SSE); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b){ float_ref = etl::detail::sum_direct((a >> a) - (b >> b), etl::detail::sum_imple::AVX); })
)

//Bench transposition
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        trans_policy,
        "r = tranpose(a) (s) [transpose][s]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1, d2), smat(d1, d2)); },
        [](smat& a, smat& r){ r = a.transpose(); }
        );

    CPM_TWO_PASS_NS_P(
        trans_policy,
        "r = tranpose(a) (d) [transpose][d]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& a, dmat& r){ r = a.transpose(); }
        );

    CPM_TWO_PASS_NS_P(
        trans_policy,
        "a = tranpose(a) (s) [transpose][inplace][s]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1, d2)); },
        [](smat& a){ a.transpose_inplace(); }
        );

    CPM_TWO_PASS_NS_P(
        trans_policy,
        "a = tranpose(a) (d) [transpose][inplace][d]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d2)); },
        [](dmat& a){ a.transpose_inplace(); }
        );
}

//Sigmoid benchmark
CPM_BENCH() {
    //Flops: 20 for exp, 1 for div, 1 for add

    CPM_TWO_PASS_NS(
        "r = sigmoid(a)",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = etl::sigmoid(a); },
        [](std::size_t d){ return 22 * d; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = sigmoid(A)",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& R){ R = etl::sigmoid(A); },
        [](auto d1, auto d2){ return 22 * d1 * d2; }
        );
}

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sigmoid(s)", sigmoid_policy,
    FLOPS([](std::size_t d){ return 22 * d; }),
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(smat(d,d), smat(d,d)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b){ a = etl::sigmoid(b); }),
    CPM_SECTION_FUNCTOR("fast", [](smat& a, smat& b){ a = etl::fast_sigmoid(b); }),
    CPM_SECTION_FUNCTOR("hard", [](smat& a, smat& b){ a = etl::hard_sigmoid(b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sigmoid(d)", sigmoid_policy,
    FLOPS([](std::size_t d){ return 22 * d; }),
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(dmat(d,d), dmat(d,d)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b){ a = etl::sigmoid(b); }),
    CPM_SECTION_FUNCTOR("fast", [](dmat& a, dmat& b){ a = etl::fast_sigmoid(b); }),
    CPM_SECTION_FUNCTOR("hard", [](dmat& a, dmat& b){ a = etl::hard_sigmoid(b); })
)
