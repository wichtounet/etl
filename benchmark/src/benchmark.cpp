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
    // TODO Find out why this is not as fast as Blaze
#ifdef ETL_EXTENDED_BENCH
    CPM_TWO_PASS_NS(
        "r = a + b (d) [std][add][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (s) [std][add][d]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& a, svec& b, svec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b + c [std][add][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& c, dvec& r){ r = a + b + c; },
        [](std::size_t d){ return 2 * d; }
        );
#endif

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

//Bench exp
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "sexp [std][exp][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = exp(a); },
        [](std::size_t d){ return 100 * d; }
        );

    CPM_TWO_PASS_NS(
        "dexp [std][exp][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = exp(a); },
        [](std::size_t d){ return 100 * d; }
        );

    CPM_TWO_PASS_NS(
        "sexp_expr [std][exp][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& a, svec& b, svec& r){ r = exp(a * 2.3 + b); },
        [](std::size_t d){ return 105 * d; }
        );
}

//Bench log
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "slog [std][log][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = log(a); },
        [](std::size_t d){ return 100 * d; }
        );

    CPM_TWO_PASS_NS(
        "dlog [std][log][d]",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = log(a); },
        [](std::size_t d){ return 100 * d; }
        );
}

CPM_DIRECT_SECTION_TWO_PASS_NS_P("ssum [std][sum][s]", dot_policy,
    CPM_SECTION_INIT([](std::size_t d1){ return std::make_tuple(svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a){ float_ref += etl::sum(a); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a){ SELECTED_SECTION(etl::sum_impl::STD){ float_ref += etl::sum(a); } })
    VEC_SECTION_FUNCTOR("vec", [](svec& a){ SELECTED_SECTION(etl::sum_impl::VEC){ float_ref += etl::sum(a); } })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dsum [std][sum][d]", dot_policy,
    CPM_SECTION_INIT([](std::size_t d1){ return std::make_tuple(dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a){ double_ref += etl::sum(a); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a){ SELECTED_SECTION(etl::sum_impl::STD){ double_ref += etl::sum(a); } })
    VEC_SECTION_FUNCTOR("vec", [](dvec& a){ SELECTED_SECTION(etl::sum_impl::VEC){ double_ref += etl::sum(a); } })
)

// Bench complex sums
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "ssum_expr1 [std][sum][s]",
        [](std::size_t d){ return std::make_tuple(svec(d)); },
        [](svec& a){ float_ref += etl::sum((a >> a) + (a >> a)); },
        [](std::size_t d){ return 4 * d; }
        );

#ifdef ETL_EXTENDED_BENCH
    CPM_TWO_PASS_NS(
        "ssum_expr2 [std][sum][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& b){ float_ref += etl::sum((a >> a) - (b >> b)); },
        [](std::size_t d){ return 4 * d; }
        );
#endif
}

//Bench transposition
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        trans_policy,
        "r = tranpose(a) (s) [transpose][s]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1, d2), smat(d2, d1)); },
        [](smat& a, smat& r){ r = a.transpose(); }
        );

    CPM_TWO_PASS_NS_P(
        trans_policy,
        "r = tranpose(a) (d) [transpose][d]",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d2), dmat(d2, d1)); },
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
CPM_DIRECT_SECTION_TWO_PASS_NS_F("a = sigmoid(b) (s) [std][sigmoid][d]",
    FLOPS([](std::size_t d){ return 22 * d; }),
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(svec(d), svec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b){ a = etl::sigmoid(b); }),
    CPM_SECTION_FUNCTOR("fast", [](svec& a, svec& b){ a = etl::fast_sigmoid(b); }),
    CPM_SECTION_FUNCTOR("hard", [](svec& a, svec& b){ a = etl::hard_sigmoid(b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_F("a = sigmoid(b) (d) [std][sigmoid][d]",
    FLOPS([](std::size_t d){ return 22 * d; }),
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(dvec(d), dvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b){ a = etl::sigmoid(b); }),
    CPM_SECTION_FUNCTOR("fast", [](dvec& a, dvec& b){ a = etl::fast_sigmoid(b); }),
    CPM_SECTION_FUNCTOR("hard", [](dvec& a, dvec& b){ a = etl::hard_sigmoid(b); })
)

#ifdef ETL_EXTENDED_BENCH

//Bench etl_complex
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a >> b (ec) [std][mul][etl_complex][c]",
        [](std::size_t d){ return std::make_tuple(ecvec(d), ecvec(d), ecvec(d)); },
        [](ecvec& a, ecvec& b, ecvec& r){ r = a >> b; },
        [](std::size_t d){ return 6 * d; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (ez) [std][mul][etl_complex][z]",
        [](std::size_t d){ return std::make_tuple(ezvec(d), ezvec(d), ezvec(d)); },
        [](ezvec& a, ezvec& b, ezvec& r){ r = a >> b; },
        [](std::size_t d){ return 6 * d; }
        );
}

#endif

CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_h(c=2) (s) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(smat(d, d), smat(d,d)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_h<2,2>(a); },
        [](std::size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "dyn_pmp_h(c=2) (s) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(smat(d, d), smat(d, d)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_h(a, 2, 2); },
        [](std::size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_p(c=2) (s) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(smat(d, d), smat(d/2,d/2)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_p<2,2>(a); },
        [](std::size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_h(c=4) (s) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(smat(d, d), smat(d,d)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_h<4,4>(a); },
        [](std::size_t d){ return 2 * d * d * 4 * 4; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_p(c=4) (s) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(smat(d, d), smat(d/4,d/4)); },
        [](smat& a, smat& r){ r = etl::p_max_pool_p<4,4>(a); },
        [](std::size_t d){ return 2 * d * d * 4 * 4; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_h(c=2) (d) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(dmat(d, d), dmat(d,d)); },
        [](dmat& a, dmat& r){ r = etl::p_max_pool_h<2,2>(a); },
        [](std::size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_p(c=2) (d) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(dmat(d, d), dmat(d/2,d/2)); },
        [](dmat& a, dmat& r){ r = etl::p_max_pool_p<2,2>(a); },
        [](std::size_t d){ return 2 * d * d * 2 * 2; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_h(c=4) (d) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(dmat(d, d), dmat(d,d)); },
        [](dmat& a, dmat& r){ r = etl::p_max_pool_h<4,4>(a); },
        [](std::size_t d){ return 2 * d * d * 4 * 4; }
        );

    CPM_TWO_PASS_NS_P(
        pmp_policy,
        "pmp_p(c=4) (d) [pmp][s]",
        [](std::size_t d){ return std::make_tuple(dmat(d, d), dmat(d/4,d/4)); },
        [](dmat& a, dmat& r){ r = etl::p_max_pool_p<4,4>(a); },
        [](std::size_t d){ return 2 * d * d * 4 * 4; }
        );
}

//Bench scalar operations
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a + 1.25 (s) [std][scalar][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a + 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r = a - 1.25 (s) [std][scalar][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a - 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r = a * 1.25 (s) [std][scalar][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a >> 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r = a / 1.25 (s) [std][scalar][s]",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a / 1.25f; }
        );
}

// Bench scalar compound operations

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r += 1.25 (s) [std][scalar][s]", large_vector_policy,
    FLOPS([](std::size_t d){ return d; }),
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(svec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a){ a += 1.25f; }),
    CPM_SECTION_FUNCTOR("std", [](svec& a){ SELECTED_SECTION(etl::scalar_impl::STD) { a += 1.25f; } })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a){ SELECTED_SECTION(etl::scalar_impl::BLAS) { a += 1.25f; } })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r -= 1.25 (s) [std][scalar][s]", large_vector_policy,
    FLOPS([](std::size_t d){ return d; }),
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(svec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a){ a -= 1.25f; }),
    CPM_SECTION_FUNCTOR("std", [](svec& a){ SELECTED_SECTION(etl::scalar_impl::STD) { a -= 1.25f; } })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a){ SELECTED_SECTION(etl::scalar_impl::BLAS) { a -= 1.25f; } })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r *= 1.25 (s) [std][scalar][s]", large_vector_policy,
    FLOPS([](std::size_t d){ return d; }),
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(svec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a){ a *= 1.25f; }),
    CPM_SECTION_FUNCTOR("std", [](svec& a){ SELECTED_SECTION(etl::scalar_impl::STD) { a *= 1.25f; } })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a){ SELECTED_SECTION(etl::scalar_impl::BLAS) { a *= 1.25f; } })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r /= 1.25 (s) [std][scalar][s]", large_vector_policy,
    FLOPS([](std::size_t d){ return d; }),
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(svec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a){ a /= 1.25f; }),
    CPM_SECTION_FUNCTOR("std", [](svec& a){ SELECTED_SECTION(etl::scalar_impl::STD) { a /= 1.25f; } })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a){ SELECTED_SECTION(etl::scalar_impl::BLAS) { a /= 1.25f; } })
)
