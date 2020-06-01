//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (d) [std][assign][d]",
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (c) [std][assign][c]",
        [](size_t d){ return std::make_tuple(cvec(d), cvec(d)); },
        [](cvec& a, cvec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (z) [std][assign][z]",
        [](size_t d){ return std::make_tuple(zvec(d), zvec(d)); },
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
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (s) [std][add][d]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& a, svec& b, svec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b + c [std][add][d]",
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& c, dvec& r){ r = a + b + c; },
        [](size_t d){ return 2 * d; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (i8) [std][add][i]",
        [](size_t d){ return std::make_tuple(i8vec(d), i8vec(d), i8vec(d)); },
        [](i8vec& a, i8vec& b, i8vec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (i16) [std][add][i]",
        [](size_t d){ return std::make_tuple(i16vec(d), i16vec(d), i16vec(d)); },
        [](i16vec& a, i16vec& b, i16vec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (i32) [std][add][i]",
        [](size_t d){ return std::make_tuple(i32vec(d), i32vec(d), i32vec(d)); },
        [](i32vec& a, i32vec& b, i32vec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (i64) [std][add][i]",
        [](size_t d){ return std::make_tuple(i64vec(d), i64vec(d), i64vec(d)); },
        [](i64vec& a, i64vec& b, i64vec& r){ r = a + b; }
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
        [](size_t d1, size_t d2){ return 2 * d1 * d2; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (c) [std][add][complex][c]",
        [](size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a + b; },
        [](size_t d){ return 2 * d; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (z) [std][add][complex][z]",
        [](size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a + b; },
        [](size_t d){ return 2 * d; }
        );
}

//Bench subtraction
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a - b [std][sub][d]",
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a - b; }
        );
}

//Bench multiplication
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a >> b (s) [std][mul][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& a, svec& b, svec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (d) [std][mul][d]",
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (c) [std][mul][complex][c]",
        [](size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (z) [std][mul][complex][z]",
        [](size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a >> b; },
        [](size_t d){ return 6 * d; }
        );
}

//Bench division
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a / b (s) [std][div][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& a, svec& b, svec& r){ r = a / b; }
        );

    CPM_TWO_PASS_NS(
        "r = a / b (d) [std][div][d]",
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a / b; }
        );

    CPM_TWO_PASS_NS(
        "r = a / b (c) [std][div][complex][c]",
        [](size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a / b; },
        [](size_t d){ return 6 * d; }
        );

    CPM_TWO_PASS_NS(
        "r = a / b (z) [std][div][complex][z]",
        [](size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a / b; },
        [](size_t d){ return 6 * d; }
        );
}

//Bench saxpy
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "saxpy [std][axpy][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a * 2.3 + r; },
        [](size_t d){ return 2 * d; }
        );

    CPM_TWO_PASS_NS(
        "daxpy [std][axpy][d]",
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = a * 2.3 + r; },
        [](size_t d){ return 2 * d; }
        );
}

//Bench exp
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "sexp [std][exp][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = exp(a); },
        [](size_t d){ return 100 * d; }
        );

    CPM_TWO_PASS_NS(
        "dexp [std][exp][d]",
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = exp(a); },
        [](size_t d){ return 100 * d; }
        );

    CPM_TWO_PASS_NS(
        "sexp_expr [std][exp][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d), svec(d)); },
        [](svec& a, svec& b, svec& r){ r = exp(a * 2.3 + b); },
        [](size_t d){ return 105 * d; }
        );
}

//Bench log
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "slog [std][log][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = log(a); },
        [](size_t d){ return 100 * d; }
        );

    CPM_TWO_PASS_NS(
        "dlog [std][log][d]",
        [](size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = log(a); },
        [](size_t d){ return 100 * d; }
        );
}

CPM_DIRECT_SECTION_TWO_PASS_NS_P("ssum [std][sum][s]", dot_policy,
    CPM_SECTION_INIT([](size_t d1){ return std::make_tuple(svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a){ float_ref += etl::sum(a); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a){ SELECTED_SECTION(etl::sum_impl::STD){ float_ref += etl::sum(a); } })
    VEC_SECTION_FUNCTOR("vec", [](svec& a){ SELECTED_SECTION(etl::sum_impl::VEC){ float_ref += etl::sum(a); } })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a){ SELECTED_SECTION(etl::sum_impl::BLAS){ float_ref += etl::sum(a); } })
    CUBLAS_SECTION_FUNCTOR("cublas", [](svec& a){ SELECTED_SECTION(etl::sum_impl::CUBLAS){ float_ref += etl::sum(a); } })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dsum [std][sum][d]", dot_policy,
    CPM_SECTION_INIT([](size_t d1){ return std::make_tuple(dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a){ double_ref += etl::sum(a); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a){ SELECTED_SECTION(etl::sum_impl::STD){ double_ref += etl::sum(a); } })
    VEC_SECTION_FUNCTOR("vec", [](dvec& a){ SELECTED_SECTION(etl::sum_impl::VEC){ double_ref += etl::sum(a); } })
    BLAS_SECTION_FUNCTOR("blas", [](dvec& a){ SELECTED_SECTION(etl::sum_impl::BLAS){ float_ref += etl::sum(a); } })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dvec& a){ SELECTED_SECTION(etl::sum_impl::CUBLAS){ float_ref += etl::sum(a); } })
)

// Bench complex sums
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "ssum_expr1 [std][sum][s]",
        [](size_t d){ return std::make_tuple(svec(d)); },
        [](svec& a){ float_ref += etl::sum((a >> a) + (a >> a)); },
        [](size_t d){ return 4 * d; }
        );

#ifdef ETL_EXTENDED_BENCH
    CPM_TWO_PASS_NS(
        "ssum_expr2 [std][sum][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& b){ float_ref += etl::sum((a >> a) - (b >> b)); },
        [](size_t d){ return 4 * d; }
        );
#endif
}

CPM_DIRECT_SECTION_TWO_PASS_NS_P("strans [transpose][s]", trans_policy,
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d2), smat(d2,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& r){ r = transpose(a); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& r){ r = selected_helper(etl::transpose_impl::STD, transpose(a)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& r){ r = selected_helper(etl::transpose_impl::VEC, transpose(a)); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& r){ r = selected_helper(etl::transpose_impl::MKL, transpose(a)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& r){ r = selected_helper(etl::transpose_impl::CUBLAS, transpose(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("inplace_strans [transpose][s]", trans_policy,
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& r){ r.transpose_inplace(); }),
    CPM_SECTION_FUNCTOR("std", [](smat& r){ SELECTED_SECTION(etl::transpose_impl::STD){ r.transpose_inplace(); } })
    BLAS_SECTION_FUNCTOR("blas", [](smat& r){ SELECTED_SECTION(etl::transpose_impl::MKL){ r.transpose_inplace(); } })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& r){ SELECTED_SECTION(etl::transpose_impl::CUBLAS){ r.transpose_inplace(); } })
)

//Sigmoid benchmark
CPM_DIRECT_SECTION_TWO_PASS_NS_F("a = sigmoid(b) (s) [std][sigmoid][d]",
    FLOPS([](size_t d){ return 22 * d; }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(svec(d), svec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b){ a = etl::sigmoid(b); }),
    CPM_SECTION_FUNCTOR("fast", [](svec& a, svec& b){ a = etl::fast_sigmoid(b); }),
    CPM_SECTION_FUNCTOR("hard", [](svec& a, svec& b){ a = etl::hard_sigmoid(b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_F("a = sigmoid(b) (d) [std][sigmoid][d]",
    FLOPS([](size_t d){ return 22 * d; }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(dvec(d), dvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b){ a = etl::sigmoid(b); }),
    CPM_SECTION_FUNCTOR("fast", [](dvec& a, dvec& b){ a = etl::fast_sigmoid(b); }),
    CPM_SECTION_FUNCTOR("hard", [](dvec& a, dvec& b){ a = etl::hard_sigmoid(b); })
)

#ifdef ETL_EXTENDED_BENCH

//Bench etl_complex
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a >> b (ec) [std][mul][etl_complex][c]",
        [](size_t d){ return std::make_tuple(ecvec(d), ecvec(d), ecvec(d)); },
        [](ecvec& a, ecvec& b, ecvec& r){ r = a >> b; },
        [](size_t d){ return 6 * d; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (ez) [std][mul][etl_complex][z]",
        [](size_t d){ return std::make_tuple(ezvec(d), ezvec(d), ezvec(d)); },
        [](ezvec& a, ezvec& b, ezvec& r){ r = a >> b; },
        [](size_t d){ return 6 * d; }
        );
}

#endif

//Bench scalar operations
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a + 1.25 (s) [std][scalar][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a + 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r = a - 1.25 (s) [std][scalar][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a - 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r = a * 1.25 (s) [std][scalar][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a >> 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r = a / 1.25 (s) [std][scalar][s]",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a / 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r += 1.25 (s) [std][scalar][s]",
        [](size_t d){ return std::make_tuple(svec(d)); },
        [](svec& r){ r += 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r -= 1.25 (s) [std][scalar][s]",
        [](size_t d){ return std::make_tuple(svec(d)); },
        [](svec& r){ r -= 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r *= 1.25 (s) [std][scalar][s]",
        [](size_t d){ return std::make_tuple(svec(d)); },
        [](svec& r){ r *= 1.25f; }
        );

    CPM_TWO_PASS_NS(
        "r /= 1.25 (s) [std][scalar][s]",
        [](size_t d){ return std::make_tuple(svec(d)); },
        [](svec& r){ r /= 1.25f; }
        );
}

//Bench activation functions
CPM_BENCH() {
    CPM_TWO_PASS_NS("nn_relu",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = relu(a); },
        [](size_t d){ return d; }
    );

    CPM_TWO_PASS_NS("nn_relu_der",
        [](size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = relu_derivative(a); },
        [](size_t d){ return d; }
    );
}
