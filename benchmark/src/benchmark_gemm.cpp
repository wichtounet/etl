//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

namespace {

float float_ref = 0.0;

#ifdef ETL_EXTENDED_BENCH
double double_ref = 0.0;
#endif

} //end of anonymous namespace

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (s) [gemm]", gemm_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (d) [gemm]", gemm_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (c) [gemm]", small_square_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 6 * 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1,d2), cmat(d1,d2), cmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cmat& b, cmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (z) [gemm]", small_square_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 6 * 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1,d2), zmat(d1,d2), zmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zmat& b, zmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (cm/s) [gemm]", square_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_cm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (s) [gemm]", gemv_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d2), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, svec& b, svec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (d) [gemm]", gemv_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d2), dvec(d2), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dvec& b, dvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dvec& b, dvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](dmat& a, dvec& b, dvec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](dmat& a, dvec& b, dvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dmat& a, dvec& b, dvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (c) [gemm]", gemv_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 6 * 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1,d2), cvec(d2), cvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cvec& b, cvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cvec& b, cvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](cmat& a, cvec& b, cvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cmat& a, cvec& b, cvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (z) [gemm]", gemv_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 6 * 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1,d2), zvec(d2), zvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zvec& b, zvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zvec& b, zvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](zmat& a, zvec& b, zvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zmat& a, zvec& b, zvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (s) [gemm]", gemv_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), smat(d1,d2), svec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, smat& b, svec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (d) [gemm]", gemv_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dmat& b, dvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dmat& b, dvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](dvec& a, dmat& b, dvec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](dvec& a, dmat& b, dvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dvec& a, dmat& b, dvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (c) [gemm]", gemv_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 6 * 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cvec(d1), cmat(d1,d2), cvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cmat& b, cvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cmat& b, cvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](cvec& a, cmat& b, cvec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](cvec& a, cmat& b, cvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cvec& a, cmat& b, cvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (z) [gemm]", gemv_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 6 * 2 * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zvec(d1), zmat(d1,d2), zvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zmat& b, zvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zmat& b, zvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](zvec& a, zmat& b, zvec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](zvec& a, zmat& b, zvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zvec& a, zmat& b, zvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)
#endif

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000),
        "r = A * (a + b) [gemm]",
        [](std::size_t d){ return std::make_tuple(dmat(d,d), dvec(d), dvec(d), dvec(d)); },
        [](dmat& A, dvec& a, dvec& b, dvec& r){ r = A * (a + b); }
        , [](std::size_t d){ return d + 2 * d * d; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000),
        "r = A * (a + b + c) [gemm]",
        [](std::size_t d){ return std::make_tuple(dmat(d,d), dvec(d), dvec(d), dvec(d), dvec(d)); },
        [](dmat& A, dvec& a, dvec& b, dvec& c, dvec& r){ r = A * (a + b + c); }
        , [](std::size_t d){ return 2 * d + 2 * d * d; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = A * (B + C) [gemm]",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& R){ R = A * (B + C); }
        , [](std::size_t d1, std::size_t d2){ return d1 * d2  + 2 * d1 * d2 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = A * (B + C + D) [gemm]",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& D, dmat& R){ R = A * (B + C + D); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2  + 2 * d1 * d2 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = (A + B) * (C + D) [gemm]",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& D, dmat& R){ R = (A + B) * (C + D); }
        , [](std::size_t d1, std::size_t d2){ return d1 * d2  + d1 * d2 + 2 * d1 * d2 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * (A + B - C) [gemm]",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dmat& C, dvec& r){ r = a * (A + B - C); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 + 2 * d1 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * (A * B) [gemm][order]",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dvec& r){ r = a * (A * B); }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 + 2 * d1 * d2 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * A * B [gemm][order]",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dvec& r){ r = a * A * B; }
        , [](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 + 2 * d1 * d2; }
        );
}

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r = a *o b (s)", outer_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, smat& c){ c = etl::outer(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, smat& c){ c = selected_helper(etl::outer_impl::STD, etl::outer(a, b)); })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, svec& b, smat& c){ c = selected_helper(etl::outer_impl::BLAS, etl::outer(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r = batch_outer(a,b) (s)", outer_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 128UL * d1 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(128UL, d1), smat(128UL, d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = etl::batch_outer(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::outer_impl::STD, etl::batch_outer(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::outer_impl::VEC, etl::batch_outer(a, b)); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::outer_impl::BLAS, etl::batch_outer(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r = a dot b (s)", dot_policy,
    FLOPS([](std::size_t d1){ return 2 * d1; }),
    CPM_SECTION_INIT([](std::size_t d1){ return std::make_tuple(svec(d1), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b){ float_ref += etl::dot(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b){ SELECTED_SECTION(etl::dot_impl::STD) { float_ref += etl::dot(a, b); } })
    VEC_SECTION_FUNCTOR("vec", [](svec& a, svec& b){ SELECTED_SECTION(etl::dot_impl::VEC) { float_ref += etl::dot(a, b); } })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, svec& b){ SELECTED_SECTION(etl::dot_impl::BLAS) { float_ref += etl::dot(a, b); } })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r = a dot b (d)", dot_policy,
    FLOPS([](std::size_t d1){ return 2 * d1; }),
    CPM_SECTION_INIT([](std::size_t d1){ return std::make_tuple(dvec(d1), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b){ double_ref += etl::dot(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b){ SELECTED_SECTION(etl::dot_impl::STD) { double_ref += etl::dot(a, b); } })
    VEC_SECTION_FUNCTOR("vec", [](dvec& a, dvec& b){ SELECTED_SECTION(etl::dot_impl::VEC) { double_ref += etl::dot(a, b); } })
    BLAS_SECTION_FUNCTOR("blas", [](dvec& a, dvec& b){ SELECTED_SECTION(etl::dot_impl::BLAS) { double_ref += etl::dot(a, b); } })
)
#endif
