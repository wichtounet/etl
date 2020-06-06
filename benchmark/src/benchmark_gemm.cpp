//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"
#include "benchmark_gemm.hpp"

namespace {

float float_ref = 0.0;

#ifdef ETL_EXTENDED_BENCH
double double_ref = 0.0;
#endif

} //end of anonymous namespace

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("2.5f * (A * B) (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = 2.5f * (a * b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::STD, 2.5f * (a * b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::VEC, 2.5f * (a * b)); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::BLAS, 2.5f * (a * b)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, 2.5f * (a * b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (d) [gemm]", gemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (c) [gemm]", small_square_policy,
    FLOPS([](size_t d1, size_t d2){ return 6 * 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(cmat(d1,d2), cmat(d1,d2), cmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cmat& b, cmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

#ifdef ETL_EXTENDED_BENCH

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("RM = CM * RM (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_rm(d1,d2), smat_rm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("RM = RM * CM (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_rm(d1,d2), smat_rm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_rm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("RM = CM * CM (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_cm(d1,d2), smat_rm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_cm& b, smat_rm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_cm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, smat_cm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_cm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_cm& b, smat_rm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("CM = RM * CM (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_rm(d1,d2), smat_cm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_rm& a, smat_cm& b, smat_cm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_rm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_rm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_rm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_rm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("CM = CM * RM (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_rm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_rm& b, smat_cm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_rm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, smat_rm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_rm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_rm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("CM = RM * RM (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_rm(d1,d2), smat_rm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_rm& a, smat_rm& b, smat_cm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_rm& a, smat_rm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_rm& a, smat_rm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_rm& a, smat_rm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_rm& a, smat_rm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (z) [gemm]", small_square_policy,
    FLOPS([](size_t d1, size_t d2){ return 6 * 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(zmat(d1,d2), zmat(d1,d2), zmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zmat& b, zmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * trans(B) (s) [gemm]", gemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = a * transpose(b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::STD, a * transpose(b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::VEC, a * transpose(b)); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * transpose(b)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * transpose(b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * trans(B) (cm/s) [gemm]", gemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_cm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = a * transpose(b); }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::STD, a * transpose(b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::VEC, a * transpose(b)); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * transpose(b)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * transpose(b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("trans(A) * B (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = transpose(a) * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::STD, transpose(a) * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::VEC, transpose(a) * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("trans(A) * B (cm/s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_cm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = transpose(a) * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::STD, transpose(a) * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::VEC, transpose(a) * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("trans(A) * trans(B) (s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = transpose(a) * transpose(b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::STD, transpose(a) * transpose(b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::VEC, transpose(a) * transpose(b)); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * transpose(b)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * transpose(b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("trans(A) * trans(B) (cm/s) [gemm]", sgemm_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_cm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = transpose(a) * transpose(b); }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::STD, transpose(a) * transpose(b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::VEC, transpose(a) * transpose(b)); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * transpose(b)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * transpose(b)); })
)

#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * B (cm/s) [gemm]", square_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_cm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1,d2), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, svec& b, svec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (d) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(dmat(d1,d2), dvec(d2), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dvec& b, dvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dvec& b, dvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](dmat& a, dvec& b, dvec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](dmat& a, dvec& b, dvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dmat& a, dvec& b, dvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (cm/s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d1,d2), svec_cm(d2), svec_cm(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, svec_cm& b, svec_cm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, svec_cm& b, svec_cm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, svec_cm& b, svec_cm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, svec_cm& b, svec_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, svec_cm& b, svec_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

#ifdef ETL_EXTENDED_BENCH

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("trans(A) * x (s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d2,d1), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, svec& b, svec& c){ c = transpose(a) * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::STD, transpose(a) * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::VEC, transpose(a) * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("trans(A) * x (cm/s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat_cm(d2,d1), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, svec& b, svec& c){ c = transpose(a) * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::STD, transpose(a) * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat_cm& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::VEC, transpose(a) * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::BLAS, transpose(a) * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, svec& b, svec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, transpose(a) * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (c) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 6 * 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(cmat(d1,d2), cvec(d2), cvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cvec& b, cvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cvec& b, cvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](cmat& a, cvec& b, cvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cmat& a, cvec& b, cvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("A * x (z) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 6 * 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(zmat(d1,d2), zvec(d2), zvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zvec& b, zvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zvec& b, zvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](zmat& a, zvec& b, zvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zmat& a, zvec& b, zvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(svec(d1), smat(d1,d2), svec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, smat& b, svec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (d) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dmat& b, dvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dmat& b, dvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](dvec& a, dmat& b, dvec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](dvec& a, dmat& b, dvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dvec& a, dmat& b, dvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (cm/s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(svec_cm(d1), smat_cm(d1,d2), svec_cm(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](svec_cm& a, smat_cm& b, svec_cm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](svec_cm& a, smat_cm& b, svec_cm& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](svec_cm& a, smat_cm& b, svec_cm& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](svec_cm& a, smat_cm& b, svec_cm& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](svec_cm& a, smat_cm& b, svec_cm& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

#ifdef ETL_EXTENDED_BENCH

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * trans(A) (s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(svec(d1), smat(d2, d1), svec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, smat& b, svec& c){ c = a * transpose(b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::STD, a * transpose(b)); })
    VEC_SECTION_FUNCTOR("vec", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::VEC, a * transpose(b)); })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * transpose(b)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](svec& a, smat& b, svec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * transpose(b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * trans(A) (cm/s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(svec(d1), smat_cm(d2, d1), svec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, smat_cm& b, svec& c){ c = a * transpose(b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, smat_cm& b, svec& c){ c = selected_helper(etl::gemm_impl::STD, a * transpose(b)); })
    VEC_SECTION_FUNCTOR("vec", [](svec& a, smat_cm& b, svec& c){ c = selected_helper(etl::gemm_impl::VEC, a * transpose(b)); })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, smat_cm& b, svec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * transpose(b)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](svec& a, smat_cm& b, svec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * transpose(b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (c) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 6 * 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(cvec(d1), cmat(d1,d2), cvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cmat& b, cvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cmat& b, cvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](cvec& a, cmat& b, cvec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](cvec& a, cmat& b, cvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cvec& a, cmat& b, cvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x * A (z) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 6 * 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(zvec(d1), zmat(d1,d2), zvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zmat& b, zvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zmat& b, zvec& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](zvec& a, zmat& b, zvec& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](zvec& a, zmat& b, zvec& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zvec& a, zmat& b, zvec& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

size_t special_n = 1;

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("x(n) * A (s) [gemm]", gemv_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(2UL,d1), smat(d1,d2), smat(2UL,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c(special_n) = a(special_n) * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c(special_n) = selected_helper(etl::gemm_impl::STD, a(special_n) * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c(special_n) = selected_helper(etl::gemm_impl::VEC, a(special_n) * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c(special_n) = selected_helper(etl::gemm_impl::BLAS, a(special_n) * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c(special_n) = selected_helper(etl::gemm_impl::CUBLAS, a(special_n) * b); })
)
#endif

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000),
        "r = A * (a + b) [gemm]",
        [](size_t d){ return std::make_tuple(dmat(d,d), dvec(d), dvec(d), dvec(d)); },
        [](dmat& A, dvec& a, dvec& b, dvec& r){ r = A * (a + b); }
        , [](size_t d){ return d + 2 * d * d; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000),
        "r = A * (a + b + c) [gemm]",
        [](size_t d){ return std::make_tuple(dmat(d,d), dvec(d), dvec(d), dvec(d), dvec(d)); },
        [](dmat& A, dvec& a, dvec& b, dvec& c, dvec& r){ r = A * (a + b + c); }
        , [](size_t d){ return 2 * d + 2 * d * d; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = A * (B + C) [gemm]",
        [](size_t d1,size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& R){ R = A * (B + C); }
        , [](size_t d1, size_t d2){ return d1 * d2  + 2 * d1 * d2 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = A * (B + C + D) [gemm]",
        [](size_t d1,size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& D, dmat& R){ R = A * (B + C + D); }
        , [](size_t d1, size_t d2){ return 2 * d1 * d2  + 2 * d1 * d2 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = (A + B) * (C + D) [gemm]",
        [](size_t d1,size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& D, dmat& R){ R = (A + B) * (C + D); }
        , [](size_t d1, size_t d2){ return d1 * d2  + d1 * d2 + 2 * d1 * d2 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * (A + B - C) [gemm]",
        [](size_t d1,size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dmat& C, dvec& r){ r = a * (A + B - C); }
        , [](size_t d1, size_t d2){ return 2 * d1 * d2 + 2 * d1 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * (A * B) [gemm][order]",
        [](size_t d1,size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dvec& r){ r = a * (A * B); }
        , [](size_t d1, size_t d2){ return 2 * d1 * d2 + 2 * d1 * d2 * d2; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * A * B [gemm][order]",
        [](size_t d1,size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dvec& r){ r = a * A * B; }
        , [](size_t d1, size_t d2){ return 2 * d1 * d2 + 2 * d1 * d2; }
        );
}

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r = a *o b (s) [outer]", outer_policy,
    FLOPS([](size_t d1, size_t d2){ return d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(svec(d1), svec(d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, smat& c){ c = etl::outer(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, smat& c){ c = selected_helper(etl::outer_impl::STD, etl::outer(a, b)); })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, svec& b, smat& c){ c = selected_helper(etl::outer_impl::BLAS, etl::outer(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r = batch_outer(a,b) (s) [batch_outer]", outer_policy,
    FLOPS([](size_t d1, size_t d2){ return 128UL * d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(128UL, d1), smat(128UL, d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = etl::batch_outer(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::outer_impl::STD, etl::batch_outer(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::outer_impl::VEC, etl::batch_outer(a, b)); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::outer_impl::BLAS, etl::batch_outer(a, b)); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::outer_impl::CUBLAS, etl::batch_outer(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sbias_add", bias_add_policy,
    FLOPS([](size_t d1, size_t d2, size_t d3, size_t d4){ return d1 * d2 * d3 * d4; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2, size_t d3, size_t d4){ return std::make_tuple(smat4(d1, d2, d3, d4), svec(d2), smat4(d1, d2, d3, d4)); }),
    CPM_SECTION_FUNCTOR("default", [](smat4& a, svec& b, smat4& c){ c = etl::bias_add_4d(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat4& a, svec& b, smat4& c){ c = selected_helper(etl::bias_add_impl::STD, etl::bias_add_4d(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat4& a, svec& b, smat4& c){ c = selected_helper(etl::bias_add_impl::VEC, etl::bias_add_4d(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("sbias_add_2d", bias_add_2d_policy,
    FLOPS([](size_t d1, size_t d2){ return d1 * d2; }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(smat(d1, d2), svec(d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, svec& b, smat& c){ c = etl::bias_add_2d(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, svec& b, smat& c){ c = selected_helper(etl::bias_add_impl::STD, etl::bias_add_2d(a, b)); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, svec& b, smat& c){ c = selected_helper(etl::bias_add_impl::VEC, etl::bias_add_2d(a, b)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r = a dot b (s)", dot_policy,
    FLOPS([](size_t d1){ return 2 * d1; }),
    CPM_SECTION_INIT([](size_t d1){ return std::make_tuple(svec(d1), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b){ float_ref += etl::dot(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b){ SELECTED_SECTION(etl::dot_impl::STD) { float_ref += etl::dot(a, b); } })
    VEC_SECTION_FUNCTOR("vec", [](svec& a, svec& b){ SELECTED_SECTION(etl::dot_impl::VEC) { float_ref += etl::dot(a, b); } })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, svec& b){ SELECTED_SECTION(etl::dot_impl::BLAS) { float_ref += etl::dot(a, b); } })
    CUBLAS_SECTION_FUNCTOR("cublas", [](svec& a, svec& b){ SELECTED_SECTION(etl::dot_impl::CUBLAS) { float_ref += etl::dot(a, b); } })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("r = a dot b (d)", dot_policy,
    FLOPS([](size_t d1){ return 2 * d1; }),
    CPM_SECTION_INIT([](size_t d1){ return std::make_tuple(dvec(d1), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b){ double_ref += etl::dot(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b){ SELECTED_SECTION(etl::dot_impl::STD) { double_ref += etl::dot(a, b); } })
    VEC_SECTION_FUNCTOR("vec", [](dvec& a, dvec& b){ SELECTED_SECTION(etl::dot_impl::VEC) { double_ref += etl::dot(a, b); } })
    BLAS_SECTION_FUNCTOR("blas", [](dvec& a, dvec& b){ SELECTED_SECTION(etl::dot_impl::BLAS) { double_ref += etl::dot(a, b); } })
)
#endif
