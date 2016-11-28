//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

#ifdef ETL_THESIS_BENCH

/* GEMM Benchmark */

using thesis_gemm_policy = NARY_POLICY(
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000),
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));

using thesis_small_gemm_policy = NARY_POLICY(
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500),
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500));

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_sgemm [thesis]", thesis_gemm_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_dgemm [thesis]", thesis_gemm_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dmat& a, dmat& b, dmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_cgemm [thesis]", thesis_small_gemm_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 6 * 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1,d2), cmat(d1,d2), cmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cmat& a, cmat& b, cmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_zgemm [thesis]", thesis_small_gemm_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 6 * 2 * d1 * d2 * d2; }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1,d2), zmat(d1,d2), zmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::STD, a * b); })
    VEC_SECTION_FUNCTOR("vec", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::VEC, a * b); })
    BLAS_SECTION_FUNCTOR("blas", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::BLAS, a * b); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zmat& a, zmat& b, zmat& c){ c = selected_helper(etl::gemm_impl::CUBLAS, a * b); })
)

/* FFT */

using thesis_fft_policy = NARY_POLICY(
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000),
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000));

using thesis_fft_many_policy = NARY_POLICY(
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200),
    VALUES_POLICY(10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200));

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_cfft [thesis]", thesis_fft_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1,d2), cmat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_2d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_2d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_zfft [thesis]", thesis_fft_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1,d2), zmat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_2d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_2d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_cfft_many [thesis]", thesis_fft_many_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * 512 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat3(512UL, d1,d2), cmat3(512UL, d1,d2)); }),
    CPM_SECTION_FUNCTOR("std", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_2d_many(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_2d_many(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d_many(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_zfft_many [thesis]", thesis_fft_many_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * 512 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat3(512UL, d1,d2), zmat3(512UL, d1,d2)); }),
    CPM_SECTION_FUNCTOR("std", [](zmat3& a, zmat3& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_2d_many(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat3& a, zmat3& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_2d_many(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zmat3& a, zmat3& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d_many(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_cifft [thesis]", thesis_fft_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1,d2), cmat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::STD, etl::ifft_2d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::MKL, etl::ifft_2d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::ifft_2d(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("thesis_zifft [thesis]", thesis_fft_policy,
    FLOPS([](std::size_t d1, std::size_t d2){ return 2 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1,d2), zmat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::STD, etl::ifft_2d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::MKL, etl::ifft_2d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::ifft_2d(a)); })
)

/* Batch Valid Convolution */

/* Batch Full Convolution */

#endif
