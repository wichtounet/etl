//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("cfft_1d(2^b) [fft]", fft_1d_policy_2,
    FLOPS([](size_t d){ return 2 * d * std::log2(d); }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(cvec(d), cvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cvec& b){ b = etl::fft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_1d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_1d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_1d(a)); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("zfft_1d(2^b) [fft]", fft_1d_policy_2,
    FLOPS([](size_t d){ return 2 * d * std::log2(d); }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(zvec(d), zvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zvec& b){ b = etl::fft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_1d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_1d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_1d(a)); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("cfft_1d(10^b) [fft]", fft_1d_policy,
    FLOPS([](size_t d){ return 2 * d * std::log2(d); }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(cvec(d), cvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cvec& b){ b = etl::fft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_1d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_1d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_1d(a)); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("zfft_1d(10^b) [fft]", fft_1d_policy,
    FLOPS([](size_t d){ return 2 * d * std::log2(d); }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(zvec(d), zvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zvec& b){ b = etl::fft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_1d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_1d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_1d(a)); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("cifft_1d(2^b) [fft]", fft_1d_policy_2,
    FLOPS([](size_t d){ return 2 * d * std::log2(d); }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(cvec(d), cvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cvec& b){ b = etl::ifft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::STD, etl::ifft_1d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::MKL, etl::ifft_1d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cvec& a, cvec& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::ifft_1d(a)); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("zifft_1d(2^b) [fft]", fft_1d_policy_2,
    FLOPS([](size_t d){ return 2 * d * std::log2(d); }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(zvec(d), zvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zvec& b){ b = etl::ifft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::STD, etl::ifft_1d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::MKL, etl::ifft_1d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zvec& a, zvec& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::ifft_1d(a)); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("fft_1d_many(1000) (c) [fft]", fft_1d_many_policy,
    FLOPS([](size_t d){ return 2 * 1000 * d * std::log2(d); }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(cmat(1000UL, d), cmat(1000UL, d)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cmat& b){ b = etl::fft_1d_many(a); }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_1d_many(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_1d_many(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_1d_many(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("fft_1d_many(1000) (z) [fft]", fft_1d_many_policy,
    FLOPS([](size_t d){ return 2 * 1000 * d * std::log2(d); }),
    CPM_SECTION_INIT([](size_t d){ return std::make_tuple(zmat(1000UL, d), zmat(1000UL, d)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zmat& b){ b = etl::fft_1d_many(a); }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_1d_many(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_1d_many(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_1d_many(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("cfft_2d(2^b) [fft]", fft_2d_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(cmat(d1,d2), cmat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cmat& b){ b = etl::fft_2d(a); }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_2d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_2d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat& a, cmat& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d(a)); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("zfft_2d(2^b) [fft]", fft_2d_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(zmat(d1,d2), zmat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zmat& b){ b = etl::fft_2d(a); }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_2d(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_2d(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zmat& a, zmat& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d(a)); })
)
#endif

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("cfft_2d_many (512) [fft]", fft_2d_many_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * 512 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(cmat3(512UL, d1,d2), cmat3(512UL, d1,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat3& a, cmat3& b){ b = etl::fft_2d_many(a); }),
    CPM_SECTION_FUNCTOR("std", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_2d_many(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_2d_many(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d_many(a)); })
)

#ifdef ETL_EXTENDED_BENCH
CPM_DIRECT_SECTION_TWO_PASS_NS_PF("zfft_2d_many (512) [fft]", fft_2d_many_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * 512 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(zmat3(512UL, d1,d2), zmat3(512UL, d1,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat3& a, zmat3& b){ b = etl::fft_2d_many(a); }),
    CPM_SECTION_FUNCTOR("std", [](zmat3& a, zmat3& b){ b = selected_helper(etl::fft_impl::STD, etl::fft_2d_many(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat3& a, zmat3& b){ b = selected_helper(etl::fft_impl::MKL, etl::fft_2d_many(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zmat3& a, zmat3& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::fft_2d_many(a)); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_PF("cifft_2d_many (512) [fft]", fft_2d_many_policy,
    FLOPS([](size_t d1, size_t d2){ return 2 * 512 * d1 * d2 * std::log2(d1 * d2); }),
    CPM_SECTION_INIT([](size_t d1, size_t d2){ return std::make_tuple(cmat3(512UL, d1,d2), cmat3(512UL, d1,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat3& a, cmat3& b){ b = etl::ifft_2d_many(a); }),
    CPM_SECTION_FUNCTOR("std", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::STD, etl::ifft_2d_many(a)); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::MKL, etl::ifft_2d_many(a)); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat3& a, cmat3& b){ b = selected_helper(etl::fft_impl::CUFFT, etl::ifft_2d_many(a)); })
)

#endif
