//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#define CPM_LIB
#include "benchmark.hpp"

CPM_DIRECT_SECTION_TWO_PASS_NS_P("cfft_1d(2^b)", fft_1d_policy_2,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(cvec(d), cvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cvec& b){ b = etl::fft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cvec& b){ etl::impl::standard::fft1(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](cvec& a, cvec& b){ etl::impl::blas::fft1(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cvec& a, cvec& b){ etl::impl::cufft::fft1(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("zfft_1d(2^b)", fft_1d_policy_2,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(zvec(d), zvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zvec& b){ b = etl::fft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zvec& b){ etl::impl::standard::fft1(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](zvec& a, zvec& b){ etl::impl::blas::fft1(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zvec& a, zvec& b){ etl::impl::cufft::fft1(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("cfft_1d(10^b)", fft_1d_policy,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(cvec(d), cvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cvec& b){ b = etl::fft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cvec& b){ etl::impl::standard::fft1(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](cvec& a, cvec& b){ etl::impl::blas::fft1(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cvec& a, cvec& b){ etl::impl::cufft::fft1(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("zfft_1d(10^b)", fft_1d_policy,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(zvec(d), zvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zvec& b){ b = etl::fft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zvec& b){ etl::impl::standard::fft1(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](zvec& a, zvec& b){ etl::impl::blas::fft1(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zvec& a, zvec& b){ etl::impl::cufft::fft1(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("cifft_1d(2^b)", fft_1d_policy_2,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(cvec(d), cvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cvec& b){ b = etl::ifft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cvec& b){ etl::impl::standard::ifft1(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](cvec& a, cvec& b){ etl::impl::blas::ifft1(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cvec& a, cvec& b){ etl::impl::cufft::ifft1(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("zifft_1d(2^b)", fft_1d_policy_2,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(zvec(d), zvec(d)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zvec& b){ b = etl::ifft_1d(a); }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zvec& b){ etl::impl::standard::ifft1(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](zvec& a, zvec& b){ etl::impl::blas::ifft1(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zvec& a, zvec& b){ etl::impl::cufft::ifft1(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("fft_1d_many(1000) (c)", fft_1d_many_policy,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(cmat(1000UL, d), cmat(1000UL, d)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cmat& b){ b = etl::fft_1d_many(a); }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b){ etl::impl::standard::fft1_many(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat& a, cmat& b){ etl::impl::blas::fft1_many(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat& a, cmat& b){ etl::impl::cufft::fft1_many(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("fft_1d_many(1000) (z)", fft_1d_many_policy,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(zmat(1000UL, d), zmat(1000UL, d)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zmat& b){ b = etl::fft_1d_many(a); }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b){ etl::impl::standard::fft1_many(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat& a, zmat& b){ etl::impl::blas::fft1_many(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zmat& a, zmat& b){ etl::impl::cufft::fft1_many(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("cfft_2d(2^b)", fft_2d_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1,d2), cmat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cmat& b){ b = etl::fft_2d(a); }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b){ etl::impl::standard::fft2(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](cmat& a, cmat& b){ etl::impl::blas::fft2(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](cmat& a, cmat& b){ etl::impl::cufft::fft2(a, b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("zfft_2d(2^b)", fft_2d_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1,d2), zmat(d1,d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zmat& b){ b = etl::fft_2d(a); }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b){ etl::impl::standard::fft2(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat& a, zmat& b){ etl::impl::blas::fft2(a, b); })
    CUFFT_SECTION_FUNCTOR("cufft", [](zmat& a, zmat& b){ etl::impl::cufft::fft2(a, b); })
)
