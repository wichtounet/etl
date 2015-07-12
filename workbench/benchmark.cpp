//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#include "etl/etl.hpp"

#define CPM_PARALLEL_RANDOMIZE
#define CPM_FAST_RANDOMIZE

#define CPM_WARMUP 3
#define CPM_REPEAT 10

#define CPM_BENCHMARK "Tests Benchmarks"
#include "cpm/cpm.hpp"

#ifdef ETL_VECTORIZE_IMPL
#ifdef __SSE3__
#define TEST_SSE
#endif
#ifdef __AVX__
#define TEST_AVX
#endif
#endif

#ifdef ETL_MKL_MODE
#define TEST_MKL
#endif

#ifdef ETL_BLAS_MODE
#define TEST_BLAS
#endif

#ifdef ETL_BENCH_STRASSEN
#define TEST_STRASSEN
#endif

#ifdef ETL_BENCH_MMUL_CONV
#define TEST_MMUL_CONV
#endif

using dvec = etl::dyn_vector<double>;
using dmat = etl::dyn_matrix<double>;
using dmat2 = etl::dyn_matrix<double, 2>;
using dmat3 = etl::dyn_matrix<double, 3>;
using dmat4 = etl::dyn_matrix<double, 4>;

using svec = etl::dyn_vector<float>;
using smat = etl::dyn_matrix<float>;

using cvec = etl::dyn_vector<std::complex<float>>;
using cmat = etl::dyn_matrix<std::complex<float>>;

using zvec = etl::dyn_vector<std::complex<double>>;
using zmat = etl::dyn_matrix<std::complex<double>>;

using mat_policy = VALUES_POLICY(10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000);
using mat_policy_2d = NARY_POLICY(mat_policy, mat_policy);

using conv_1d_large_policy = NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000));
using conv_2d_large_policy = NARY_POLICY(VALUES_POLICY(100, 105, 110, 115, 120, 125, 130, 135, 140), VALUES_POLICY(50, 50, 55, 55, 60, 60, 65, 65, 70));

using fft_1d_policy = VALUES_POLICY(10, 100, 1000, 10000, 100000, 1000000, 10000000);
using fft_2d_policy = NARY_POLICY(VALUES_POLICY(200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000), VALUES_POLICY(200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000));

using sigmoid_policy = VALUES_POLICY(250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500);

#ifdef TEST_SSE
#define SSE_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define SSE_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_AVX
#define AVX_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define AVX_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_MKL
#define MKL_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define MKL_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_MMUL_CONV
#define MC_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define MC_SECTION_FUNCTOR(name, ...)
#endif

//Bench addition
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a + b",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b + c",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& c, dvec& r){ r = a + b + c; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = A + B",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& R){ R = A + B; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R += A",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& R){ R += A; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R += A + B",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& R){ R += A + B; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = A + B + C",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& R){ R = A + B + C; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (c)",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a + b; }
        );

    CPM_TWO_PASS_NS(
        "r = a + b (z)",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a + b; }
        );
}

//Bench subtraction
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a - b",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a - b; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = A - B",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& R){ R = A - B; }
        );
}

//Bench multiplication
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a >> b",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (c)",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS(
        "r = a >> b (z)",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a >> b; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = A >> B",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& R){ R = A >> B; }
        );
}

//Bench division
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a / b",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d), dvec(d)); },
        [](dvec& a, dvec& b, dvec& r){ r = a / b; }
        );

    CPM_TWO_PASS_NS(
        "r = a / b (c)",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d), cvec(d)); },
        [](cvec& a, cvec& b, cvec& r){ r = a / b; }
        );

    CPM_TWO_PASS_NS(
        "r = a / b (z)",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d), zvec(d)); },
        [](zvec& a, zvec& b, zvec& r){ r = a / b; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = A / B",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& R){ R = A / B; }
        );
}

//Sigmoid benchmark
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = sigmoid(a)",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = etl::sigmoid(a); }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = sigmoid(A)",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& R){ R = etl::sigmoid(A); }
        );
}

//1D-Convolution benchmarks with large-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000)),
        "r = conv_1d_full(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000)),
        "r = conv_1d_same(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000)),
        "r = conv_1d_valid(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); }
        );
}

//1D-Convolution benchmarks with small-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200)),
        "r = conv_1d_full(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200)),
        "r = conv_1d_same(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200)),
        "r = conv_1d_valid(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); }
        );
}

//2D-Convolution benchmarks with large-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 110, 120, 130, 140, 150, 160, 170, 180), VALUES_POLICY(50, 55, 60, 65, 70, 75, 80, 85, 90)),
        "r = conv_2d_full(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 110, 120, 130, 140, 150, 160, 170, 180), VALUES_POLICY(50, 55, 60, 65, 70, 75, 80, 85, 90)),
        "r = conv_2d_same(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1, d1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 110, 120, 130, 140, 150, 160, 170, 180), VALUES_POLICY(50, 55, 60, 65, 70, 75, 80, 85, 90)),
        "r = conv_2d_valid(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }
        );
}

//2D-Convolution benchmarks with small-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40, 45)),
        "r = conv_2d_full(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40, 45)),
        "r = conv_2d_same(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1, d1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40, 45)),
        "r = conv_2d_valid(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }
        );
}

CPM_DIRECT_BENCH_TWO_PASS_P(
    NARY_POLICY(VALUES_POLICY(16, 16, 32, 32, 64, 64), VALUES_POLICY(4, 8, 8, 16, 16, 24)),
    "convmtx2",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat((d1 + d2 - 1)*(d1 + d2 - 1), d2 * d2)); }, [](std::size_t /*d1*/, std::size_t d2, dmat& a, dmat& b){ b = etl::convmtx2(a, d2, d2); }
)

CPM_DIRECT_BENCH_TWO_PASS_P(
    NARY_POLICY(VALUES_POLICY(16, 16, 32, 32, 64, 64, 128), VALUES_POLICY(4, 8, 8, 16, 16, 32, 32)),
    "convmtx2_t",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat((d1 + d2 - 1)*(d1 + d2 - 1), d2 * d2)); },
    [](std::size_t /*d1*/, std::size_t d2, dmat& a, dmat& b){ etl::convmtx2_direct_t(b, a, d2, d2); }
)

CPM_DIRECT_BENCH_TWO_PASS_P(
    NARY_POLICY(VALUES_POLICY(16, 16, 32, 32, 64, 64, 128), VALUES_POLICY(4, 8, 8, 16, 16, 32, 32)),
    "im2col_direct",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2 * d2, (d1 - d2 + 1)*(d1 - d2 + 1))); },
    [](std::size_t /*d1*/, std::size_t d2, dmat& a, dmat& b){ etl::im2col_direct(b, a, d2, d2); }
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv1_valid", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ etl::impl::standard::conv1_valid(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ etl::impl::sse::sconv1_valid(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ etl::impl::avx::sconv1_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv1_valid", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ etl::impl::standard::conv1_valid(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ etl::impl::sse::dconv1_valid(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ etl::impl::avx::dconv1_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv1_same", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ etl::impl::standard::conv1_same(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ etl::impl::sse::sconv1_same(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ etl::impl::avx::sconv1_same(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv1_same", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ etl::impl::standard::conv1_same(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ etl::impl::sse::dconv1_same(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ etl::impl::avx::dconv1_same(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv1_full", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ etl::impl::standard::conv1_full(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ etl::impl::sse::sconv1_full(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ etl::impl::avx::sconv1_full(a, b, r); })
    MKL_SECTION_FUNCTOR("fft", [](svec& a, svec& b, svec& r){ r = etl::fft_conv_1d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](svec& a, svec& b, svec& r){ etl::impl::reduc::conv1_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv1_full", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ etl::impl::standard::conv1_full(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ etl::impl::sse::dconv1_full(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ etl::impl::avx::dconv1_full(a, b, r); })
    MKL_SECTION_FUNCTOR("fft", [](dvec& a, dvec& b, dvec& r){ r = etl::fft_conv_1d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](dvec& a, dvec& b, dvec& r){ etl::impl::reduc::conv1_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv2_valid", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ etl::impl::standard::conv2_valid(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ etl::impl::sse::sconv2_valid(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ etl::impl::avx::sconv2_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv2_valid", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ etl::impl::standard::conv2_valid(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ etl::impl::sse::dconv2_valid(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ etl::impl::avx::dconv2_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv2_same", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ etl::impl::standard::conv2_same(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ etl::impl::sse::sconv2_same(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ etl::impl::avx::sconv2_same(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv2_same", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ etl::impl::standard::conv2_same(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ etl::impl::sse::dconv2_same(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ etl::impl::avx::dconv2_same(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv2_full", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ etl::impl::standard::conv2_full(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ etl::impl::sse::sconv2_full(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ etl::impl::avx::sconv2_full(a, b, r); })
    MKL_SECTION_FUNCTOR("fft", [](smat& a, smat& b, smat& r){ r = etl::fft_conv_2d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](smat& a, smat& b, smat& r){ etl::impl::reduc::conv2_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv2_full", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ etl::impl::standard::conv2_full(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ etl::impl::sse::dconv2_full(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ etl::impl::avx::dconv2_full(a, b, r); })
    MKL_SECTION_FUNCTOR("fft", [](dmat& a, dmat& b, dmat& r){ r = etl::fft_conv_2d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](dmat& a, dmat& b, dmat& r){ etl::impl::reduc::conv2_full(a, b, r); })
)

using square_policy = NARY_POLICY(VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = A * B (float)",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1,d2)); },
        [](smat& A, smat& B, smat& R){ R = A * B; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = A * B (double)",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& R){ R = A * B; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000),
        "r = A * (a + b)",
        [](std::size_t d){ return std::make_tuple(dmat(d,d), dvec(d), dvec(d), dvec(d)); },
        [](dmat& A, dvec& a, dvec& b, dvec& r){ r = A * (a + b); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000),
        "r = A * (a + b + c)",
        [](std::size_t d){ return std::make_tuple(dmat(d,d), dvec(d), dvec(d), dvec(d), dvec(d)); },
        [](dmat& A, dvec& a, dvec& b, dvec& c, dvec& r){ r = A * (a + b + c); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = A * (B + C)",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& R){ R = A * (B + C); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = A * (B + C + D)",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& D, dmat& R){ R = A * (B + C + D); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "R = (A + B) * (C + D)",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& D, dmat& R){ R = (A + B) * (C + D); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * (A + B - C)",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dmat& C, dvec& r){ r = a * (A + B - C); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * (A * B)",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dvec& r){ r = a * (A * B); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        square_policy,
        "r = a * A * B",
        [](std::size_t d1,std::size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dmat(d1,d2), dvec(d2)); },
        [](dvec& a, dmat& A, dmat& B, dvec& r){ r = a * (A * B); }
        );
}

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(100, 500, 550, 600, 1000, 2000, 2500), VALUES_POLICY(100, 120, 500, 1000, 1200, 2000, 5000)),
    "rbm_hidden",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d2), dvec(d2), dmat(d1, d2));},
    [](dvec& v, dvec& h, dvec& b, dvec& t, dmat& w){ h = etl::sigmoid(b + etl::mul(v, w, t)); }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(100, 500, 550, 600, 1000, 2000, 2500), VALUES_POLICY(100, 120, 500, 1000, 1200, 2000, 5000)),
    "rbm_visible",
    [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1), dvec(d1), dmat(d1, d2));},
    [](dvec& v, dvec& h, dvec& c, dvec& t, dmat& w){ v = etl::sigmoid(c + etl::mul(w, h, t)); }
)

//Small optimizations
// 1. Some time can be saved by keeping one matrix for the im2col result and pass it to conv_2d_valid_multi
// 2. fflip is already done in conv_2d_multi and fflip(fflip(A)) = A, therefore only tranpose is necessary.
// This means calling the _prepared version of conv_2d_valid_multi

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40), VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40), VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36), VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),
    "conv_rbm_hidden",
    [](std::size_t nc, std::size_t k, std::size_t nv, std::size_t nh){
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dmat4(nc,k,nw,nw), dvec(k), dmat3(nc,nv,nv), dmat3(k, nh, nh), dmat4(2ul, k, nh, nh));},
    [](dmat4& w, dmat4& w_t, dvec& b, dmat3& v, dmat3& h, dmat4 v_cv){
        v_cv(1) = 0;

        w_t = w;

        for(std::size_t channel = 0; channel < etl::dim<0>(w_t); ++channel){
            for(size_t k = 0; k < etl::dim<0>(b); ++k){
                w_t(channel)(k).fflip_inplace();
            }
        }

        for(std::size_t channel = 0; channel < etl::dim<0>(w_t); ++channel){
            conv_2d_valid_multi(v(channel), w_t(channel), v_cv(0));

            v_cv(1) += v_cv(0);
        }

        h = etl::sigmoid(etl::rep(b, etl::dim<1>(h), etl::dim<2>(h)) + v_cv(1));
    }
)

CPM_DIRECT_BENCH_TWO_PASS_NS_P(
    NARY_POLICY(VALUES_POLICY(1, 1, 1, 3, 3, 3, 30, 40), VALUES_POLICY(10, 10, 30, 30, 30, 40, 40, 40), VALUES_POLICY(28, 28, 28, 28, 36, 36, 36, 36), VALUES_POLICY(5, 11, 19, 19, 19, 19, 19, 19)),
    "conv_rbm_visible",
    [](std::size_t nc, std::size_t k, std::size_t nv, std::size_t nh){
        auto nw = nv - nh + 1;
        return std::make_tuple(dmat4(nc,k,nw,nw), dvec(k), dvec(nc), dmat3(nc,nv,nv), dmat3(k,nh,nh), dmat3(k,nv,nv));},
    [](dmat4& w, dvec& b, dvec& c, dmat3& v, dmat3& h, dmat3& h_cv){
        for(std::size_t channel = 0; channel < etl::dim<0>(c); ++channel){
            for(std::size_t k = 0; k < etl::dim<0>(b); ++k){
                h_cv(k) = etl::conv_2d_full(h(k), w(channel)(k));
            }

            v(channel) = sigmoid(c(channel) + sum_l(h_cv));
        }
    }
)

#ifdef TEST_MKL

// Bench 2D-FFT
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        fft_2d_policy,
        "sfft_2d",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1, d2), cmat(d1, d2)); },
        [](smat& a, cmat& r){ r = etl::fft_2d(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_2d_policy,
        "dfft_2d",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d2), zmat(d1, d2)); },
        [](dmat& a, zmat& r){ r = etl::fft_2d(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_2d_policy,
        "cfft_2d",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1, d2), cmat(d1, d2)); },
        [](cmat& a, cmat& r){ r = etl::fft_2d(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_2d_policy,
        "zfft_2d",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1, d2), zmat(d1, d2)); },
        [](zmat& a, zmat& r){ r = etl::fft_2d(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_2d_policy,
        "sifft_2d",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1, d2), smat(d1, d2)); },
        [](cmat& a, smat& r){ r = etl::ifft_2d_real(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_2d_policy,
        "difft_2d",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1, d2), dmat(d1, d2)); },
        [](zmat& a, dmat& r){ r = etl::ifft_2d_real(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_2d_policy,
        "cifft_2d",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1, d2), cmat(d1, d2)); },
        [](cmat& a, cmat& r){ r = etl::ifft_2d(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_2d_policy,
        "zifft_2d",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1, d2), zmat(d1, d2)); },
        [](zmat& a, zmat& r){ r = etl::ifft_2d(a); }
        );
}

// Bench 1D-FFT
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        fft_1d_policy,
        "sfft_1d",
        [](std::size_t d){ return std::make_tuple(svec(d), cvec(d)); },
        [](svec& a, cvec& r){ r = etl::fft_1d(a); }
        ); CPM_TWO_PASS_NS_P( fft_1d_policy, "dfft_1d", [](std::size_t d){ return std::make_tuple(dvec(d), zvec(d)); }, [](dvec& a, zvec& r){ r = etl::fft_1d(a); });

    CPM_TWO_PASS_NS_P(
        fft_1d_policy,
        "cfft_1d",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d)); },
        [](cvec& a, cvec& r){ r = etl::fft_1d(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_1d_policy,
        "zfft_1d",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d)); },
        [](zvec& a, zvec& r){ r = etl::fft_1d(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_1d_policy,
        "sifft_1d",
        [](std::size_t d){ return std::make_tuple(cvec(d), svec(d)); },
        [](cvec& a, svec& r){ r = etl::ifft_1d_real(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_1d_policy,
        "difft_1d",
        [](std::size_t d){ return std::make_tuple(zvec(d), dvec(d)); },
        [](zvec& a, dvec& r){ r = etl::ifft_1d_real(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_1d_policy,
        "cifft_1d",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d)); },
        [](cvec& a, cvec& r){ r = etl::ifft_1d(a); }
        );

    CPM_TWO_PASS_NS_P(
        fft_1d_policy,
        "zifft_1d",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d)); },
        [](zvec& a, zvec& r){ r = etl::ifft_1d(a); }
        );
}

#endif // TEST_MKL

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sigmoid(s)", sigmoid_policy,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(smat(d,d), smat(d,d)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b){ a = etl::sigmoid(b); }),
    CPM_SECTION_FUNCTOR("fast", [](smat& a, smat& b){ a = etl::fast_sigmoid(b); }),
    CPM_SECTION_FUNCTOR("hard", [](smat& a, smat& b){ a = etl::hard_sigmoid(b); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sigmoid(d)", sigmoid_policy,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(dmat(d,d), dmat(d,d)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b){ a = etl::sigmoid(b); }),
    CPM_SECTION_FUNCTOR("fast", [](dmat& a, dmat& b){ a = etl::fast_sigmoid(b); }),
    CPM_SECTION_FUNCTOR("hard", [](dmat& a, dmat& b){ a = etl::hard_sigmoid(b); })
)
