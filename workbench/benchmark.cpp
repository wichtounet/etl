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

#define CPM_WARMUP 2
#define CPM_REPEAT 8

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

#ifdef ETL_CUBLAS_MODE
#define TEST_CUBLAS
#endif

#ifdef ETL_CUFFT_MODE
#define TEST_CUFFT
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

using smat_cm = etl::dyn_matrix_cm<float>;
using dmat_cm = etl::dyn_matrix_cm<double>;
using cmat_cm = etl::dyn_matrix_cm<std::complex<float>>;
using zmat_cm = etl::dyn_matrix_cm<std::complex<double>>;

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

using fft_1d_policy = VALUES_POLICY(10, 100, 1000, 10000, 100000, 500000);
using fft_1d_policy_2 = VALUES_POLICY(16, 64, 256, 1024, 16384, 131072, 1048576, 2097152);
using fft_1d_many_policy = VALUES_POLICY(10, 50, 100, 500, 1000, 5000, 10000);

using fft_2d_policy = NARY_POLICY(
    VALUES_POLICY(8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    VALUES_POLICY(8, 16, 32, 64, 128, 256, 512, 1024, 2048));

using sigmoid_policy = VALUES_POLICY(250, 500, 750, 1000, 1250, 1500, 1750, 2000);

using small_square_policy = NARY_POLICY(VALUES_POLICY(50, 100, 150, 200, 250, 300, 350, 400, 450, 500), VALUES_POLICY(50, 100, 150, 200, 250, 300, 350, 400, 450, 500));
using square_policy = NARY_POLICY(VALUES_POLICY(50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000), VALUES_POLICY(50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000));

using gemv_policy = NARY_POLICY(
    VALUES_POLICY(250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000),
    VALUES_POLICY(250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000));

using trans_sub_policy = VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000);
using trans_policy = NARY_POLICY(
    VALUES_POLICY(100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000),
    VALUES_POLICY(100, 200, 200, 300, 300, 400, 400, 500, 500, 600, 600, 700, 700, 800, 800, 900, 900, 1000, 1000, 1100));

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

#ifdef TEST_BLAS
#define BLAS_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define BLAS_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_CUBLAS
#define CUBLAS_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define CUBLAS_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_CUFFT
#define CUFFT_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define CUFFT_SECTION_FUNCTOR(name, ...)
#endif

#ifdef TEST_MMUL_CONV
#define MC_SECTION_FUNCTOR(name, ...) , CPM_SECTION_FUNCTOR(name, __VA_ARGS__)
#else
#define MC_SECTION_FUNCTOR(name, ...)
#endif

//Bench assignment
CPM_BENCH() {
    CPM_TWO_PASS_NS(
        "r = a (s)",
        [](std::size_t d){ return std::make_tuple(svec(d), svec(d)); },
        [](svec& a, svec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (d)",
        [](std::size_t d){ return std::make_tuple(dvec(d), dvec(d)); },
        [](dvec& a, dvec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (c)",
        [](std::size_t d){ return std::make_tuple(cvec(d), cvec(d)); },
        [](cvec& a, cvec& r){ r = a; }
        );

    CPM_TWO_PASS_NS(
        "r = a (z)",
        [](std::size_t d){ return std::make_tuple(zvec(d), zvec(d)); },
        [](zvec& a, zvec& r){ r = a; }
        );

    CPM_TWO_PASS_NS_P(
        mat_policy_2d,
        "R = A (d)",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& R){ R = A; }
        );
}

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

//Bench transposition
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        trans_policy,
        "r = tranpose(a) (s)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1, d2), smat(d1, d2)); },
        [](smat& a, smat& r){ r = a.transpose(); }
        );

    CPM_TWO_PASS_NS_P(
        trans_policy,
        "r = tranpose(a) (d)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& a, dmat& r){ r = a.transpose(); }
        );

    CPM_TWO_PASS_NS_P(
        trans_policy,
        "a = tranpose(a) (s)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1, d2)); },
        [](smat& a){ a.transpose_inplace(); }
        );

    CPM_TWO_PASS_NS_P(
        trans_policy,
        "a = tranpose(a) (d)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d2)); },
        [](dmat& a){ a.transpose_inplace(); }
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
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)),
        "r = conv_1d_full(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)),
        "r = conv_1d_same(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000)),
        "r = conv_1d_valid(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); }
        );
}

//1D-Convolution benchmarks with small-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)),
        "r = conv_1d_full(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)),
        "r = conv_1d_same(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000), VALUES_POLICY(100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)),
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
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40)),
        "r = conv_2d_full(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40)),
        "r = conv_2d_same(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1, d1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40)),
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
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ etl::impl::sse::conv1_valid(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ etl::impl::avx::conv1_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv1_valid", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ etl::impl::standard::conv1_valid(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ etl::impl::sse::conv1_valid(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ etl::impl::avx::conv1_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv1_same", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ etl::impl::standard::conv1_same(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ etl::impl::sse::conv1_same(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ etl::impl::avx::conv1_same(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv1_same", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ etl::impl::standard::conv1_same(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ etl::impl::sse::conv1_same(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ etl::impl::avx::conv1_same(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv1_full", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, svec& b, svec& r){ r = etl::conv_1d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, svec& b, svec& r){ etl::impl::standard::conv1_full(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](svec& a, svec& b, svec& r){ etl::impl::sse::conv1_full(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](svec& a, svec& b, svec& r){ etl::impl::avx::conv1_full(a, b, r); })
    MKL_SECTION_FUNCTOR("fft", [](svec& a, svec& b, svec& r){ r = etl::fft_conv_1d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](svec& a, svec& b, svec& r){ etl::impl::reduc::conv1_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv1_full", conv_1d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dvec& b, dvec& r){ etl::impl::standard::conv1_full(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dvec& a, dvec& b, dvec& r){ etl::impl::sse::conv1_full(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dvec& a, dvec& b, dvec& r){ etl::impl::avx::conv1_full(a, b, r); })
    MKL_SECTION_FUNCTOR("fft", [](dvec& a, dvec& b, dvec& r){ r = etl::fft_conv_1d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](dvec& a, dvec& b, dvec& r){ etl::impl::reduc::conv1_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv2_valid", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ etl::impl::standard::conv2_valid(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ etl::impl::sse::conv2_valid(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ etl::impl::avx::conv2_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv2_valid", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ etl::impl::standard::conv2_valid(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ etl::impl::sse::conv2_valid(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ etl::impl::avx::conv2_valid(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv2_same", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ etl::impl::standard::conv2_same(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ etl::impl::sse::conv2_same(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ etl::impl::avx::conv2_same(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv2_same", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1,d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ etl::impl::standard::conv2_same(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ etl::impl::sse::conv2_same(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ etl::impl::avx::conv2_same(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("sconv2_full", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d1), smat(d2,d2), smat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& r){ etl::impl::standard::conv2_full(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](smat& a, smat& b, smat& r){ etl::impl::sse::conv2_full(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](smat& a, smat& b, smat& r){ etl::impl::avx::conv2_full(a, b, r); })
    MKL_SECTION_FUNCTOR("fft", [](smat& a, smat& b, smat& r){ r = etl::fft_conv_2d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](smat& a, smat& b, smat& r){ etl::impl::reduc::conv2_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("dconv2_full", conv_2d_large_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d1), dmat(d2,d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& r){ etl::impl::standard::conv2_full(a, b, r); })
    SSE_SECTION_FUNCTOR("sse", [](dmat& a, dmat& b, dmat& r){ etl::impl::sse::conv2_full(a, b, r); })
    AVX_SECTION_FUNCTOR("avx", [](dmat& a, dmat& b, dmat& r){ etl::impl::avx::conv2_full(a, b, r); })
    MKL_SECTION_FUNCTOR("fft", [](dmat& a, dmat& b, dmat& r){ r = etl::fft_conv_2d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](dmat& a, dmat& b, dmat& r){ etl::impl::reduc::conv2_full(a, b, r); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * B (s)", square_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d2), smat(d1,d2), smat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, smat& b, smat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, smat& b, smat& c){ etl::impl::standard::mm_mul(a, b, c); }),
    CPM_SECTION_FUNCTOR("eblas", [](smat& a, smat& b, smat& c){ etl::impl::eblas::gemm(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, smat& b, smat& c){ etl::impl::blas::gemm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, smat& b, smat& c){ etl::impl::cublas::gemm(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * B (cm/s)", square_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat_cm(d1,d2), smat_cm(d1,d2), smat_cm(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](smat_cm& a, smat_cm& b, smat_cm& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat_cm& a, smat_cm& b, smat_cm& c){ etl::impl::standard::mm_mul(a, b, c); }),
    CPM_SECTION_FUNCTOR("eblas", [](smat_cm& a, smat_cm& b, smat_cm& c){ etl::impl::eblas::gemm(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](smat_cm& a, smat_cm& b, smat_cm& c){ etl::impl::blas::gemm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat_cm& a, smat_cm& b, smat_cm& c){ etl::impl::cublas::gemm(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * B (d)", square_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d2), dmat(d1,d2), dmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dmat& b, dmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dmat& b, dmat& c){ etl::impl::standard::mm_mul(a, b, c); }),
    CPM_SECTION_FUNCTOR("eblas", [](dmat& a, dmat& b, dmat& c){ etl::impl::eblas::gemm(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](dmat& a, dmat& b, dmat& c){ etl::impl::blas::gemm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dmat& a, dmat& b, dmat& c){ etl::impl::cublas::gemm(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * B (c)", small_square_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1,d2), cmat(d1,d2), cmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cmat& b, cmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cmat& b, cmat& c){ etl::impl::standard::mm_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](cmat& a, cmat& b, cmat& c){ etl::impl::blas::gemm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cmat& a, cmat& b, cmat& c){ etl::impl::cublas::gemm(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * B (z)", small_square_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1,d2), zmat(d1,d2), zmat(d1, d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zmat& b, zmat& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b, zmat& c){ etl::impl::standard::mm_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](zmat& a, zmat& b, zmat& c){ etl::impl::blas::gemm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zmat& a, zmat& b, zmat& c){ etl::impl::cublas::gemm(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * x (s)", gemv_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(smat(d1,d2), svec(d2), svec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](smat& a, svec& b, svec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](smat& a, svec& b, svec& c){ etl::impl::standard::mv_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](smat& a, svec& b, svec& c){ etl::impl::blas::gemv(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](smat& a, svec& b, svec& c){ etl::impl::cublas::gemv(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * x (d)", gemv_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1,d2), dvec(d2), dvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](dmat& a, dvec& b, dvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dmat& a, dvec& b, dvec& c){ etl::impl::standard::mv_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](dmat& a, dvec& b, dvec& c){ etl::impl::blas::gemv(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dmat& a, dvec& b, dvec& c){ etl::impl::cublas::gemv(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * x (c)", gemv_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cmat(d1,d2), cvec(d2), cvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](cmat& a, cvec& b, cvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cmat& a, cvec& b, cvec& c){ etl::impl::standard::mv_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](cmat& a, cvec& b, cvec& c){ etl::impl::blas::gemv(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cmat& a, cvec& b, cvec& c){ etl::impl::cublas::gemv(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("A * x (z)", gemv_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zmat(d1,d2), zvec(d2), zvec(d1)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zvec& b, zvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zvec& b, zvec& c){ etl::impl::standard::mv_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](zmat& a, zvec& b, zvec& c){ etl::impl::blas::gemv(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zmat& a, zvec& b, zvec& c){ etl::impl::cublas::gemv(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("x * A (s)", gemv_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), smat(d1,d2), svec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](svec& a, smat& b, svec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](svec& a, smat& b, svec& c){ etl::impl::standard::vm_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](svec& a, smat& b, svec& c){ etl::impl::blas::gevm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](svec& a, smat& b, svec& c){ etl::impl::cublas::gevm(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("x * A (d)", gemv_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dmat(d1,d2), dvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](dvec& a, dmat& b, dvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](dvec& a, dmat& b, dvec& c){ etl::impl::standard::vm_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](dvec& a, dmat& b, dvec& c){ etl::impl::blas::gevm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](dvec& a, dmat& b, dvec& c){ etl::impl::cublas::gevm(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("x * A (c)", gemv_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(cvec(d1), cmat(d1,d2), cvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](cvec& a, cmat& b, cvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](cvec& a, cmat& b, cvec& c){ etl::impl::standard::vm_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](cvec& a, cmat& b, cvec& c){ etl::impl::blas::gevm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](cvec& a, cmat& b, cvec& c){ etl::impl::cublas::gevm(a, b, c); })
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("x * A (z)", gemv_policy,
    CPM_SECTION_INIT([](std::size_t d1, std::size_t d2){ return std::make_tuple(zvec(d1), zmat(d1,d2), zvec(d2)); }),
    CPM_SECTION_FUNCTOR("default", [](zvec& a, zmat& b, zvec& c){ c = a * b; }),
    CPM_SECTION_FUNCTOR("std", [](zvec& a, zmat& b, zvec& c){ etl::impl::standard::vm_mul(a, b, c); })
    BLAS_SECTION_FUNCTOR("blas", [](zvec& a, zmat& b, zvec& c){ etl::impl::blas::gevm(a, b, c); })
    CUBLAS_SECTION_FUNCTOR("cublas", [](zvec& a, zmat& b, zvec& c){ etl::impl::cublas::gevm(a, b, c); })
)


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
)

CPM_DIRECT_SECTION_TWO_PASS_NS_P("fft_1d_many(1000) (z)", fft_1d_many_policy,
    CPM_SECTION_INIT([](std::size_t d){ return std::make_tuple(zmat(1000UL, d), zmat(1000UL, d)); }),
    CPM_SECTION_FUNCTOR("default", [](zmat& a, zmat& b){ b = etl::fft_1d_many(a); }),
    CPM_SECTION_FUNCTOR("std", [](zmat& a, zmat& b){ etl::impl::standard::fft1_many(a, b); })
    MKL_SECTION_FUNCTOR("mkl", [](zmat& a, zmat& b){ etl::impl::blas::fft1_many(a, b); })
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
