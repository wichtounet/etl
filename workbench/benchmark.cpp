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

typedef std::chrono::steady_clock timer_clock;
typedef std::chrono::milliseconds milliseconds;
typedef std::chrono::microseconds microseconds;

namespace {

#define TER_FUNCTOR(name, ...) \
template<typename A, typename B, typename C> \
struct name { \
    static void apply(A& a, B& b, C& c){ \
        (__VA_ARGS__); \
    } \
}

#ifdef TEST_SSE
#define TER_FUNCTOR_SSE(name, ...) TER_FUNCTOR(name, __VA_ARGS__)
#else
#define TER_FUNCTOR_SSE(name, ...)
#endif

#ifdef TEST_AVX
#define TER_FUNCTOR_AVX(name, ...) TER_FUNCTOR(name, __VA_ARGS__)
#else
#define TER_FUNCTOR_AVX(name, ...)
#endif

#ifdef TEST_MKL
#define TER_FUNCTOR_MKL(name, ...) TER_FUNCTOR(name, __VA_ARGS__)
#else
#define TER_FUNCTOR_MKL(name, ...)
#endif

#ifdef TEST_MMUL_CONV
#define TER_FUNCTOR_CM(name, ...) TER_FUNCTOR(name, __VA_ARGS__)
#else
#define TER_FUNCTOR_CM(name, ...)
#endif

template<typename T>
void randomize_double(T& container){
    static std::default_random_engine rand_engine(std::time(nullptr));
    static std::uniform_real_distribution<double> real_distribution(-1000.0, 1000.0);
    static auto generator = std::bind(real_distribution, rand_engine);

    for(auto& v : container){
        v = generator();
    }
}

template<typename T1>
void randomize(T1& container){
    randomize_double(container);
}

template<typename T1, typename... TT>
void randomize(T1& container, TT&... containers){
    randomize_double(container);
    randomize(containers...);
}

std::string duration_str(std::size_t duration_us){
    double duration = duration_us;

    if(duration > 1000 * 1000){
        return std::to_string(duration / 1000.0 / 1000.0) + "s";
    } else if(duration > 1000){
        return std::to_string(duration / 1000.0) + "ms";
    } else {
        return std::to_string(duration_us) + "us";
    }
}

constexpr const std::size_t WARM = 25;
constexpr const std::size_t MEASURE = 100;

template<typename Functor, typename... T>
std::size_t measure_only(Functor&& functor, T&... references){
    for(std::size_t i = 0; i < WARM; ++i){
        randomize(references...);
        functor();
    }

    std::size_t duration_acc = 0;

    for(std::size_t i = 0; i < MEASURE; ++i){
        randomize(references...);
        auto start_time = timer_clock::now();
        functor();
        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }

    return duration_acc;
}

template<typename Functor, typename... T>
std::size_t measure_only_safe(Functor&& functor, T&... references){
    for(std::size_t i = 0; i < WARM; ++i){
        randomize(references...);
        functor(i);
    }

    std::size_t duration_acc = 0;

    for(std::size_t i = 0; i < MEASURE; ++i){
        randomize(references...);
        auto start_time = timer_clock::now();
        functor(i);
        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }

    return duration_acc;
}

template<typename Functor, typename... T>
void measure(const std::string& title, Functor&& functor, T&... references){
    auto duration_acc = measure_only(std::forward<Functor>(functor), references...);
    std::cout << title << " took " << duration_str(duration_acc) << "\n";
}

template<bool Enable = true>
struct sub_measure {
    template<typename Functor, typename... T>
    static void measure_sub(const std::string& title, Functor&& functor, T&... references){
        auto duration_acc = measure_only_safe(std::forward<Functor>(functor), references...);

        std::cout << "\t" << title << std::string(20 - title.size(), ' ') << duration_str(duration_acc) << std::endl;
    }
};

template<>
struct sub_measure<false> {
    template<typename Functor, typename... T>
    static void measure_sub(const std::string& , Functor&& , T&... ){}
};

template<bool Enable = true, typename Functor, typename... T>
void measure_sub(const std::string& title, Functor&& functor, T&... references){
    sub_measure<Enable>::measure_sub(title, std::forward<Functor>(functor), references...);
}

template<template <typename, typename, typename> class Functor, bool Enable = true>
struct sub_measure_ter {
    template<typename A, typename B, typename C>
    static void measure_sub(const std::string& title, A& a, B& b, C& c){
        std::size_t duration_acc = measure_only_safe([&a, &b, &c](int){Functor<A, B, C>::apply(a, b, c);}, a, b);

        std::cout << "\t" << title << std::string(20 - title.size(), ' ') << duration_str(duration_acc) << std::endl;
    }
};

template<template <typename, typename, typename> class Functor>
struct sub_measure_ter<Functor, false> {
    template<typename A, typename B, typename C>
    static void measure_sub(const std::string& , A&, B&, C&){}
};

template<template<typename, typename, typename> class Functor, bool Enable = true, typename A, typename B, typename C>
void measure_sub_ter(const std::string& title, A& a, B& b, C& c){
    sub_measure_ter<Functor, Enable>::measure_sub(title, a, b, c);
}

using dvec = etl::dyn_vector<double>;
using dmat = etl::dyn_matrix<double>;
using dmat2 = etl::dyn_matrix<double, 2>;
using dmat3 = etl::dyn_matrix<double, 3>;
using dmat4 = etl::dyn_matrix<double, 4>;

using svec = etl::dyn_vector<float>;
using smat = etl::dyn_matrix<float>;

using mat_policy = VALUES_POLICY(10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000);
using mat_policy_2d = NARY_POLICY(mat_policy, mat_policy);

using conv_1d_large_policy = NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000));
using conv_2d_large_policy = NARY_POLICY(VALUES_POLICY(100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150), VALUES_POLICY(50, 50, 55, 55, 60, 60, 65, 65, 70, 70, 75));

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
        "R = A + B",
        [](auto d1, auto d2){ return std::make_tuple(dmat(d1, d2), dmat(d1, d2), dmat(d1, d2), dmat(d1, d2)); },
        [](dmat& A, dmat& B, dmat& C, dmat& R){ R = A + B + C; }
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
        NARY_POLICY(VALUES_POLICY(100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200), VALUES_POLICY(50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)),
        "r = conv_2d_full(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200), VALUES_POLICY(50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)),
        "r = conv_2d_same(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1, d1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200), VALUES_POLICY(50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)),
        "r = conv_2d_valid(a,b)(large)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }
        );
}

//2D-Convolution benchmarks with small-kernel
CPM_BENCH() {
    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450, 500), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40, 45, 50)),
        "r = conv_2d_full(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 + d2 - 1, d1 + d2 - 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_full(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450, 500), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40, 45, 50)),
        "r = conv_2d_same(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1, d1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_same(a, b); }
        );

    CPM_TWO_PASS_NS_P(
        NARY_POLICY(VALUES_POLICY(100, 150, 200, 250, 300, 350, 400, 450, 500), VALUES_POLICY(10, 15, 20, 25, 30, 35, 40, 45, 50)),
        "r = conv_2d_valid(a,b)(small)",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dmat(d1, d1), dmat(d2, d2), dmat(d1 - d2 + 1, d1 - d2 + 1)); },
        [](dmat& a, dmat& b, dmat& r){ r = etl::conv_2d_valid(a, b); }
        );
}

void bench_dyn_convmtx2(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b((d1 + d2 - 1)*(d1 + d2 - 1), d2 * d2);

    measure("convmtx2(" + std::to_string(d1) + "x" + std::to_string(d1) + "," + std::to_string(d2) + "," + std::to_string(d2) + ")",
        [&a, &b, d2](){b = etl::convmtx2(a, d2, d2);}
        , a);
}

void bench_dyn_convmtx2_t(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b((d1 + d2 - 1)*(d1 + d2 - 1), d2 * d2);

    measure("convmtx2_direct_t(" + std::to_string(d1) + "x" + std::to_string(d1) + "," + std::to_string(d2) + "," + std::to_string(d2) + ")",
        [&a, &b, d2](){etl::convmtx2_direct_t(b, a, d2, d2);}
        , a);
}

void bench_dyn_im2col_direct(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2 * d2, (d1 - d2 + 1)*(d1 - d2 + 1));

    measure("im2col_direct(" + std::to_string(d1) + "x" + std::to_string(d1) + "," + std::to_string(d2) + "," + std::to_string(d2) + ")",
        [&a, &b, d2](){etl::im2col_direct(b, a, d2, d2);}
        , a);
}

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
    MKL_SECTION_FUNCTOR("fft", [](svec& a, svec& b, svec& r){ r = etl::fft_conv_1d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](svec& a, svec& b, svec& r){ etl::impl::reduc::conv1_full(a, b, r); })
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
    MKL_SECTION_FUNCTOR("fft", [](smat& a, smat& b, smat& r){ r = etl::fft_conv_1d_full(a, b); })
     MC_SECTION_FUNCTOR("mmul", [](smat& a, smat& b, smat& r){ etl::impl::reduc::conv1_full(a, b, r); })
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

void bench_standard(){
    std::cout << "Start benchmarking...\n";
    std::cout << "... all structures are on stack\n\n";

    bench_dyn_convmtx2(16, 4);
    bench_dyn_convmtx2(32, 4);
    bench_dyn_convmtx2(32, 8);
    bench_dyn_convmtx2(32, 16);
    bench_dyn_convmtx2(64, 32);

    bench_dyn_convmtx2_t(16, 4);
    bench_dyn_convmtx2_t(32, 4);
    bench_dyn_convmtx2_t(32, 8);
    bench_dyn_convmtx2_t(32, 16);
    bench_dyn_convmtx2_t(64, 16);
    bench_dyn_convmtx2_t(64, 32);

    bench_dyn_im2col_direct(16, 4);
    bench_dyn_im2col_direct(32, 4);
    bench_dyn_im2col_direct(32, 8);
    bench_dyn_im2col_direct(32, 16);
    bench_dyn_im2col_direct(64, 32);
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "r = A * B (float)",
        [](std::size_t d){ return std::make_tuple(smat(d,d), smat(d,d), smat(d,d)); },
        [](smat& A, smat& B, smat& R){ R = A * B; }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "r = A * B (double)",
        [](std::size_t d){ return std::make_tuple(dmat(d,d), dmat(d,d), dmat(d,d)); },
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
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "R = A * (B + C)",
        [](std::size_t d){ return std::make_tuple(dmat(d,d), dmat(d,d), dmat(d,d), dmat(d,d)); },
        [](dmat& A, dmat& B, dmat& C, dmat& R){ R = A * (B + C); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "R = A * (B + C + D)",
        [](std::size_t d){ return std::make_tuple(dmat(d,d), dmat(d,d), dmat(d,d), dmat(d,d), dmat(d,d)); },
        [](dmat& A, dmat& B, dmat& C, dmat& D, dmat& R){ R = A * (B + C + D); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "R = (A + B) * (C + D)",
        [](std::size_t d){ return std::make_tuple(dmat(d,d), dmat(d,d), dmat(d,d), dmat(d,d), dmat(d,d)); },
        [](dmat& A, dmat& B, dmat& C, dmat& D, dmat& R){ R = (A + B) * (C + D); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "r = a * (A + B - C)",
        [](std::size_t d){ return std::make_tuple(dvec(d), dmat(d,d), dmat(d,d), dmat(d,d), dvec(d)); },
        [](dvec& a, dmat& A, dmat& B, dmat& C, dvec& r){ r = a * (A + B - C); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "r = a * (A * B)",
        [](std::size_t d){ return std::make_tuple(dvec(d), dmat(d,d), dmat(d,d), dvec(d)); },
        [](dvec& a, dmat& A, dmat& B, dvec& r){ r = a * (A * B); }
        );
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "r = a * A * B",
        [](std::size_t d){ return std::make_tuple(dvec(d), dmat(d,d), dmat(d,d), dvec(d)); },
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

} //end of anonymous namespace

int main_old(){
    bench_standard();

    return 0;
}
