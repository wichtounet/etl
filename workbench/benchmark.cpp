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

using svec = etl::dyn_vector<float>;
using smat = etl::dyn_matrix<float>;

using mat_policy = VALUES_POLICY(10, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000);
using mat_policy_2d = NARY_POLICY(mat_policy, mat_policy);

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

using conv_1d_large_policy = NARY_POLICY(VALUES_POLICY(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000), VALUES_POLICY(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000));

CPM_SECTION_P(conv_1d_large_policy, "sconv1_valid") 
    CPM_TWO_PASS_NS(
        "default",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 - d2 + 1)); },
        [](svec& a, svec& b, svec& r){ r = etl::conv_1d_valid(a, b); }
        );
    
    CPM_TWO_PASS_NS(
        "std",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 - d2 + 1)); },
        [](svec& a, svec& b, svec& r){ etl::impl::standard::conv1_valid(a, b, r); }
        );
#ifdef TEST_SSE
    CPM_TWO_PASS_NS(
        "sse",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 - d2 + 1)); },
        [](svec& a, svec& b, svec& r){ etl::impl::sse::sconv1_valid(a, b, r); }
        );
#endif
#ifdef TEST_AVX
    CPM_TWO_PASS_NS(
        "avx",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(svec(d1), svec(d2), svec(d1 - d2 + 1)); },
        [](svec& a, svec& b, svec& r){ etl::impl::avx::sconv1_valid(a, b, r); }
        );
#endif
}

CPM_SECTION_P(conv_1d_large_policy, "dconv1_valid") 
    CPM_TWO_PASS_NS(
        "default",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ r = etl::conv_1d_valid(a, b); }
        );
    
    CPM_TWO_PASS_NS(
        "std",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ etl::impl::standard::conv1_valid(a, b, r); }
        );
#ifdef TEST_SSE
    CPM_TWO_PASS_NS(
        "sse",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ etl::impl::sse::dconv1_valid(a, b, r); }
        );
#endif
#ifdef TEST_AVX
    CPM_TWO_PASS_NS(
        "avx",
        [](std::size_t d1, std::size_t d2){ return std::make_tuple(dvec(d1), dvec(d2), dvec(d1 - d2 + 1)); },
        [](dvec& a, dvec& b, dvec& r){ etl::impl::avx::dconv1_valid(a, b, r); }
        );
#endif
}

TER_FUNCTOR(default_conv_1d_full, c = etl::conv_1d_full(a, b));
TER_FUNCTOR(std_conv_1d_full, etl::impl::standard::conv1_full(a, b, c));
TER_FUNCTOR(mmul_conv_1d_full, etl::impl::reduc::conv1_full(a, b, c));
TER_FUNCTOR(fft_conv_1d_full, c = etl::fft_conv_1d_full(a, b));
TER_FUNCTOR_SSE(sse_sconv_1d_full, etl::impl::sse::sconv1_full(a, b, c));
TER_FUNCTOR_SSE(sse_dconv_1d_full, etl::impl::sse::dconv1_full(a, b, c));
TER_FUNCTOR_AVX(avx_sconv_1d_full, etl::impl::avx::sconv1_full(a, b, c));
TER_FUNCTOR_AVX(avx_dconv_1d_full, etl::impl::avx::dconv1_full(a, b, c));

template<typename A, typename B, typename C>
void measure_full_convolution_1d(A& a, B& b, C& c){
    measure_sub_ter<default_conv_1d_full>("default", a, b, c);
    measure_sub_ter<std_conv_1d_full>("std", a, b, c);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub_ter<sse_sconv_1d_full, F>("sse", a, b, c);
    measure_sub_ter<sse_dconv_1d_full, !F>("sse", a, b, c);
#endif

#ifdef TEST_AVX
    measure_sub_ter<avx_sconv_1d_full, F>("avx", a, b, c);
    measure_sub_ter<avx_dconv_1d_full, !F>("avx", a, b, c);
#endif

#ifdef TEST_MKL
    measure_sub_ter<fft_conv_1d_full>("fft", a, b, c);
#endif

#ifdef TEST_MMUL_CONV
    measure_sub_ter<mmul_conv_1d_full>("mmul", a, b, c);
#endif

    cpp_unused(F);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_full_convolution_1d_d(){
    etl::fast_matrix<double, D1> a;
    etl::fast_matrix<double, D2> b;
    etl::fast_matrix<double, D1+D2-1> c;

    std::cout << "fast_full_convolution_1d_d" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_full_convolution_1d(a, b, c);
}

void bench_dyn_full_convolution_1d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double,1> a(d1);
    etl::dyn_matrix<double,1> b(d2);
    etl::dyn_matrix<double,1> c(d1+d2-1);

    std::cout << "dyn_full_convolution_1d_d" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_full_convolution_1d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_full_convolution_1d_s(){
    etl::fast_matrix<float, D1> a;
    etl::fast_matrix<float, D2> b;
    etl::fast_matrix<float, D1+D2-1> c;

    std::cout << "fast_full_convolution_1d_s" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_full_convolution_1d(a, b, c);
}

void bench_dyn_full_convolution_1d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float,1> a(d1);
    etl::dyn_matrix<float,1> b(d2);
    etl::dyn_matrix<float,1> c(d1+d2-1);

    std::cout << "dyn_full_convolution_1d_s" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_full_convolution_1d(a, b, c);
}

TER_FUNCTOR(default_conv_1d_same, c = etl::conv_1d_same(a, b));
TER_FUNCTOR(std_conv_1d_same, etl::impl::standard::conv1_same(a, b, c));
TER_FUNCTOR_SSE(sse_sconv_1d_same, etl::impl::sse::sconv1_same(a, b, c));
TER_FUNCTOR_SSE(sse_dconv_1d_same, etl::impl::sse::dconv1_same(a, b, c));
TER_FUNCTOR_AVX(avx_sconv_1d_same, etl::impl::avx::sconv1_same(a, b, c));
TER_FUNCTOR_AVX(avx_dconv_1d_same, etl::impl::avx::dconv1_same(a, b, c));

template<typename A, typename B, typename C>
void measure_same_convolution_1d(A& a, B& b, C& c){
    measure_sub_ter<default_conv_1d_same>("default", a, b, c);
    measure_sub_ter<std_conv_1d_same>("std", a, b, c);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub_ter<sse_sconv_1d_same, F>("sse", a, b, c);
    measure_sub_ter<sse_dconv_1d_same, !F>("sse", a, b, c);
#endif

#ifdef TEST_AVX
    measure_sub_ter<avx_sconv_1d_same, F>("avx", a, b, c);
    measure_sub_ter<avx_dconv_1d_same, !F>("avx", a, b, c);
#endif

    cpp_unused(F);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_same_convolution_1d_d(){
    etl::fast_matrix<double, D1> a;
    etl::fast_matrix<double, D2> b;
    etl::fast_matrix<double, D1> c;

    std::cout << "fast_same_convolution_1d_d" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_same_convolution_1d(a, b, c);
}

void bench_dyn_same_convolution_1d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double,1> a(d1);
    etl::dyn_matrix<double,1> b(d2);
    etl::dyn_matrix<double,1> c(d1);

    std::cout << "dyn_same_convolution_1d_d" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_same_convolution_1d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_same_convolution_1d_s(){
    etl::fast_matrix<float, D1> a;
    etl::fast_matrix<float, D2> b;
    etl::fast_matrix<float, D1> c;

    std::cout << "fast_same_convolution_1d_s" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_same_convolution_1d(a, b, c);
}

void bench_dyn_same_convolution_1d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float,1> a(d1);
    etl::dyn_matrix<float,1> b(d2);
    etl::dyn_matrix<float,1> c(d1);

    std::cout << "dyn_same_convolution_1d_s" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_same_convolution_1d(a, b, c);
}

TER_FUNCTOR(default_conv_1d_valid, c = etl::conv_1d_valid(a, b));
TER_FUNCTOR(std_conv_1d_valid, etl::impl::standard::conv1_valid(a, b, c));
TER_FUNCTOR_SSE(sse_sconv_1d_valid, etl::impl::sse::sconv1_valid(a, b, c));
TER_FUNCTOR_SSE(sse_dconv_1d_valid, etl::impl::sse::dconv1_valid(a, b, c));
TER_FUNCTOR_AVX(avx_sconv_1d_valid, etl::impl::avx::sconv1_valid(a, b, c));
TER_FUNCTOR_AVX(avx_dconv_1d_valid, etl::impl::avx::dconv1_valid(a, b, c));

template<typename A, typename B, typename C>
void measure_valid_convolution_1d(A& a, B& b, C& c){
    measure_sub_ter<default_conv_1d_valid>("default", a, b, c);
    measure_sub_ter<std_conv_1d_valid>("std", a, b, c);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub_ter<sse_sconv_1d_valid, F>("sse", a, b, c);
    measure_sub_ter<sse_dconv_1d_valid, !F>("sse", a, b, c);
#endif

#ifdef TEST_AVX
    measure_sub_ter<avx_sconv_1d_valid, F>("avx", a, b, c);
    measure_sub_ter<avx_dconv_1d_valid, !F>("avx", a, b, c);
#endif

    cpp_unused(F);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_valid_convolution_1d_d(){
    etl::fast_matrix<double, D1> a;
    etl::fast_matrix<double, D2> b;
    etl::fast_matrix<double, D1-D2+1> c;

    std::cout << "fast_valid_convolution_1d_d" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_valid_convolution_1d(a, b, c);
}

void bench_dyn_valid_convolution_1d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double,1> a(d1);
    etl::dyn_matrix<double,1> b(d2);
    etl::dyn_matrix<double,1> c(d1-d2+1);

    std::cout << "dyn_valid_convolution_1d_d" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_valid_convolution_1d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_valid_convolution_1d_s(){
    etl::fast_matrix<float, D1> a;
    etl::fast_matrix<float, D2> b;
    etl::fast_matrix<float, D1-D2+1> c;

    std::cout << "fast_valid_convolution_1d_s" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_valid_convolution_1d(a, b, c);
}

void bench_dyn_valid_convolution_1d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float,1> a(d1);
    etl::dyn_matrix<float,1> b(d2);
    etl::dyn_matrix<float,1> c(d1-d2+1);

    std::cout << "dyn_valid_convolution_1d_s" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_valid_convolution_1d(a, b, c);
}

std::string x2(std::size_t D){
    return std::to_string(D) + "x" + std::to_string(D);
}

TER_FUNCTOR(default_conv_2d_full, c = etl::conv_2d_full(a, b));
TER_FUNCTOR(std_conv_2d_full, etl::impl::standard::conv2_full(a, b, c));
TER_FUNCTOR(mmul_conv_2d_full, etl::impl::reduc::conv2_full(a, b, c));
TER_FUNCTOR(fft_conv_2d_full, c = etl::fft_conv_2d_full(a, b));
TER_FUNCTOR_SSE(sse_sconv_2d_full, etl::impl::sse::sconv2_full(a, b, c));
TER_FUNCTOR_SSE(sse_dconv_2d_full, etl::impl::sse::dconv2_full(a, b, c));
TER_FUNCTOR_AVX(avx_sconv_2d_full, etl::impl::avx::sconv2_full(a, b, c));
TER_FUNCTOR_AVX(avx_dconv_2d_full, etl::impl::avx::dconv2_full(a, b, c));

template<typename A, typename B, typename C>
void measure_full_convolution_2d(A& a, B& b, C& c){
    measure_sub_ter<default_conv_2d_full>("default", a, b, c);
    measure_sub_ter<std_conv_2d_full>("std", a, b, c);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub_ter<sse_sconv_2d_full, F>("sse", a, b, c);
    measure_sub_ter<sse_dconv_2d_full, !F>("sse", a, b, c);
#endif

#ifdef TEST_AVX
    measure_sub_ter<avx_sconv_2d_full, F>("avx", a, b, c);
    measure_sub_ter<avx_dconv_2d_full, !F>("avx", a, b, c);
#endif

#ifdef TEST_MKL
    measure_sub_ter<fft_conv_2d_full>("fft", a, b, c);
#endif

#ifdef TEST_MMUL_CONV
    measure_sub_ter<mmul_conv_2d_full>("mmul", a, b, c);
#endif

    cpp_unused(F);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_full_convolution_2d_d(){
    etl::fast_matrix<double, D1, D1> a;
    etl::fast_matrix<double, D2, D2> b;
    etl::fast_matrix<double, D1+D2-1, D1+D2-1> c;

    std::cout << "fast_full_convolution_2d_d" << "(" << x2( D1 ) << "," << x2( D2 ) << ")" << std::endl;
    measure_full_convolution_2d(a, b, c);
}

void bench_dyn_full_convolution_2d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2, d2);
    etl::dyn_matrix<double> c(d1+d2-1, d1+d2-1);

    std::cout << "dyn_full_convolution_2d_d" << "(" << x2( d1 ) << "," << x2( d2 ) << ")" << std::endl;
    measure_full_convolution_2d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_full_convolution_2d_s(){
    etl::fast_matrix<float, D1, D1> a;
    etl::fast_matrix<float, D2, D2> b;
    etl::fast_matrix<float, D1+D2-1, D1+D2-1> c;

    std::cout << "fast_full_convolution_2d_s" << "(" << x2( D1 ) << "," << x2( D2 ) << ")" << std::endl;
    measure_full_convolution_2d(a, b, c);
}

void bench_dyn_full_convolution_2d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float> a(d1, d1);
    etl::dyn_matrix<float> b(d2, d2);
    etl::dyn_matrix<float> c(d1+d2-1, d1+d2-1);

    std::cout << "dyn_full_convolution_2d_s" << "(" << x2( d1 ) << "," << x2( d2 ) << ")" << std::endl;
    measure_full_convolution_2d(a, b, c);
}

TER_FUNCTOR(default_conv_2d_same, c = etl::conv_2d_same(a, b));
TER_FUNCTOR(std_conv_2d_same, etl::impl::standard::conv2_same(a, b, c));
TER_FUNCTOR_SSE(sse_sconv_2d_same, etl::impl::sse::sconv2_same(a, b, c));
TER_FUNCTOR_SSE(sse_dconv_2d_same, etl::impl::sse::dconv2_same(a, b, c));
TER_FUNCTOR_AVX(avx_sconv_2d_same, etl::impl::avx::sconv2_same(a, b, c));
TER_FUNCTOR_AVX(avx_dconv_2d_same, etl::impl::avx::dconv2_same(a, b, c));

template<typename A, typename B, typename C>
void measure_same_convolution_2d(A& a, B& b, C& c){
    measure_sub_ter<default_conv_2d_same>("default", a, b, c);
    measure_sub_ter<std_conv_2d_same>("std", a, b, c);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub_ter<sse_sconv_2d_same, F>("sse", a, b, c);
    measure_sub_ter<sse_dconv_2d_same, !F>("sse", a, b, c);
#endif

#ifdef TEST_AVX
    measure_sub_ter<avx_sconv_2d_same, F>("avx", a, b, c);
    measure_sub_ter<avx_dconv_2d_same, !F>("avx", a, b, c);
#endif

    cpp_unused(F);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_same_convolution_2d_d(){
    etl::fast_matrix<double, D1, D1> a;
    etl::fast_matrix<double, D2, D2> b;
    etl::fast_matrix<double, D1, D1> c;

    std::cout << "fast_same_convolution_2d_d" << "(" << x2( D1 ) << "," << x2( D2 ) << ")" << std::endl;
    measure_same_convolution_2d(a, b, c);
}

void bench_dyn_same_convolution_2d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2, d2);
    etl::dyn_matrix<double> c(d1, d1);

    std::cout << "dyn_same_convolution_2d_d" << "(" << x2( d1 ) << "," << x2( d2 ) << ")" << std::endl;
    measure_same_convolution_2d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_same_convolution_2d_s(){
    etl::fast_matrix<float, D1, D1> a;
    etl::fast_matrix<float, D2, D2> b;
    etl::fast_matrix<float, D1, D1> c;

    std::cout << "fast_same_convolution_2d_s" << "(" << x2( D1 ) << "," << x2( D2 ) << ")" << std::endl;
    measure_same_convolution_2d(a, b, c);
}

void bench_dyn_same_convolution_2d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float> a(d1, d1);
    etl::dyn_matrix<float> b(d2, d2);
    etl::dyn_matrix<float> c(d1, d1);

    std::cout << "dyn_same_convolution_2d_s" << "(" << x2( d1 ) << "," << x2( d2 ) << ")" << std::endl;
    measure_same_convolution_2d(a, b, c);
}

TER_FUNCTOR(default_conv_2d_valid, c = etl::conv_2d_valid(a, b));
TER_FUNCTOR(std_conv_2d_valid, etl::impl::standard::conv2_valid(a, b, c));
TER_FUNCTOR_SSE(sse_sconv_2d_valid, etl::impl::sse::sconv2_valid(a, b, c));
TER_FUNCTOR_SSE(sse_dconv_2d_valid, etl::impl::sse::dconv2_valid(a, b, c));
TER_FUNCTOR_AVX(avx_sconv_2d_valid, etl::impl::avx::sconv2_valid(a, b, c));
TER_FUNCTOR_AVX(avx_dconv_2d_valid, etl::impl::avx::dconv2_valid(a, b, c));

template<typename A, typename B, typename C>
void measure_valid_convolution_2d(A& a, B& b, C& c){
    measure_sub_ter<default_conv_2d_valid>("default", a, b, c);
    measure_sub_ter<std_conv_2d_valid>("std", a, b, c);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub_ter<sse_sconv_2d_valid, F>("sse", a, b, c);
    measure_sub_ter<sse_dconv_2d_valid, !F>("sse", a, b, c);
#endif

#ifdef TEST_AVX
    measure_sub_ter<avx_sconv_2d_valid, F>("avx", a, b, c);
    measure_sub_ter<avx_dconv_2d_valid, !F>("avx", a, b, c);
#endif

    cpp_unused(F);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_valid_convolution_2d_d(){
    etl::fast_matrix<double, D1, D1> a;
    etl::fast_matrix<double, D2, D2> b;
    etl::fast_matrix<double, D1-D2+1, D1-D2+1> c;

    std::cout << "fast_valid_convolution_2d_d" << "(" << x2( D1 ) << "," << x2( D2 ) << ")" << std::endl;
    measure_valid_convolution_2d(a, b, c);
}

void bench_dyn_valid_convolution_2d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2, d2);
    etl::dyn_matrix<double> c(d1-d2+1, d1-d2+1);

    std::cout << "dyn_valid_convolution_2d_d" << "(" << x2( d1 ) << "," << x2( d2 ) << ")" << std::endl;
    measure_valid_convolution_2d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_valid_convolution_2d_s(){
    etl::fast_matrix<float, D1, D1> a;
    etl::fast_matrix<float, D2, D2> b;
    etl::fast_matrix<float, D1-D2+1, D1-D2+1> c;

    std::cout << "fast_valid_convolution_2d_s" << "(" << x2( D1 ) << "," << x2( D2 ) << ")" << std::endl;
    measure_valid_convolution_2d(a, b, c);
}

void bench_dyn_valid_convolution_2d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float> a(d1, d1);
    etl::dyn_matrix<float> b(d2, d2);
    etl::dyn_matrix<float> c(d1-d2+1, d1-d2+1);

    std::cout << "dyn_valid_convolution_2d_s" << "(" << x2( d1 ) << "," << x2( d2 ) << ")" << std::endl;
    measure_valid_convolution_2d(a, b, c);
}

TER_FUNCTOR(default_mmul, c = etl::mul(a, b));
TER_FUNCTOR(std_mmul, etl::impl::standard::mm_mul(a, b, c));
TER_FUNCTOR(lazy_mmul, c = etl::lazy_mul(a, b));
TER_FUNCTOR(eblas_mmul_s, etl::impl::eblas::fast_sgemm(a, b, c));
TER_FUNCTOR(eblas_mmul_d, etl::impl::eblas::fast_dgemm(a, b, c));
TER_FUNCTOR(blas_mmul_s, etl::impl::blas::sgemm(a, b, c));
TER_FUNCTOR(blas_mmul_d, etl::impl::blas::dgemm(a, b, c));
TER_FUNCTOR(strassen_mmul, c = etl::strassen_mul(a, b));

template<typename A, typename B, typename C>
void measure_mmul(A& a, B& b, C& c){
    measure_sub_ter<default_mmul>("default", a, b, c);
    measure_sub_ter<std_mmul>("std", a, b, c);
    measure_sub_ter<lazy_mmul>("lazy", a, b, c);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

    measure_sub_ter<eblas_mmul_s, F>("eblas", a, b, c);
    measure_sub_ter<eblas_mmul_d, !F>("eblas", a, b, c);

#ifdef TEST_BLAS
    measure_sub_ter<blas_mmul_s, F>("blas", a, b, c);
    measure_sub_ter<blas_mmul_d, !F>("blas", a, b, c);
#endif

#ifdef TEST_STRASSEN
    measure_sub_ter<strassen_mmul>("strassen", a, b, c);
#endif
}

template<std::size_t D1, std::size_t D2>
void bench_fast_mmul(){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D2, D1> b;
    etl::fast_matrix<double, D1, D1> c;

    std::cout << "fast_mmul_d" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_mmul(a, b, c);
}

void bench_dyn_mmul(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d2);
    etl::dyn_matrix<double> b(d2, d1);
    etl::dyn_matrix<double> c(d1, d1);

    std::cout << "dyn_mmul_d" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_mmul(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_mmul_s(){
    etl::fast_matrix<float, D1, D2> a;
    etl::fast_matrix<float, D2, D1> b;
    etl::fast_matrix<float, D1, D1> c;

    std::cout << "fast_mmul_s" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_mmul(a, b, c);
}

void bench_dyn_mmul_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float> a(d1, d2);
    etl::dyn_matrix<float> b(d2, d1);
    etl::dyn_matrix<float> c(d1, d1);

    std::cout << "dyn_mmul_s" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_mmul(a, b, c);
}

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

    bench_fast_valid_convolution_1d_d<1024, 64>();
    bench_fast_valid_convolution_1d_d<2048, 128>();
    bench_dyn_valid_convolution_1d_d(1024, 64);
    bench_dyn_valid_convolution_1d_d(2048, 64);

    bench_fast_valid_convolution_1d_s<2*1024, 2*64>();
    bench_fast_valid_convolution_1d_s<2*2048, 2*128>();
    bench_dyn_valid_convolution_1d_s(2*1024, 2*64);
    bench_dyn_valid_convolution_1d_s(2*2048, 2*64);

    bench_fast_same_convolution_1d_d<2*1024, 2*64>();
    bench_fast_same_convolution_1d_d<2*2048, 2*128>();
    bench_dyn_same_convolution_1d_d(2*1024, 2*64);
    bench_dyn_same_convolution_1d_d(2*2048, 2*64);

    bench_fast_same_convolution_1d_s<2*1024, 2*64>();
    bench_fast_same_convolution_1d_s<2*2048, 2*128>();
    bench_dyn_same_convolution_1d_s(2*1024, 2*64);
    bench_dyn_same_convolution_1d_s(2*2048, 2*64);

    bench_fast_full_convolution_1d_d<2*1024, 2*64>();
    bench_fast_full_convolution_1d_d<2*2048, 2*128>();
    bench_dyn_full_convolution_1d_d(2*1024, 2*64);
    bench_dyn_full_convolution_1d_d(2*2048, 2*64);

    bench_fast_full_convolution_1d_s<2*1024, 2*64>();
    bench_fast_full_convolution_1d_s<2*2048, 2*128>();
    bench_dyn_full_convolution_1d_s(2*1024, 2*64);
    bench_dyn_full_convolution_1d_s(2*2048, 2*64);

    bench_fast_valid_convolution_2d_d<64, 32>();
    bench_fast_valid_convolution_2d_d<128, 32>();
    bench_dyn_valid_convolution_2d_d(64, 32);
    bench_dyn_valid_convolution_2d_d(128, 32);

    bench_fast_valid_convolution_2d_s<64, 32>();
    bench_fast_valid_convolution_2d_s<128, 32>();
    bench_dyn_valid_convolution_2d_s(64, 32);
    bench_dyn_valid_convolution_2d_s(128, 32);

    bench_fast_same_convolution_2d_d<64, 32>();
    bench_fast_same_convolution_2d_d<128, 32>();
    bench_dyn_same_convolution_2d_d(64, 32);
    bench_dyn_same_convolution_2d_d(128, 32);

    bench_fast_same_convolution_2d_s<64, 32>();
    bench_fast_same_convolution_2d_s<128, 32>();
    bench_dyn_same_convolution_2d_s(64, 32);
    bench_dyn_same_convolution_2d_s(128, 32);

    bench_fast_full_convolution_2d_d<30, 13>();
    bench_fast_full_convolution_2d_d<32, 16>();
    bench_fast_full_convolution_2d_d<64, 32>();
    bench_fast_full_convolution_2d_d<128, 32>();
    bench_dyn_full_convolution_2d_d(64, 32);
    bench_dyn_full_convolution_2d_d(128, 32);

    bench_fast_full_convolution_2d_s<64, 32>();
    bench_fast_full_convolution_2d_s<128, 32>();
    bench_dyn_full_convolution_2d_s(64, 32);
    bench_dyn_full_convolution_2d_s(128, 32);

    bench_fast_mmul<64, 32>();
    bench_fast_mmul<128, 64>();
    bench_fast_mmul<256, 128>();
    bench_dyn_mmul(64, 32);
    bench_dyn_mmul(128, 64);
    bench_dyn_mmul(256, 128);
    bench_dyn_mmul(512, 256);

    bench_fast_mmul_s<2*64, 2*32>();
    bench_fast_mmul_s<2*128, 2*64>();
    bench_fast_mmul_s<2*256, 2*128>();
    bench_dyn_mmul_s(64, 32);
    bench_dyn_mmul_s(128, 64);
    bench_dyn_mmul_s(256, 128);
    bench_dyn_mmul_s(512, 256);
}

CPM_BENCH(){
    CPM_TWO_PASS_NS_P(
        VALUES_POLICY(10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100),
        "r = A * B",
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

void bench_rbm_hidden(std::size_t n_v, std::size_t n_h){
    etl::dyn_vector<double> v(n_v);
    etl::dyn_vector<double> h(n_h);
    etl::dyn_vector<double> b(n_h);
    etl::dyn_vector<double> t(n_h);
    etl::dyn_matrix<double> w(n_v, n_h);

    measure("RBM Hidden Activation (" + std::to_string(n_v) + "->" + std::to_string(n_h) + ")", [&](){
        h = etl::sigmoid(b + etl::mul(v, w, t));
    }, b, v, w);
}

void bench_rbm_visible(std::size_t n_v, std::size_t n_h){
    etl::dyn_vector<double> v(n_v);
    etl::dyn_vector<double> h(n_h);
    etl::dyn_vector<double> c(n_v);
    etl::dyn_vector<double> t(n_v);
    etl::dyn_matrix<double> w(n_v, n_h);

    measure("RBM Visible Activation (" + std::to_string(n_v) + "->" + std::to_string(n_h) + ")", [&](){
        v = etl::sigmoid(c + etl::mul(w, h, t));
    }, c, h, w);
}

void bench_conv_rbm_hidden(std::size_t NC, std::size_t K, std::size_t NV, std::size_t NH){
    auto NW = NV - NH + 1;

    etl::dyn_matrix<double, 4> w(NC, K, NW, NW);
    etl::dyn_matrix<double, 4> w_t(NC, K, NW, NW);
    etl::dyn_vector<double> b(K);

    etl::dyn_matrix<double, 3> v(NC, NV, NV);
    etl::dyn_matrix<double, 3> h(K, NH, NH);

    etl::dyn_matrix<double, 4> v_cv(2UL, K, NH, NH);
    etl::dyn_matrix<double, 3> h_cv(2UL, NV, NV);

    //Small optimizations
    // 1. Some time can be saved by keeping one matrix for the im2col result and pass it to conv_2d_valid_multi
    // 2. fflip is already done in conv_2d_multi and fflip(fflip(A)) = A, therefore only tranpose is necessary.
    // This means calling the _prepared version of conv_2d_valid_multi

    measure("CRBM Hidden Activation (" + std::to_string(NC) + "x" + std::to_string(NV) + "^2 -> " +
        std::to_string(K) + "x" + std::to_string(NH) + "^2)",
        [&](){
            v_cv(1) = 0;

            for(std::size_t channel = 0; channel < NC; ++channel){
                for(size_t k = 0; k < K; ++k){
                    w_t(channel)(k).fflip_inplace();
                }
            }

            for(std::size_t channel = 0; channel < NC; ++channel){
                conv_2d_valid_multi(v(channel), w_t(channel), v_cv(0));

                v_cv(1) += v_cv(0);
            }

            h = etl::sigmoid(etl::rep(b, NH, NH) + v_cv(1));
    }, b, v, w);
}

void bench_conv_rbm_visible(std::size_t NC, std::size_t K, std::size_t NV, std::size_t NH){
    auto NW = NV - NH + 1;

    etl::dyn_matrix<double, 4> w(NC, K, NW, NW);
    etl::dyn_vector<double> b(K);
    etl::dyn_vector<double> c(NC);

    etl::dyn_matrix<double, 3> v(NC, NV, NV);
    etl::dyn_matrix<double, 3> h(K, NH, NH);

    etl::dyn_matrix<double, 3> h_cv(K, NV, NV);

    measure("CRBM Visible Activation (" + std::to_string(NC) + "x" + std::to_string(NV) + "^2 -> " +
        std::to_string(K) + "x" + std::to_string(NH) + "^2)",
        [&](){
            for(std::size_t channel = 0; channel < NC; ++channel){
                for(std::size_t k = 0; k < K; ++k){
                    h_cv(k) = etl::conv_2d_full(h(k), w(channel)(k));
                }

                v(channel) = sigmoid(c(channel) + sum_l(h_cv));
            }
    }, c, h, w);
}

void bench_dll(){
    std::cout << "Start DLL benchmarking...\n";

    bench_rbm_hidden(100, 100);
    bench_rbm_hidden(200, 200);
    bench_rbm_hidden(500, 200);
    bench_rbm_hidden(500, 2000);
    bench_rbm_hidden(1000, 1000);

    bench_rbm_visible(100, 100);
    bench_rbm_visible(200, 200);
    bench_rbm_visible(500, 200);
    bench_rbm_visible(500, 2000);
    bench_rbm_visible(1000, 1000);

    bench_conv_rbm_hidden(1, 10, 10, 7);
    bench_conv_rbm_hidden(1, 10, 30, 7);
    bench_conv_rbm_hidden(1, 40, 30, 7);
    bench_conv_rbm_hidden(3, 10, 30, 7);
    bench_conv_rbm_hidden(3, 40, 30, 7);
    bench_conv_rbm_hidden(3, 40, 30, 14);
    bench_conv_rbm_hidden(40, 40, 30, 16);

    bench_conv_rbm_visible(1, 10, 10, 7);
    bench_conv_rbm_visible(1, 10, 30, 7);
    bench_conv_rbm_visible(1, 40, 30, 7);
    bench_conv_rbm_visible(3, 10, 30, 7);
    bench_conv_rbm_visible(3, 40, 30, 7);
    bench_conv_rbm_visible(3, 40, 30, 14);
    bench_conv_rbm_visible(40, 40, 30, 16);
}

} //end of anonymous namespace

int main_old(int argc, char* argv[]){
    std::vector<std::string> args;

    for(int i = 1; i < argc; ++i){
        args.emplace_back(argv[i]);
    }

    bool dll = false;

    for(auto& arg : args){
        if(arg == "dll"){
            dll = true;
        }
    }

    if(dll){
        bench_dll();
    } else {
        bench_standard();
    }

    return 0;
}
