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

#ifdef ETL_VECTORIZE
#ifdef __SSE3__
#define TEST_SSE
#endif
#ifdef __AVX__
#define TEST_AVX
#endif
#endif

#ifdef ETL_BLAS_MODE
#define TEST_BLAS
#endif

#ifdef ETL_BENCH_STRASSEN
#define TEST_STRASSEN
#endif

typedef std::chrono::high_resolution_clock timer_clock;
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

template<std::size_t D>
void bench_fast_vector_simple(){
    etl::fast_vector<double, D> a;
    etl::fast_vector<double, D> b;
    etl::fast_vector<double, D> c;

    measure("fast_vector_simple(" + std::to_string(D) + ")", [&a, &b, &c](){
        c = 3.5 * a + etl::sigmoid(1.0 + b);
    }, a, b);
}

void bench_dyn_vector_simple(std::size_t d){
    etl::dyn_vector<double> a(d);
    etl::dyn_vector<double> b(d);
    etl::dyn_vector<double> c(d);

    measure("dyn_vector_simple(" + std::to_string(d) + ")", [&a, &b, &c](){
        c = 3.5 * a + etl::sigmoid(1.0 + b);
    }, a, b);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_matrix_simple(){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D1, D2> b;
    etl::fast_matrix<double, D1, D2> c;

    measure("fast_matrix_simple(" + std::to_string(D1) + "," + std::to_string(D2) + ")(" + std::to_string(D1 * D2) + ")",
        [&a, &b, &c](){c = 3.5 * a + etl::sigmoid(1.0 + b);}
        , a, b);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_matrix_sigmoid(){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D1, D2> b;
    etl::fast_matrix<double, D1, D2> c;

    measure("fast_matrix_sigmoid(" + std::to_string(D1) + "," + std::to_string(D2) + ")(" + std::to_string(D1 * D2) + ")",
        [&a, &b, &c](){c = etl::sigmoid(1.0 + b);}
        , a, b);
}

void bench_dyn_matrix_simple(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d2);
    etl::dyn_matrix<double> b(d1, d2);
    etl::dyn_matrix<double> c(d1, d2);

    measure("dyn_matrix_simple(" + std::to_string(d1) + "," + std::to_string(d2) + ")(" + std::to_string(d1 * d2) + ")",
        [&a, &b, &c](){c = 3.5 * a + etl::sigmoid(1.0 + b);}
        , a, b);
}

void bench_dyn_matrix_sigmoid(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d2);
    etl::dyn_matrix<double> b(d1, d2);
    etl::dyn_matrix<double> c(d1, d2);

    measure("dyn_matrix_sigmoid(" + std::to_string(d1) + "," + std::to_string(d2) + ")(" + std::to_string(d1 * d2) + ")",
        [&a, &b, &c](){c = etl::sigmoid(1.0 + b);}
        , a, b);
}

TER_FUNCTOR(default_conv_1d_full, c = etl::conv_1d_full(a, b));
TER_FUNCTOR(std_conv_1d_full, etl::impl::standard::conv1_full(a, b, c));
TER_FUNCTOR(mmul_conv_1d_full, etl::impl::reduc::conv1_full(a, b, c));
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

    measure_sub_ter<mmul_conv_1d_full>("mmul", a, b, c);

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

TER_FUNCTOR(default_conv_2d_full, c = etl::conv_2d_full(a, b));
TER_FUNCTOR(std_conv_2d_full, etl::impl::standard::conv2_full(a, b, c));
TER_FUNCTOR(mmul_conv_2d_full, etl::impl::reduc::conv2_full(a, b, c));
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

    measure_sub_ter<mmul_conv_2d_full>("mmul", a, b, c);

    cpp_unused(F);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_full_convolution_2d_d(){
    etl::fast_matrix<double, D1, D1> a;
    etl::fast_matrix<double, D2, D2> b;
    etl::fast_matrix<double, D1+D2-1, D1+D2-1> c;

    std::cout << "fast_full_convolution_2d_d" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_full_convolution_2d(a, b, c);
}

void bench_dyn_full_convolution_2d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2, d2);
    etl::dyn_matrix<double> c(d1+d2-1, d1+d2-1);

    std::cout << "dyn_full_convolution_2d_d" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_full_convolution_2d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_full_convolution_2d_s(){
    etl::fast_matrix<float, D1, D1> a;
    etl::fast_matrix<float, D2, D2> b;
    etl::fast_matrix<float, D1+D2-1, D1+D2-1> c;

    std::cout << "fast_full_convolution_2d_s" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_full_convolution_2d(a, b, c);
}

void bench_dyn_full_convolution_2d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float> a(d1, d1);
    etl::dyn_matrix<float> b(d2, d2);
    etl::dyn_matrix<float> c(d1+d2-1, d1+d2-1);

    std::cout << "dyn_full_convolution_2d_s" << "(" << d1 << "," << d2 << ")" << std::endl;
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

    std::cout << "fast_same_convolution_2d_d" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_same_convolution_2d(a, b, c);
}

void bench_dyn_same_convolution_2d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2, d2);
    etl::dyn_matrix<double> c(d1, d1);

    std::cout << "dyn_same_convolution_2d_d" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_same_convolution_2d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_same_convolution_2d_s(){
    etl::fast_matrix<float, D1, D1> a;
    etl::fast_matrix<float, D2, D2> b;
    etl::fast_matrix<float, D1, D1> c;

    std::cout << "fast_same_convolution_2d_s" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_same_convolution_2d(a, b, c);
}

void bench_dyn_same_convolution_2d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float> a(d1, d1);
    etl::dyn_matrix<float> b(d2, d2);
    etl::dyn_matrix<float> c(d1, d1);

    std::cout << "dyn_same_convolution_2d_s" << "(" << d1 << "," << d2 << ")" << std::endl;
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

    std::cout << "fast_valid_convolution_2d_d" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_valid_convolution_2d(a, b, c);
}

void bench_dyn_valid_convolution_2d_d(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2, d2);
    etl::dyn_matrix<double> c(d1-d2+1, d1-d2+1);

    std::cout << "dyn_valid_convolution_2d_d" << "(" << d1 << "," << d2 << ")" << std::endl;
    measure_valid_convolution_2d(a, b, c);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_valid_convolution_2d_s(){
    etl::fast_matrix<float, D1, D1> a;
    etl::fast_matrix<float, D2, D2> b;
    etl::fast_matrix<float, D1-D2+1, D1-D2+1> c;

    std::cout << "fast_valid_convolution_2d_s" << "(" << D1 << "," << D2 << ")" << std::endl;
    measure_valid_convolution_2d(a, b, c);
}

void bench_dyn_valid_convolution_2d_s(std::size_t d1, std::size_t d2){
    etl::dyn_matrix<float> a(d1, d1);
    etl::dyn_matrix<float> b(d2, d2);
    etl::dyn_matrix<float> c(d1-d2+1, d1-d2+1);

    std::cout << "dyn_valid_convolution_2d_s" << "(" << d1 << "," << d2 << ")" << std::endl;
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

    bench_fast_matrix_sigmoid<16, 256>();
    bench_fast_matrix_sigmoid<256, 128>();
    bench_dyn_matrix_sigmoid(16, 256);
    bench_dyn_matrix_sigmoid(256, 128);

    bench_fast_vector_simple<4096>();
    bench_fast_vector_simple<16384>();
    bench_dyn_vector_simple(4096);
    bench_dyn_vector_simple(16384);

    bench_fast_matrix_simple<16, 256>();
    bench_fast_matrix_simple<256, 128>();
    bench_dyn_matrix_simple(16, 256);
    bench_dyn_matrix_simple(256, 128);

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


void bench_smart_1(std::size_t d){
    etl::dyn_matrix<double> A(d, d);
    etl::dyn_matrix<double> B(d, d);
    etl::dyn_matrix<double> C(d, d);
    etl::dyn_matrix<double> result(d, d);

    measure("A * (B + C) (" + std::to_string(d) + "x" + std::to_string(d) + ")", [&A, &B, &C, &result](){
        result = A * (B + C);
    }, A, B, C);
}

void bench_smart_2(std::size_t dd){
    etl::dyn_matrix<double> A(dd, dd);
    etl::dyn_vector<double> b(dd);
    etl::dyn_vector<double> c(dd);
    etl::dyn_vector<double> d(dd);
    etl::dyn_vector<double> result(dd);

    measure("A * (b + c + d) (" + std::to_string(dd) + "x" + std::to_string(dd) + ")", [&A, &b, &c, &d, &result](){
        result = A * (b + c + d);
    }, A, b, c, d);
}

void bench_smart(){
    std::cout << "Start Smart benchmarking...\n";

    bench_smart_1(50);
    bench_smart_1(100);
    bench_smart_1(250);
    bench_smart_1(500);

    bench_smart_2(50);
    bench_smart_2(100);
    bench_smart_2(250);
    bench_smart_2(500);
}

} //end of anonymous namespace

int main(int argc, char* argv[]){
    std::vector<std::string> args;

    for(int i = 1; i < argc; ++i){
        args.emplace_back(argv[i]);
    }

    bool smart = false;

    for(auto& arg : args){
        if(arg == "smart"){
            smart = true;
        }
    }

    if(smart){
        bench_smart();
    } else {
        bench_standard();
    }

    return 0;
}
