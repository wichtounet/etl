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

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::milliseconds milliseconds;
typedef std::chrono::microseconds microseconds;

namespace {

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
        functor(start_time);
        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }

    return duration_acc;
}

template<typename Functor, typename... T>
void measure(const std::string& title, const std::string& reference, Functor&& functor, T&... references){
    auto duration_acc = measure_only(std::forward<Functor>(functor), references...);
    std::cout << title << " took " << duration_str(duration_acc) << " (reference: " << reference << ")\n";
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

template<std::size_t D>
void bench_fast_vector_simple(const std::string& reference){
    etl::fast_vector<double, D> a;
    etl::fast_vector<double, D> b;
    etl::fast_vector<double, D> c;

    measure("fast_vector_simple(" + std::to_string(D) + ")", reference, [&a, &b, &c](){
        c = 3.5 * a + etl::sigmoid(1.0 + b);
    }, a, b);
}

void bench_dyn_vector_simple(std::size_t d, const std::string& reference){
    etl::dyn_vector<double> a(d);
    etl::dyn_vector<double> b(d);
    etl::dyn_vector<double> c(d);

    measure("dyn_vector_simple(" + std::to_string(d) + ")", reference, [&a, &b, &c](){
        c = 3.5 * a + etl::sigmoid(1.0 + b);
    }, a, b);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_matrix_simple(const std::string& reference){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D1, D2> b;
    etl::fast_matrix<double, D1, D2> c;

    measure("fast_matrix_simple(" + std::to_string(D1) + "," + std::to_string(D2) + ")(" + std::to_string(D1 * D2) + ")", reference,
        [&a, &b, &c](){c = 3.5 * a + etl::sigmoid(1.0 + b);}
        , a, b);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_matrix_sigmoid(const std::string& reference){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D1, D2> b;
    etl::fast_matrix<double, D1, D2> c;

    measure("fast_matrix_sigmoid(" + std::to_string(D1) + "," + std::to_string(D2) + ")(" + std::to_string(D1 * D2) + ")", reference,
        [&a, &b, &c](){c = etl::sigmoid(1.0 + b);}
        , a, b);
}

void bench_dyn_matrix_simple(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<double> a(d1, d2);
    etl::dyn_matrix<double> b(d1, d2);
    etl::dyn_matrix<double> c(d1, d2);

    measure("dyn_matrix_simple(" + std::to_string(d1) + "," + std::to_string(d2) + ")(" + std::to_string(d1 * d2) + ")", reference,
        [&a, &b, &c](){c = 3.5 * a + etl::sigmoid(1.0 + b);}
        , a, b);
}

void bench_dyn_matrix_sigmoid(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<double> a(d1, d2);
    etl::dyn_matrix<double> b(d1, d2);
    etl::dyn_matrix<double> c(d1, d2);

    measure("dyn_matrix_sigmoid(" + std::to_string(d1) + "," + std::to_string(d2) + ")(" + std::to_string(d1 * d2) + ")", reference,
        [&a, &b, &c](){c = etl::sigmoid(1.0 + b);}
        , a, b);
}

template<typename A, typename B, typename C>
void measure_full_convolution_1d(A& a, B& b, C& c){
    measure_sub("default", [&a, &b, &c](auto&){c = etl::conv_1d_full(a, b);} , a, b);

    measure_sub("std", [&a, &b, &c](auto&){etl::impl::standard::conv1_full(a, b, c);} , a, b);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub<F>("sse", [&a, &b, &c](auto&){etl::impl::sse::sconv1_full(a, b, c);} , a, b);
    measure_sub<!F>("sse", [&a, &b, &c](auto&){etl::impl::sse::dconv1_full(a, b, c);} , a, b);
#endif

#ifdef TEST_AVX
    measure_sub<F>("avx", [&a, &b, &c](auto&){etl::impl::avx::sconv1_full(a, b, c);} , a, b);
    measure_sub<!F>("avx", [&a, &b, &c](auto&){etl::impl::avx::dconv1_full(a, b, c);} , a, b);
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

template<typename A, typename B, typename C>
void measure_same_convolution_1d(A& a, B& b, C& c){
    measure_sub("default", [&a, &b, &c](auto&){c = etl::conv_1d_same(a, b);} , a, b);

    measure_sub("std", [&a, &b, &c](auto&){etl::impl::standard::conv1_same(a, b, c);} , a, b);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub<F>("sse", [&a, &b, &c](auto&){etl::impl::sse::sconv1_same(a, b, c);} , a, b);
    measure_sub<!F>("sse", [&a, &b, &c](auto&){etl::impl::sse::dconv1_same(a, b, c);} , a, b);
#endif

#ifdef TEST_AVX
    measure_sub<F>("avx", [&a, &b, &c](auto&){etl::impl::avx::sconv1_same(a, b, c);} , a, b);
    measure_sub<!F>("avx", [&a, &b, &c](auto&){etl::impl::avx::dconv1_same(a, b, c);} , a, b);
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

template<typename A, typename B, typename C>
void measure_valid_convolution_1d(A& a, B& b, C& c){
    measure_sub("default", [&a, &b, &c](auto&){c = etl::conv_1d_valid(a, b);} , a, b);

    measure_sub("std", [&a, &b, &c](auto&){etl::impl::standard::conv1_valid(a, b, c);} , a, b);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub<F>("sse", [&a, &b, &c](auto&){etl::impl::sse::sconv1_valid(a, b, c);} , a, b);
    measure_sub<!F>("sse", [&a, &b, &c](auto&){etl::impl::sse::dconv1_valid(a, b, c);} , a, b);
#endif

#ifdef TEST_AVX
    measure_sub<F>("avx", [&a, &b, &c](auto&){etl::impl::avx::sconv1_valid(a, b, c);} , a, b);
    measure_sub<!F>("avx", [&a, &b, &c](auto&){etl::impl::avx::dconv1_valid(a, b, c);} , a, b);
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

template<typename A, typename B, typename C>
void measure_full_convolution_2d(A& a, B& b, C& c){
    measure_sub("default", [&a, &b, &c](auto&){c = etl::conv_2d_full(a, b);} , a, b);

    measure_sub("std", [&a, &b, &c](auto&){etl::impl::standard::conv2_full(a, b, c);} , a, b);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub<F>("sse", [&a, &b, &c](auto&){etl::impl::sse::sconv2_full(a, b, c);} , a, b);
    measure_sub<!F>("sse", [&a, &b, &c](auto&){etl::impl::sse::dconv2_full(a, b, c);} , a, b);
#endif

#ifdef TEST_AVX
    measure_sub<F>("avx", [&a, &b, &c](auto&){etl::impl::avx::sconv2_full(a, b, c);} , a, b);
    measure_sub<!F>("avx", [&a, &b, &c](auto&){etl::impl::avx::dconv2_full(a, b, c);} , a, b);
#endif

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

template<typename A, typename B, typename C>
void measure_same_convolution_2d(A& a, B& b, C& c){
    measure_sub("default", [&a, &b, &c](auto&){c = etl::conv_2d_same(a, b);} , a, b);

    measure_sub("std", [&a, &b, &c](auto&){etl::impl::standard::conv2_same(a, b, c);} , a, b);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub<F>("sse", [&a, &b, &c](auto&){etl::impl::sse::sconv2_same(a, b, c);} , a, b);
    measure_sub<!F>("sse", [&a, &b, &c](auto&){etl::impl::sse::dconv2_same(a, b, c);} , a, b);
#endif

#ifdef TEST_AVX
    measure_sub<F>("avx", [&a, &b, &c](auto&){etl::impl::avx::sconv2_same(a, b, c);} , a, b);
    measure_sub<!F>("avx", [&a, &b, &c](auto&){etl::impl::avx::dconv2_same(a, b, c);} , a, b);
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

template<typename A, typename B, typename C>
void measure_valid_convolution_2d(A& a, B& b, C& c){
    measure_sub("default", [&a, &b, &c](auto&){c = etl::conv_2d_valid(a, b);} , a, b);

    measure_sub("std", [&a, &b, &c](auto&){etl::impl::standard::conv2_valid(a, b, c);} , a, b);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

#ifdef TEST_SSE
    measure_sub<F>("sse", [&a, &b, &c](auto&){etl::impl::sse::sconv2_valid(a, b, c);} , a, b);
    measure_sub<!F>("sse", [&a, &b, &c](auto&){etl::impl::sse::dconv2_valid(a, b, c);} , a, b);
#endif

#ifdef TEST_AVX
    measure_sub<F>("avx", [&a, &b, &c](auto&){etl::impl::avx::sconv2_valid(a, b, c);} , a, b);
    measure_sub<!F>("avx", [&a, &b, &c](auto&){etl::impl::avx::dconv2_valid(a, b, c);} , a, b);
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

template<typename A, typename B, typename C>
void measure_mmul(A& a, B& b, C& c){
    measure_sub("default", [&a, &b, &c](auto&){c = etl::mmul(a, b);} , a, b);

    measure_sub("std", [&a, &b, &c](auto&){etl::impl::standard::mmul(a, b, c);} , a, b);

    measure_sub("lazy", [&a, &b, &c](auto&){c = etl::lazy_mmul(a, b);} , a, b);

    constexpr const bool F = std::is_same<float, typename A::value_type>::value;

    measure_sub<F>("eblas", [&a, &b, &c](auto&){etl::impl::eblas::fast_sgemm(a, b, c);} , a, b);
    measure_sub<!F>("eblas", [&a, &b, &c](auto&){etl::impl::eblas::fast_dgemm(a, b, c);} , a, b);

#ifdef TEST_BLAS
    measure_sub<F>("blas", [&a, &b, &c](auto&){etl::impl::blas::sgemm(a, b, c);} , a, b);
    measure_sub<!F>("blas", [&a, &b, &c](auto&){etl::impl::blas::dgemm(a, b, c);} , a, b);
#endif

    measure_sub("strassen", [&a, &b, &c](auto&){*etl::strassen_mmul(a, b, c);} , a, b);
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

void bench_stack(){
    std::cout << "Start benchmarking...\n";
    std::cout << "... all structures are on stack\n\n";

    bench_fast_matrix_sigmoid<16, 256>("TODOms");
    bench_fast_matrix_sigmoid<256, 128>("TODOms");
    bench_dyn_matrix_sigmoid(16, 256, "TODOms");
    bench_dyn_matrix_sigmoid(256, 128, "TODOms");

    bench_fast_vector_simple<4096>("TODOms");
    bench_fast_vector_simple<16384>("TODOms");
    bench_dyn_vector_simple(4096, "TODOms");
    bench_dyn_vector_simple(16384, "TODOms");

    bench_fast_matrix_simple<16, 256>("TODOms");
    bench_fast_matrix_simple<256, 128>("TODOms");
    bench_dyn_matrix_simple(16, 256, "TODOms");
    bench_dyn_matrix_simple(256, 128, "TODOms");

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

} //end of anonymous namespace

int main(){
    bench_stack();

    return 0;
}
