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
#include "etl/convolution.hpp"
#include "etl/multiplication.hpp"

#ifdef ETL_VECTORIZE
#ifdef __SSE3__
#define TEST_SSE
#endif
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

template<typename Functor, typename... T>
void measure(const std::string& title, const std::string& reference, Functor&& functor, T&... references){
    for(std::size_t i = 0; i < 100; ++i){
        randomize(references...);
        functor();
    }

    std::size_t duration_acc = 0;

    for(std::size_t i = 0; i < 100; ++i){
        randomize(references...);
        auto start_time = timer_clock::now();
        functor();
        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }

    std::cout << title << " took " << duration_str(duration_acc) << " (reference: " << reference << ")\n";
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

template<std::size_t D1, std::size_t D2>
void bench_fast_full_convolution_1d(const std::string& reference){
    etl::fast_matrix<double, D1> a;
    etl::fast_matrix<double, D2> b;
    etl::fast_matrix<double, D1+D2-1> c;

    measure("fast_full_convolution_1d(default)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::convolve_1d_full(a, b, c);}
        , a, b);

    measure("fast_full_convolution_1d(std)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::impl::standard::conv1_full(a, b, c);}
        , a, b);

#ifdef TEST_SSE
    measure("fast_full_convolution_1d(sse)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::impl::sse::dconv1_full(a, b, c);}
        , a, b);
#endif
}

void bench_dyn_full_convolution_1d(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<double,1> a(d1);
    etl::dyn_matrix<double,1> b(d2);
    etl::dyn_matrix<double,1> c(d1+d2-1);

    measure("dyn_full_convolution_1d(default)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::convolve_1d_full(a, b, c);}
        , a, b);

    measure("dyn_full_convolution_1d(std)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::impl::standard::conv1_full(a, b, c);}
        , a, b);

#ifdef TEST_SSE
    measure("dyn_full_convolution_1d(sse)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::impl::sse::dconv1_full(a, b, c);}
        , a, b);
#endif
}

template<std::size_t D1, std::size_t D2>
void bench_fast_valid_convolution_1d_d(const std::string& reference){
    etl::fast_matrix<double, D1> a;
    etl::fast_matrix<double, D2> b;
    etl::fast_matrix<double, D1-D2+1> c;

    measure("fast_valid_convolution_1d_d(default)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::convolve_1d_valid(a, b, c);}
        , a, b);

    measure("fast_valid_convolution_1d_d(std)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::impl::standard::conv1_valid(a, b, c);}
        , a, b);

#ifdef TEST_SSE
    measure("fast_valid_convolution_1d_d(sse)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::impl::sse::dconv1_valid(a, b, c);}
        , a, b);
#endif
}

void bench_dyn_valid_convolution_1d_d(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<double,1> a(d1);
    etl::dyn_matrix<double,1> b(d2);
    etl::dyn_matrix<double,1> c(d1-d2+1);

    measure("dyn_valid_convolution_1d_d(default)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::convolve_1d_valid(a, b, c);}
        , a, b);

    measure("dyn_valid_convolution_1d_d(std)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::impl::standard::conv1_valid(a, b, c);}
        , a, b);

#ifdef TEST_SSE
    measure("dyn_valid_convolution_1d_d(sse)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::impl::sse::dconv1_valid(a, b, c);}
        , a, b);
#endif
}

template<std::size_t D1, std::size_t D2>
void bench_fast_valid_convolution_1d_s(const std::string& reference){
    etl::fast_matrix<float, D1> a;
    etl::fast_matrix<float, D2> b;
    etl::fast_matrix<float, D1-D2+1> c;

    measure("fast_valid_convolution_1d_s(default)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::convolve_1d_valid(a, b, c);}
        , a, b);

    measure("fast_valid_convolution_1d_s(std)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::impl::standard::conv1_valid(a, b, c);}
        , a, b);

#ifdef TEST_SSE
    measure("fast_valid_convolution_1d_s(sse)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::impl::sse::sconv1_valid(a, b, c);}
        , a, b);
#endif
}

void bench_dyn_valid_convolution_1d_s(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<float,1> a(d1);
    etl::dyn_matrix<float,1> b(d2);
    etl::dyn_matrix<float,1> c(d1-d2+1);

    measure("dyn_valid_convolution_1d_s(default)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::convolve_1d_valid(a, b, c);}
        , a, b);

    measure("dyn_valid_convolution_1d_s(std)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::impl::standard::conv1_valid(a, b, c);}
        , a, b);

#ifdef TEST_SSE
    measure("dyn_valid_convolution_1d_s(std)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::impl::sse::sconv1_valid(a, b, c);}
        , a, b);
#endif
}

template<std::size_t D1, std::size_t D2>
void bench_fast_full_convolution(const std::string& reference){
    etl::fast_matrix<double, D1, D1> a;
    etl::fast_matrix<double, D2, D2> b;
    etl::fast_matrix<double, D1+D2-1, D1+D2-1> c;

    measure("fast_full_convolution(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::convolve_2d_full(a, b, c);}
        , a, b);
}

void bench_dyn_full_convolution(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2, d2);
    etl::dyn_matrix<double> c(d1+d2-1, d1+d2-1);

    measure("dyn_full_convolution(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::convolve_2d_full(a, b, c);}
        , a, b);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_valid_convolution(const std::string& reference){
    etl::fast_matrix<double, D1, D1> a;
    etl::fast_matrix<double, D2, D2> b;
    etl::fast_matrix<double, D1-D2+1, D1-D2+1> c;

    measure("fast_valid_convolution(default)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::convolve_2d_valid(a, b, c);}
        , a, b);

    measure("fast_valid_convolution(std)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::impl::standard::conv2_valid(a, b, c);}
        , a, b);

#ifdef TEST_SSE
    measure("fast_valid_convolution(sse)(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::impl::sse::dconv2_valid(a, b, c);}
        , a, b);
#endif
}

void bench_dyn_valid_convolution(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<double> a(d1, d1);
    etl::dyn_matrix<double> b(d2, d2);
    etl::dyn_matrix<double> c(d1-d2+1, d1-d2+1);

    measure("dyn_valid_convolution(default)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::convolve_2d_valid(a, b, c);}
        , a, b);

    measure("dyn_valid_convolution(std)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::impl::standard::conv2_valid(a, b, c);}
        , a, b);

#ifdef TEST_SSE
    measure("dyn_valid_convolution(sse)(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::impl::sse::dconv2_valid(a, b, c);}
        , a, b);
#endif
}

template<std::size_t D1, std::size_t D2>
void bench_fast_mmul(const std::string& reference){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D2, D1> b;
    etl::fast_matrix<double, D1, D1> c;

    measure("fast_mmul(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::mmul(a, b, c);}
        , a, b);
}

void bench_dyn_mmul(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<double> a(d1, d2);
    etl::dyn_matrix<double> b(d2, d1);
    etl::dyn_matrix<double> c(d1, d1);

    measure("dyn_mmul(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::mmul(a, b, c);}
        , a, b);
}

template<std::size_t D1, std::size_t D2>
void bench_fast_mmul_s(const std::string& reference){
    etl::fast_matrix<float, D1, D2> a;
    etl::fast_matrix<float, D2, D1> b;
    etl::fast_matrix<float, D1, D1> c;

    measure("fast_mmul_s(" + std::to_string(D1) + "," + std::to_string(D2) + ")", reference,
        [&a, &b, &c](){etl::mmul(a, b, c);}
        , a, b);
}

void bench_dyn_mmul_s(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<float> a(d1, d2);
    etl::dyn_matrix<float> b(d2, d1);
    etl::dyn_matrix<float> c(d1, d1);

    measure("dyn_mmul_s(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::mmul(a, b, c);}
        , a, b);
}

void bench_dyn_strassen_mmul(std::size_t d1, std::size_t d2, const std::string& reference){
    etl::dyn_matrix<double> a(d1, d2);
    etl::dyn_matrix<double> b(d2, d1);
    etl::dyn_matrix<double> c(d1, d1);

    measure("dyn_strassen_mmul(" + std::to_string(d1) + "," + std::to_string(d2) + ")", reference,
        [&a, &b, &c](){etl::strassen_mmul(a, b, c);}
        , a, b);
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

    bench_fast_valid_convolution_1d_d<1024, 64>("TODOms");
    bench_fast_valid_convolution_1d_d<2048, 128>("TODOms");
    bench_dyn_valid_convolution_1d_d(1024, 64, "TODOms");
    bench_dyn_valid_convolution_1d_d(2048, 64, "TODOms");

    bench_fast_valid_convolution_1d_s<2*1024, 2*64>("TODOms");
    bench_fast_valid_convolution_1d_s<2*2048, 2*128>("TODOms");
    bench_dyn_valid_convolution_1d_s(2*1024, 2*64, "TODOms");
    bench_dyn_valid_convolution_1d_s(2*2048, 2*64, "TODOms");

    bench_fast_full_convolution_1d<2*1024, 2*64>("TODOms");
    bench_fast_full_convolution_1d<2*2048, 2*128>("TODOms");
    bench_dyn_full_convolution_1d(2*1024, 2*64, "TODOms");
    bench_dyn_full_convolution_1d(2*2048, 2*64, "TODOms");

    bench_fast_full_convolution<64, 32>("TODOms");
    bench_fast_full_convolution<128, 32>("TODOms");
    bench_dyn_full_convolution(64, 32, "TODOms");
    bench_dyn_full_convolution(128, 32, "TODOms");

    bench_fast_valid_convolution<64, 32>("TODOms");
    bench_fast_valid_convolution<128, 32>("TODOms");
    bench_dyn_valid_convolution(64, 32, "TODOms");
    bench_dyn_valid_convolution(128, 32, "TODOms");

    bench_fast_mmul<64, 32>("TODOms");
    bench_fast_mmul<128, 64>("TODOms");
    bench_fast_mmul<256, 128>("TODOms");
    bench_dyn_mmul(64, 32, "TODOms");
    bench_dyn_mmul(128, 64, "TODOms");
    bench_dyn_mmul(256, 128, "TODOms");
    bench_dyn_mmul(512, 256, "TODOms");

    bench_fast_mmul_s<2*64, 2*32>("TODOms");
    bench_fast_mmul_s<2*128, 2*64>("TODOms");
    bench_fast_mmul_s<2*256, 2*128>("TODOms");
    bench_dyn_mmul_s(64, 32, "TODOms");
    bench_dyn_mmul_s(128, 64, "TODOms");
    bench_dyn_mmul_s(256, 128, "TODOms");
    bench_dyn_mmul_s(512, 256, "TODOms");

    bench_dyn_strassen_mmul(16, 16, "TODOms");
    bench_dyn_strassen_mmul(32, 32, "TODOms");
    bench_dyn_strassen_mmul(64, 32, "TODOms");
    bench_dyn_strassen_mmul(64, 64, "TODOms");
    bench_dyn_strassen_mmul(128, 64, "TODOms");
    bench_dyn_strassen_mmul(256, 128, "TODOms");
}

} //end of anonymous namespace

int main(){
    bench_stack();

    return 0;
}
