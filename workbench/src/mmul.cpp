//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#include "etl/etl.hpp"

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

std::string duration_str(size_t duration_us){
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
    for(size_t i = 0; i < 100; ++i){
        randomize(references...);
        functor();
    }

    size_t duration_acc = 0;

    for(size_t i = 0; i < 100; ++i){
        randomize(references...);
        auto start_time = timer_clock::now();
        functor();
        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }

    std::cout << title << " took " << duration_str(duration_acc) << " (reference: " << reference << ")\n";
}

template<size_t D1, size_t D2, size_t D3>
void bench_direct(const std::string& reference){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D2, D3> b;
    etl::fast_matrix<double, D1, D3> c;

    measure("direct(" + std::to_string(D1) + "," + std::to_string(D2) + "," + std::to_string(D3)  + ")", reference, [&a, &b, &c](){
        etl::mmul(a,b,c);
    }, a, b);
}

template<size_t D1, size_t D2, size_t D3>
void bench_standard(const std::string& reference){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D2, D3> b;
    etl::fast_matrix<double, D1, D3> t;
    etl::fast_matrix<double, D1, D3> c;

    measure("standard(" + std::to_string(D1) + "," + std::to_string(D2) + "," + std::to_string(D3)  + ")", reference, [&a, &b, &c, &t](){
        c = etl::mmul(a,b,t);
    }, a, b);
}

template<size_t D1, size_t D2, size_t D3>
void bench_new(const std::string& reference){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D2, D3> b;
    etl::fast_matrix<double, D1, D3> c;

    measure("new(" + std::to_string(D1) + "," + std::to_string(D2) + "," + std::to_string(D3)  + ")", reference, [&a, &b, &c](){
        c = etl::mmul(a,b);
    }, a, b);
}


template<size_t D1, size_t D2, size_t D3>
void bench_strassen_direct(const std::string& reference){
    etl::fast_matrix<double, D1, D2> a;
    etl::fast_matrix<double, D2, D3> b;
    etl::fast_matrix<double, D1, D3> c;

    measure("strassen_direct(" + std::to_string(D1) + "," + std::to_string(D2) + "," + std::to_string(D3)  + ")", reference, [&a, &b, &c](){
        etl::strassen_mmul(a,b,c);
    }, a, b);
}
template<size_t D1, size_t D2, size_t D3>
double bench_lazy(const std::string& reference){
    etl::fast_matrix<double, D1, D2> a(0.0);
    etl::fast_matrix<double, D2, D3> b(0.0);
    etl::fast_matrix<double, D1, D3> c(0.0);

    double count = 0;
    measure("lazy(" + std::to_string(D1) + "," + std::to_string(D2) + "," + std::to_string(D3)  + ")", reference, [&a, &b, &c, &count](){
        c = etl::lazy_mmul(a,b);
        count += c(3,3);
    }, a, b);

    //std::cout << count << std::endl;
    return count;
}

void bench_stack(){
    std::cout << "Start mmul benchmarking...\n";
    std::cout << "... all structures are on stack\n\n";

    bench_direct<16, 16, 16>("frigg:400us");
    bench_direct<32, 32, 32>("frigg:3ms");
    bench_direct<64, 64, 64>("frigg:24.3ms");
    bench_direct<64, 256, 32>("frigg:48.1ms");
    bench_direct<128, 128, 128>("frigg:205ms");
    bench_direct<256, 256, 256>("frigg:2.6s");

    bench_standard<16, 16, 16>("frigg:400us");
    bench_standard<32, 32, 32>("frigg:3ms");
    bench_standard<64, 64, 64>("frigg:24.4ms");
    bench_standard<64, 256, 32>("frigg:47ms");
    bench_standard<128, 128, 128>("frigg:205ms");
    bench_standard<256, 256, 256>("frigg:2.6s");

    bench_new<16, 16, 16>("frigg:402us");
    bench_new<32, 32, 32>("frigg:3.1ms");
    bench_new<64, 64, 64>("frigg:28ms");
    bench_new<64, 256, 32>("frigg:50ms");
    bench_new<128, 128, 128>("frigg:212ms");
    bench_new<256, 256, 256>("frigg:2.6s");

    bench_strassen_direct<16, 16, 16>("frigg:1.6ms");
    bench_strassen_direct<32, 32, 32>("frigg:11ms");
    bench_strassen_direct<64, 64, 64>("frigg:80ms");
    bench_strassen_direct<64, 256, 32>("frigg:4s");
    bench_strassen_direct<128, 128, 128>("frigg:600ms");
    bench_strassen_direct<256, 256, 256>("frigg:4s");

    double count = 0;
    count += bench_lazy<16, 16, 16>("frigg:400us");
    count += bench_lazy<32, 32, 32>("frigg:3ms");
    count += bench_lazy<64, 64, 64>("frigg:26ms");
    count += bench_lazy<64, 256, 32>("frigg:55ms");
    count += bench_lazy<128, 128, 128>("frigg:222ms");
    count += bench_lazy<256, 256, 256>("frigg:2.68s");
    std::cout << count << std::endl;
}

} //end of anonymous namespace

int main(){
    bench_stack();

    return 0;
}
