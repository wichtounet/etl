//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#include "etl/fast_vector.hpp"
#include "etl/fast_matrix.hpp"
#include "etl/convolution.hpp"

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

    for(std::size_t i = 0; i < 1000; ++i){
        randomize(references...);
        auto start_time = timer_clock::now();
        functor();
        auto end_time = timer_clock::now();
        auto duration = std::chrono::duration_cast<microseconds>(end_time - start_time);
        duration_acc += duration.count();
    }

    std::cout << title << " took " << duration_str(duration_acc) << " (reference: " << reference << ")" << std::endl;
}

} //end of anonymous namespace

etl::fast_vector<double, 4096> double_vector_a;
etl::fast_vector<double, 4096> double_vector_b;
etl::fast_vector<double, 4096> double_vector_c;

etl::fast_matrix<double, 16, 256> double_matrix_a;
etl::fast_matrix<double, 16, 256> double_matrix_b;
etl::fast_matrix<double, 16, 256> double_matrix_c;

etl::fast_matrix<double, 256, 128> double_matrix_d;
etl::fast_matrix<double, 256, 128> double_matrix_e;
etl::fast_matrix<double, 256, 128> double_matrix_f;

etl::fast_matrix<double, 128, 128> double_conv_a;
etl::fast_matrix<double, 32, 32> double_conv_b;
etl::fast_matrix<double, 159, 159> double_conv_c;

int main(){
    measure("fast_vector_simple(4096)", "72ms", [](){
        double_vector_c = 3.5 * double_vector_a + etl::sigmoid(1.0 + double_vector_b);
    }, double_vector_a, double_vector_b);

    measure("fast_matrix_simple(16,256)(4096)", "72ms", [](){
        double_matrix_c = 3.5 * double_matrix_a + etl::sigmoid(1.0 + double_matrix_b);
    }, double_matrix_a, double_matrix_b);

    measure("fast_matrix_simple(256,128)", "580ms", [](){
        double_matrix_f = 3.5 * double_matrix_d + etl::sigmoid(1.0 + double_matrix_e);
    }, double_matrix_d, double_matrix_e);

    measure("fast_matrix_full_convolve(128,128)", "25s", [](){
        etl::convolve_2d_full(double_conv_a, double_conv_b, double_conv_c);
    }, double_conv_a, double_conv_b);

    measure("fast_matrix_valid_convolve(128,128)", "40s", [](){
        etl::convolve_2d_valid(double_conv_a, double_conv_b, double_conv_c);
    }, double_conv_a, double_conv_b);

    return 0;
}
