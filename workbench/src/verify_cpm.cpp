//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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

void default_case(){
    etl::dyn_matrix<float, 4> A(128, 64, 100, 100);
    etl::dyn_matrix<float, 1> B(64);
    etl::dyn_matrix<float, 4> C(128, 64, 100, 100);

    for(size_t i = 0; i < 5; ++i){
        C = etl::bias_add_4d(A, B);
    }

    auto start_time = timer_clock::now();

    for(size_t i = 0; i < 10; ++i){
        C = etl::bias_add_4d(A, B);
    }

    auto end_time = timer_clock::now();
    auto duration = std::chrono::duration_cast<milliseconds>(end_time - start_time);

    std::cout << "default: " << duration.count() << "ms" << std::endl;
    std::cout << "   mean: " << (duration.count() / 10.0) << "ms" << std::endl;
}

void std_case(){
    etl::dyn_matrix<float, 4> A(128, 64, 100, 100);
    etl::dyn_matrix<float, 1> B(64);
    etl::dyn_matrix<float, 4> C(128, 64, 100, 100);

    for(size_t i = 0; i < 5; ++i){
        C = selected_helper(etl::bias_add_impl::STD, etl::bias_add_4d(A, B));
    }

    auto start_time = timer_clock::now();

    for(size_t i = 0; i < 10; ++i){
        C = selected_helper(etl::bias_add_impl::STD, etl::bias_add_4d(A, B));
    }

    auto end_time = timer_clock::now();
    auto duration = std::chrono::duration_cast<milliseconds>(end_time - start_time);

    std::cout << "    std: " << duration.count() << "ms" << std::endl;
    std::cout << "   mean: " << (duration.count() / 10.0) << "ms" << std::endl;
}

void vec_case(){
    etl::dyn_matrix<float, 4> A(128, 64, 100, 100);
    etl::dyn_matrix<float, 1> B(64);
    etl::dyn_matrix<float, 4> C(128, 64, 100, 100);

    for(size_t i = 0; i < 5; ++i){
        C = selected_helper(etl::bias_add_impl::VEC, etl::bias_add_4d(A, B));
    }

    auto start_time = timer_clock::now();

    for(size_t i = 0; i < 10; ++i){
        C = selected_helper(etl::bias_add_impl::VEC, etl::bias_add_4d(A, B));
    }

    auto end_time = timer_clock::now();
    auto duration = std::chrono::duration_cast<milliseconds>(end_time - start_time);

    std::cout << "    vec: " << duration.count() << "ms" << std::endl;
    std::cout << "   mean: " << (duration.count() / 10.0) << "ms" << std::endl;
}

int main(){
    default_case();
    std_case();
    vec_case();

    return 0;
}
