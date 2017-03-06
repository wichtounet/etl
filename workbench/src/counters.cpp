//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#define ETL_COUNTERS
#define ETL_COUNTERS_VERBOSE

#include "etl/etl.hpp"

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::milliseconds milliseconds;

float fake = 0;

/*
 *
 * Current values are (alloc/gpu_to_cpu/cpu_to_gpu):
 * Simple: 3 / 0 / 2 (Optimal!)
 * Basic: 50 / 30 / 10
 * Sub: 960 / 640 / 160
 */

void simple(){
    etl::dyn_matrix<float, 2> A(4096, 4096);
    etl::dyn_matrix<float, 2> B(4096, 4096);
    etl::dyn_matrix<float, 2> C(4096, 4096);

    A = etl::normal_generator<float>(1.0, 0.0);
    B = etl::normal_generator<float>(1.0, 0.0);
    C = etl::normal_generator<float>(1.0, 0.0);

    etl::reset_counters();

    std::cout << "Simple" << std::endl;

    for (size_t i = 0; i < 10; ++i) {
        C = A * B;
        fake += etl::mean(C);
    }

    etl::dump_counters();
}

void basic(){
    etl::dyn_matrix<float, 2> A(4096, 4096);
    etl::dyn_matrix<float, 2> B(4096, 4096);
    etl::dyn_matrix<float, 2> C(4096, 4096);
    etl::dyn_matrix<float, 2> D(4096, 4096);

    A = etl::normal_generator<float>(1.0, 0.0);
    B = etl::normal_generator<float>(1.0, 0.0);
    C = etl::normal_generator<float>(1.0, 0.0);
    D = etl::normal_generator<float>(1.0, 0.0);

    etl::reset_counters();

    std::cout << "Basic" << std::endl;

    for (size_t i = 0; i < 10; ++i) {
        C = A * B * B;
        D = A * trans(A);
        D *= 1.1;
        D += C;
        fake += etl::mean(D);
    }
    etl::dump_counters();
}

void sub(){
    etl::dyn_matrix<float, 3> A(16, 2048, 2048);
    etl::dyn_matrix<float, 3> B(16, 2048, 2048);
    etl::dyn_matrix<float, 3> C(16, 2048, 2048);
    etl::dyn_matrix<float, 3> D(16, 2048, 2048);

    A = etl::normal_generator<float>(1.0, 0.0);
    B = etl::normal_generator<float>(1.0, 0.0);
    C = etl::normal_generator<float>(1.0, 0.0);
    D = etl::normal_generator<float>(1.0, 0.0);

    etl::reset_counters();

    std::cout << "Sub" << std::endl;

    for (size_t i = 0; i < 10; ++i) {
        for (size_t k = 0; k < 16; ++k) {
            C(k) = A(k) * B(k) * B(k);
            D(k) += C(k);
            D(k) *= 1.1;
            fake += etl::mean(D(k));
        }
    }

    etl::dump_counters();
}

int main(){
    auto start_time = timer_clock::now();

    simple();
    basic();
    sub();

    auto end_time = timer_clock::now();
    auto duration = std::chrono::duration_cast<milliseconds>(end_time - start_time);

    std::cout << "duration: " << duration.count() << "ms" << std::endl;

    return (int) fake;
}
