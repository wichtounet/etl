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

#define IF_DEBUG if(false)

#include "etl/etl.hpp"

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::milliseconds milliseconds;

float fake = 0;

/*
 *
 * Current values are (alloc/gpu_to_cpu/cpu_to_gpu):
 * Simple: 3 / 0 / 2 (Optimal!)
 * Basic: 15 / 0 / 3 (Optimal!)
 * Sub: 960 / 640 / 160
 * ML: 960 / 640 / 160
 */

void simple(){
    etl::dyn_matrix<float, 2> A(4096, 4096);
    etl::dyn_matrix<float, 2> B(4096, 4096);
    etl::dyn_matrix<float, 2> C(4096, 4096);

    A = 1e-4 >> etl::sequence_generator<float>(1.0);
    B = 1e-4 >> etl::sequence_generator<float>(1.0);
    C = 1e-4 >> etl::sequence_generator<float>(1.0);

    etl::reset_counters();

    std::cout << "Simple" << std::endl;

    for (size_t i = 0; i < 10; ++i) {
        C = A * B;
        fake += etl::mean(C);
    }

    etl::dump_counters();

    std::cout << "   Result: " << fake << std::endl;
    std::cout << "Should be: 2.8826e+10" << std::endl;
}

void basic(){
    etl::dyn_matrix<float, 2> A(4096, 4096);
    etl::dyn_matrix<float, 2> B(4096, 4096);
    etl::dyn_matrix<float, 2> C(4096, 4096);
    etl::dyn_matrix<float, 2> D(4096, 4096);
    etl::dyn_matrix<float, 2> E(4096, 4096);

    A = 1e-4 >> etl::sequence_generator<float>(1.0);
    B = 1e-4 >> etl::sequence_generator<float>(1.0);
    C = 1e-4 >> etl::sequence_generator<float>(1.0);
    D = 1e-4 >> etl::sequence_generator<float>(1.0);
    E = 1e-4 >> etl::sequence_generator<float>(1.0);

    etl::reset_counters();

    std::cout << "Basic" << std::endl;

    for (size_t i = 0; i < 10; ++i) {
        IF_DEBUG std::cout << i << ":0 C = A * B * E" << std::endl;
        C = A * B * E;
        IF_DEBUG std::cout << i << ":1 D = A * trans(A)" << std::endl;
        D = A * trans(A);
        IF_DEBUG std::cout << i << ":2 D *= 1.1" << std::endl;
        D *= 1.1;
        IF_DEBUG std::cout << i << ":3 E = D" << std::endl;
        E = D;
        IF_DEBUG std::cout << i << ":4 D += C" << std::endl;
        D += C;
        IF_DEBUG std::cout << i << ":5 fake += etl::mean(D)" << std::endl;
        fake += etl::mean(D);
        IF_DEBUG std::cout << i << ":6 end" << std::endl;
    }

    etl::dump_counters();

    std::cout << "   Result: " << fake << std::endl;
    std::cout << "Should be: 3.36933e+23" << std::endl;
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

// Simulate forward propagation in a neural network (with some ops as DLL)
void ml(){
    etl::dyn_matrix<float, 4> I(32, 3, 28, 28);

    etl::dyn_matrix<float, 4> W1(16, 3, 3, 3);
    etl::dyn_matrix<float, 1> B1(16);
    etl::dyn_matrix<float, 4> O1(32, 16, 28, 28);

    etl::dyn_matrix<float, 4> P1(32, 16, 14, 14);

    etl::dyn_matrix<float, 4> W2(16, 16, 3, 3);
    etl::dyn_matrix<float, 1> B2(16);
    etl::dyn_matrix<float, 4> O2(32, 16, 14, 14);

    etl::dyn_matrix<float, 4> P2(32, 16, 7, 7);

    etl::dyn_matrix<float, 2> W3(16 * 7 * 7, 500);
    etl::dyn_matrix<float, 1> B3(500);
    etl::dyn_matrix<float, 2> O3(32, 500);

    etl::dyn_matrix<float, 2> W4(500, 10);
    etl::dyn_matrix<float, 1> B4(10);
    etl::dyn_matrix<float, 2> O4(32, 10);

    etl::reset_counters();

    std::cout << "ML" << std::endl;

    for (size_t i = 0; i < 10; ++i) {
        O1 = bias_add_4d(etl::ml::convolution_forward<1, 1, 1, 1>(I, W1), B1);
        P1 = etl::max_pool_3d<1, 2, 2>(O1);

        O2 = bias_add_4d(etl::ml::convolution_forward<1, 1, 1, 1>(P1, W2), B2);
        P2 = etl::max_pool_3d<1, 2, 2>(O2);

        O3 = bias_add_2d(etl::reshape<32, 16 * 7 * 7>(P2) * W3, B3);
        O4 = bias_add_2d(O3 * W4, B4);

        //TODO Add backward
    }

    etl::dump_counters();
}

int main(){
    auto start_time = timer_clock::now();

    simple();
    basic();
    sub();
    ml();

    auto end_time = timer_clock::now();
    auto duration = std::chrono::duration_cast<milliseconds>(end_time - start_time);

    std::cout << "duration: " << duration.count() << "ms" << std::endl;

    return (int) fake;
}
