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
//#define ETL_COUNTERS_VERBOSE

// Use GPU pool to limit allocations
#define ETL_GPU_POOL

#define IF_DEBUG if(false)

// To test more expressions
#define ETL_STRICT_DIV

#include "etl/etl.hpp"

namespace {

typedef std::chrono::high_resolution_clock timer_clock;
typedef std::chrono::milliseconds milliseconds;

float fake = 0;

/*
 *
 * Current values are (alloc/release/gpu_to_cpu/cpu_to_gpu):
 *
 * No pool
 * Simple:   3 /   3 /   0 /   2 (Optimal!)
 * Basic:   15 /  15 /   0 /   3 (Optimal!)
 * Expr:   125 / 125 /   0 /   4 (Optimal!) (TODO:Update)
 * Direct:  43 /  43 /   0 /   3 (Optimal!)
 * ML:     270 / 270 /   0 /  10 (Optimal!)
 * Sub:    163 / 163 / 160 / 480
 *
 * GPU pool
 * Simple:  3 /  0 /   0 /   2 (Optimal!)
 * Basic:   6 /  0 /   0 /   3 (Optimal!)
 * Expr:    8 /  0 /   0 /   4 (Optimal!) (TODO:Update)
 * Direct:  4 /  0 /   0 /   3 (Optimal!)
 * ML:     48 / 16 /   0 /  10 (Optimal!)
 * Sub:     4 /  0 / 160 / 480
 */

void simple() {
    std::cout << "Simple" << std::endl;

#ifdef ETL_CUDA
    etl::gpu_memory_allocator::clear();
#endif

    etl::reset_counters();

    {
        etl::dyn_matrix<float, 2> A(4096, 4096);
        etl::dyn_matrix<float, 2> B(4096, 4096);
        etl::dyn_matrix<float, 2> C(4096, 4096);

        A = 1e-4 >> etl::sequence_generator<float>(1.0);
        B = 1e-4 >> etl::sequence_generator<float>(1.0);
        C = 1e-4 >> etl::sequence_generator<float>(1.0);

        for (size_t i = 0; i < 10; ++i) {
            B = 1.0f;
            C = etl::reshape(A, 4096, 4096) * etl::reshape<4096, 4096>(B);
            fake += etl::mean(C);
        }

        std::cout << "   Result: " << fake << std::endl;
        std::cout << "Should be: 2.8826e+10" << std::endl;
    }

    etl::dump_counters();
}

void basic() {
    std::cout << "Basic" << std::endl;

#ifdef ETL_CUDA
    etl::gpu_memory_allocator::clear();
#endif

    etl::reset_counters();

    {
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

        for (size_t i = 0; i < 10; ++i) {
            IF_DEBUG std::cout << i << ":0 C = A * B * E" << std::endl;
            C = A * B * E;
            IF_DEBUG std::cout << i << ":1 D = A * trans(A)" << std::endl;
            D = A * trans(A);
            IF_DEBUG std::cout << i << ":2 D *= 1.1" << std::endl;
            D *= 1.1f;
            IF_DEBUG std::cout << i << ":3 E = D" << std::endl;
            E = D;
            IF_DEBUG std::cout << i << ":4 D += C" << std::endl;
            D += C;
            IF_DEBUG std::cout << i << ":5 fake += etl::mean(D)" << std::endl;
            fake += etl::mean(D);
            IF_DEBUG std::cout << i << ":6 end" << std::endl;
        }

        std::cout << "   Result: " << fake << std::endl;
        std::cout << "Should be: 3.36933e+23" << std::endl;
    }

    etl::dump_counters();
}

void expr() {
    std::cout << "Expr" << std::endl;

#ifdef ETL_CUDA
    etl::gpu_memory_allocator::clear();
#endif

    etl::reset_counters();

    {
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

        for (size_t i = 0; i < 10; ++i) {
            A = 1.1f * (A / B) + (C >> D) / 1.2f;
            E = 1.2f / A - D * C + 2.0f;
            D = 1.1f + E;
            B = (1.4f - D) - 2.3f;
            D = (A + B) + (C + E);
            B = etl::sqrt(D);
        }
    }

    etl::dump_counters();
}

void direct() {
    std::cout << "Direct" << std::endl;

#ifdef ETL_CUDA
    etl::gpu_memory_allocator::clear();
#endif

    etl::reset_counters();

    {
        etl::dyn_matrix<float, 2> A(2048, 2048);
        etl::dyn_matrix<float, 2> B(2048, 2048);
        etl::dyn_matrix<float, 2> C(2048, 2048);

        A = 1e-4 >> etl::sequence_generator<float>(1.0);
        B = 1e-4 >> etl::sequence_generator<float>(1.0);
        C = 1e-4 >> etl::sequence_generator<float>(1.0);

        A.ensure_gpu_up_to_date();
        B.ensure_gpu_up_to_date();
        C.ensure_gpu_up_to_date();

        for (size_t i = 0; i < 10; ++i) {
            // Test direct compound
            A += B;
            C -= A;
            B >>= C;
            B /= A;

            // Test expression compound
            A += A * B;
            C -= A * B;
            B >>= C * B;
            B /= A * C;

            // Test scalar compound
            A += 1.1f;
            B -= 1.1f;
            C *= 1.1f;
            A /= 1.1f;
        }
    }

    etl::dump_counters();
}

void sub() {
    std::cout << "Sub" << std::endl;

#ifdef ETL_CUDA
    etl::gpu_memory_allocator::clear();
#endif

    etl::reset_counters();
    {
        etl::dyn_matrix<float, 3> A(16, 2048, 2048);
        etl::dyn_matrix<float, 3> B(16, 2048, 2048);
        etl::dyn_matrix<float, 3> C(16, 2048, 2048);
        etl::dyn_matrix<float, 3> D(16, 2048, 2048);

        A = etl::normal_generator<float>(1.0, 0.0);
        B = etl::normal_generator<float>(1.0, 0.0);
        C = etl::normal_generator<float>(1.0, 0.0);
        D = etl::normal_generator<float>(1.0, 0.0);

        for (size_t i = 0; i < 10; ++i) {
            for (size_t k = 0; k < 16; ++k) {
                C(k) = A(k) * B(k) * B(k);
                D(k) += C(k);
                D(k) *= 1.1f;
                fake += etl::mean(D(k));
            }
        }
    }

    etl::dump_counters();
}

void sub_ro() {
    std::cout << "Sub Read-Only" << std::endl;

#ifdef ETL_CUDA
    etl::gpu_memory_allocator::clear();
#endif

    etl::reset_counters();
    {
        etl::dyn_matrix<float, 3> A(8, 1024, 1024);
        etl::dyn_matrix<float, 3> B(8, 1024, 1024);
        etl::dyn_matrix<float, 3> C(8, 1024, 1024);
        etl::dyn_matrix<float, 3> D(8, 1024, 1024);
        etl::dyn_matrix<float, 2> E(1024, 1024);

        A = etl::normal_generator<float>(1.0, 0.0);
        B = etl::normal_generator<float>(1.0, 0.0);
        C = etl::normal_generator<float>(1.0, 0.0);
        D = etl::normal_generator<float>(1.0, 0.0);
        E = etl::normal_generator<float>(1.0, 0.0);

        for (size_t i = 0; i < 10; ++i) {
            for (size_t k = 0; k < 8; ++k) {
                E -= A(k) * B(k) * B(k);
                E += C(k);
                E *= 1.1f;
                fake += etl::mean(E);
            }
        }
    }

    etl::dump_counters();
}

// Simulate forward propagation in a neural network (with same ops as DLL)
void ml() {
    std::cout << "ML" << std::endl;

#ifdef ETL_CUDA
    etl::gpu_memory_allocator::clear();
#endif

    etl::reset_counters();

    {
        etl::dyn_matrix<float, 4> I(32, 3, 28, 28);
        etl::dyn_matrix<float, 2> L(32, 10);

        etl::dyn_matrix<float, 4> C1_W(16, 3, 3, 3);
        etl::dyn_matrix<float, 1> C1_B(16);
        etl::dyn_matrix<float, 4> C1_W_G(16, 3, 3, 3);
        etl::dyn_matrix<float, 1> C1_B_G(16);
        etl::dyn_matrix<float, 4> C1_O(32, 16, 28, 28);
        etl::dyn_matrix<float, 4> C1_E(32, 16, 28, 28);

        etl::dyn_matrix<float, 4> P1_O(32, 16, 14, 14);
        etl::dyn_matrix<float, 4> P1_E(32, 16, 14, 14);

        etl::dyn_matrix<float, 4> C2_W(16, 16, 3, 3);
        etl::dyn_matrix<float, 1> C2_B(16);
        etl::dyn_matrix<float, 4> C2_W_G(16, 16, 3, 3);
        etl::dyn_matrix<float, 1> C2_B_G(16);
        etl::dyn_matrix<float, 4> C2_O(32, 16, 14, 14);
        etl::dyn_matrix<float, 4> C2_E(32, 16, 14, 14);

        etl::dyn_matrix<float, 4> P2_O(32, 16, 7, 7);
        etl::dyn_matrix<float, 4> P2_E(32, 16, 7, 7);

        etl::dyn_matrix<float, 2> FC1_W(16 * 7 * 7, 500);
        etl::dyn_matrix<float, 1> FC1_B(500);
        etl::dyn_matrix<float, 2> FC1_W_G(16 * 7 * 7, 500);
        etl::dyn_matrix<float, 1> FC1_B_G(500);
        etl::dyn_matrix<float, 2> FC1_O(32, 500);
        etl::dyn_matrix<float, 2> FC1_E(32, 500);

        etl::dyn_matrix<float, 2> FC2_W(500, 10);
        etl::dyn_matrix<float, 1> FC2_B(10);
        etl::dyn_matrix<float, 2> FC2_W_G(500, 10);
        etl::dyn_matrix<float, 1> FC2_B_G(10);
        etl::dyn_matrix<float, 2> FC2_O(32, 10);
        etl::dyn_matrix<float, 2> FC2_E(32, 10);

        float eps = 0.1;

        for (size_t i = 0; i < 10; ++i) {
            // Forward Propagation
            C1_O = relu(bias_add_4d(etl::ml::convolution_forward<1, 1, 1, 1>(I, C1_W), C1_B));
            P1_O = etl::max_pool_2d<2, 2>(C1_O);

            C2_O = relu(bias_add_4d(etl::ml::convolution_forward<1, 1, 1, 1>(P1_O, C2_W), C2_B));
            P2_O = etl::max_pool_2d<2, 2>(C2_O);

            FC1_O = sigmoid(bias_add_2d(etl::reshape<32, 16 * 7 * 7>(P2_O) * FC1_W, FC1_B));
            FC2_O = sigmoid(bias_add_2d(FC1_O * FC2_W, FC2_B));

            // Backward propagation of the errors

            FC2_E = L - FC2_O;                                            // Errors of last layer  (!GPU)
            FC1_E = FC2_E * trans(FC2_W);                                 // Backpropagate FC2 -> FC1
            FC1_E = etl::ml::sigmoid_backward(FC1_O, FC1_E);              // Adapt errors of FC1
            etl::reshape<32, 16 * 7 * 7>(P2_E) = FC1_E * trans(FC1_W);    // FC1 -> MP2
            C2_E = etl::max_pool_upsample_2d<2, 2>(C2_O, P2_O, P2_E);     // MP2 -> C2
            C2_E = etl::ml::relu_backward(C2_O, C2_E);                    // Adapt errors of C2
            P1_E = etl::ml::convolution_backward<1, 1, 1, 1>(C2_E, C2_W); // C2 -> MP1
            C1_E = etl::max_pool_upsample_2d<2, 2>(C1_O, P1_O, P1_E);     // MP1 -> C1
            C1_E = etl::ml::relu_backward(C1_O, C1_E);                    // Adapt errors of C1

            // Compute the gradients

            C1_W_G = etl::ml::convolution_backward_filter<1, 1, 1, 1>(I, C1_E);
            C1_B_G = bias_batch_sum_4d(C1_E);

            C2_W_G = etl::ml::convolution_backward_filter<1, 1, 1, 1>(P1_O, C2_E);
            C2_B_G = bias_batch_sum_4d(C2_E);

            FC1_W_G = batch_outer(etl::reshape<32, 16 * 7 * 7>(P2_O), FC1_E);
            FC1_B_G = bias_batch_sum_2d(FC1_E);

            FC2_W_G = batch_outer(FC1_O, FC2_E);
            FC2_B_G = bias_batch_sum_2d(FC2_E);

            // Apply the gradients

            FC2_W += (eps / 32) * FC2_W_G;
            FC2_B += (eps / 32) * FC2_B_G;

            FC1_W += (eps / 32) * FC1_W_G;
            FC1_B += (eps / 32) * FC1_B_G;

            C2_W += (eps / 32) * C2_W_G;
            C2_B += (eps / 32) * C2_B_G;

            C1_W += (eps / 32) * C1_W_G;
            C1_B += (eps / 32) * C1_B_G;
        }
    }

    etl::dump_counters();
}

void random_test() {
    std::cout << "Random" << std::endl;

#ifdef ETL_CUDA
    etl::gpu_memory_allocator::clear();
#endif

    etl::reset_counters();

    {
        using T = float;

        etl::dyn_matrix<T, 4> I(32, 3, 28, 28);
        etl::dyn_matrix<T, 2> L(32, 10);

        I = etl::normal_generator<T>();
        L = etl::uniform_generator<T>(10, 20);
    }

    etl::dump_counters();
}

} // end of anonymous namespace

int main() {
    auto start_time = timer_clock::now();

    simple();
    basic();
    expr();
    direct();
    ml();
    sub();
    sub_ro();
    random_test();

    auto end_time = timer_clock::now();
    auto duration = std::chrono::duration_cast<milliseconds>(end_time - start_time);

    std::cout << "duration: " << duration.count() << "ms" << std::endl;

    etl::exit();

    return (int)fake;
}
