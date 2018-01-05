//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#include "etl/etl.hpp"

template <typename A, typename M>
void im2col_direct_tr_multi(M& m, A&& sub, size_t k1, size_t k2) {
    const size_t N  = etl::dim<0>(sub);
    const size_t i1 = etl::dim<1>(sub);
    const size_t i2 = etl::dim<2>(sub);

    const auto height = i1 - k1 + 1;
    const auto width  = i2 - k2 + 1;

    const auto mm = m.memory_start();
    const auto ss = sub.memory_start();

    for (size_t w = 0; w < k1 * k2; ++w) {
        const size_t w_source = w % k2;
        const size_t h_source = (w / k2) % k1;
        const size_t c_source = w / (k1 * k2);

        for (size_t i = 0; i < N; ++i) {
            for (size_t h = 0; h < height; ++h) {
                const size_t block_source = ((c_source * i1 + h + h_source) * i2 + w_source) + (i) * (i1 * i2);
                const size_t block_target = (w * N + i) * (height * width) + h * width;

                etl::direct_copy_n(ss + block_source, mm + block_target, width);
            }
        }
    }
}

void test_more(){
    const size_t N = 3;
    const size_t i1 = 3;
    const size_t i2 = 3;

    //const size_t K = 2;
    const size_t k1 = 2;
    const size_t k2 = 2;

    const size_t c1 = i1 - k1 + 1;
    const size_t c2 = i1 - k1 + 1;

    using Z = float;

    etl::dyn_matrix<Z, 3> input(N, i1, i2);
    input = etl::sequence_generator(1);

    etl::dyn_matrix<Z, 2> input_col_large(k1 * k2, N * c1 * c2);
    im2col_direct_tr_multi(input_col_large, input, k1, k2);

    std::cout << "\nMore" << std::endl;
    std::cout << "Input\n";
    std::cout << etl::to_string(input) << std::endl;
    std::cout << "Input Col Large\n";
    std::cout << etl::to_string(input_col_large) << std::endl;
}

int main(){
    const size_t N = 2;
    const size_t i1 = 3;
    const size_t i2 = 3;

    const size_t K = 2;
    const size_t k1 = 2;
    const size_t k2 = 2;

    const size_t c1 = i1 - k1 + 1;
    const size_t c2 = i1 - k1 + 1;

    using Z = float;

    etl::dyn_matrix<Z, 3> input(N, i1, i2, std::initializer_list<Z>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0}));
    etl::dyn_matrix<Z, 3> kernels(K, k1, k2, std::initializer_list<Z>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));

    etl::dyn_matrix<Z, 2> input_col(k1 * k2, c1 * c2);
    etl::dyn_matrix<Z, 2> input_col_large(k1 * k2, N * c1 * c2);

    etl::dyn_matrix<Z, 3> CA(K, c1, c2);
    etl::dyn_matrix<Z, 3> CB(K, c1, c2);

    CA = etl::conv_2d_valid_multi_flipped(input(0), kernels);
    CB = etl::conv_2d_valid_multi_flipped(input(1), kernels);

    im2col_direct_tr(input_col, input(0), k1, k2);
    im2col_direct_tr_multi(input_col_large, input, k1, k2);


    std::cout << "\nInput Col Large\n";
    std::cout << etl::to_string(input_col_large) << std::endl;
    std::cout << "R Large\n";
    std::cout << etl::to_string(reshape(kernels, K, k1 * k2) * input_col_large) << std::endl;


    std::cout << "I1\n";
    std::cout << etl::to_string(input(0)) << std::endl;
    std::cout << "I2\n";
    std::cout << etl::to_string(input(1)) << std::endl;

    std::cout << "\nK1\n";
    std::cout << etl::to_string(kernels(0)) << std::endl;
    std::cout << "K1\n";
    std::cout << etl::to_string(kernels(1)) << std::endl;

    std::cout << "\nInput Col\n";
    std::cout << etl::to_string(input_col) << std::endl;
    std::cout << "Kernels\n";
    std::cout << etl::to_string(reshape(kernels, K, k1 * k2)) << std::endl;
    std::cout << "R\n";
    std::cout << etl::to_string(reshape(kernels, K, k1 * k2) * input_col) << std::endl;

    std::cout << "\nCA1\n";
    std::cout << etl::to_string(CA(0)) << std::endl;
    std::cout << "CA2\n";
    std::cout << etl::to_string(CA(1)) << std::endl;

    std::cout << "\nCB1\n";
    std::cout << etl::to_string(CB(0)) << std::endl;
    std::cout << "CB2\n";
    std::cout << etl::to_string(CB(1)) << std::endl;

    {
        test_more();
    }

    return 0;
}
