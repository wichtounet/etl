//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <chrono>
#include <random>

#include "etl/etl.hpp"

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

    etl::dyn_matrix<Z, 3> input(N, i1, i2, std::initializer_list<Z>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}));
    etl::dyn_matrix<Z, 3> kernels(K, k1, k2, std::initializer_list<Z>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}));

    etl::dyn_matrix<Z, 2> input_col(k1 * k2, c1 * c2);

    etl::dyn_matrix<Z, 3> CA(K, c1, c2);
    etl::dyn_matrix<Z, 3> CB(K, c1, c2);

    CA = etl::conv_2d_valid_multi_flipped(input(0), kernels);
    CB = etl::conv_2d_valid_multi_flipped(input(1), kernels);

    im2col_direct_tr(input_col, input(0), k1, k2);

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

    return 0;
}
