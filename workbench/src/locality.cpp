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

template<typename I, typename K, typename C>
void locality(const I& input, const K& kernel, C& conv){
    conv = 0;

    const auto n1 = etl::dim<0>(input);
    const auto n2 = etl::dim<1>(input);

    const auto k1 = etl::dim<0>(kernel);
    const auto k2 = etl::dim<1>(kernel);

    const auto c1 = etl::dim<0>(conv);
    const auto c2 = etl::dim<1>(conv);

    const auto R = std::min(k1, c1); // Max number of kernels per line of input

    // Primary steps
    for(size_t i = 0; i < k1 - 1; ++i){
        const auto M = std::min(i + 1, R);

        std::cout << "Primary:" << i << ":" << M << std::endl;

        for(size_t m = 0; m < M; ++m){
            for(size_t j = 0; j < c2;  ++j){
                const auto c_i = m;
                const auto c_j = j;

                for(size_t k = 0; k < k2; ++k){
                    const auto k_i = i - m;
                    const auto k_j = k;

                    const auto i_i = i;
                    const auto i_j = j + k;

                    std::cout << "c(" << c_i << "," << c_j << ") += i(" << i_i << "," << i_j << ") * k("  << k_i << "," << k_j << ")" << std::endl;
                    conv(c_i, c_j) += input(i_i, i_j) * kernel(k_i, k_j);
                }
            }
        }
    }

    // Main steps
    for(size_t i = k1 - 1; i < c1; ++i){
        const auto M = R;

        std::cout << "Main:" << i << ":" << M << std::endl;

        for(size_t m = 0; m < M; ++m){
            for(size_t j = 0; j < c2;  ++j){
                const auto c_i = i - m;
                const auto c_j = j;

                for(size_t k = 0; k < k2; ++k){
                    const auto k_i = m;
                    const auto k_j = k;

                    const auto i_i = i;
                    const auto i_j = j + k;

                    std::cout << "c(" << c_i << "," << c_j << ") += i(" << i_i << "," << i_j << ") * k("  << k_i << "," << k_j << ")" << std::endl;
                    conv(c_i, c_j) += input(i_i, i_j) * kernel(k_i, k_j);
                }
            }
        }
    }

    // Secondary steps
    for(size_t i = c1; i < n1; ++i){
        auto M = std::min(n1 - i, R);

        std::cout << "Secondary:" << i << ":" << M << std::endl;

        for(size_t m = 0; m < M; ++m){
            for(size_t j = 0; j < c2;  ++j){
                const auto c_i = m + i - k1 + 1;
                const auto c_j = j;

                for(size_t k = 0; k < k2; ++k){
                    const auto k_i = M - m - c1 + i;
                    const auto k_j = k;

                    const auto i_i = i;
                    const auto i_j = j + k;

                    std::cout << "c(" << c_i << "," << c_j << ") += i(" << i_i << "," << i_j << ") * k("  << k_i << "," << k_j << ")" << std::endl;
                    conv(c_i, c_j) += input(i_i, i_j) * kernel(k_i, k_j);
                }
            }
        }
    }

    cpp_unused(n2);
    cpp_unused(k2);
    cpp_unused(c2);
}

int main(){
    const size_t i1 = 28;
    const size_t i2 = 28;

    const size_t k1 = 7;
    const size_t k2 = 5;

    const size_t c1 = i1 - k1 + 1;
    const size_t c2 = i2 - k2 + 1;

    std::cout << "Input: [" << i1 << "x" << i2 << "]" << std::endl;
    std::cout << "Kernel: [" << k1 << "x" << k2 << "]" << std::endl;
    std::cout << "Output: [" << c1 << "x" << c2 << "]" << std::endl;

    using Z = float;

    etl::dyn_matrix<Z, 2> input(i1, i2);
    etl::dyn_matrix<Z, 2> kernel(k1, k2);

    input = etl::sequence_generator(1.0);
    kernel = etl::sequence_generator(1.0);

    etl::dyn_matrix<Z, 2> CA(c1, c2);
    etl::dyn_matrix<Z, 2> CB(c1, c2);

    CA = etl::conv_2d_valid_flipped(input, kernel);
    locality(input, kernel, CB);

    if(i1 * i2 < 100){
        std::cout << std::endl;
        std::cout << etl::to_string(CA) << std::endl << std::endl;
        std::cout << etl::to_string(CB) << std::endl << std::endl;
    }

    std::cout << etl::sum(CA - CB) << std::endl;
    std::cout << etl::sum(etl::abs(CA - CB)) << std::endl;

    return 0;
}
