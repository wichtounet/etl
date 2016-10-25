//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

namespace vec {

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    using T = value_t<I>;

    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);

    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    const auto n1 = etl::dim<0>(input);

    const auto k1 = etl::dim<0>(kernel);
    const auto k2 = etl::dim<1>(kernel);

    const auto c1 = etl::dim<0>(conv);
    const auto c2 = etl::dim<1>(conv);

    const auto R = std::min(k1, c1); // Max number of kernels per line of input

    conv = 0;

    // Primary steps
    for(size_t i = 0; i < k1 - 1; ++i){
        const auto M = std::min(i + 1, R);

        for(size_t m = 0; m < M; ++m){
            const auto k_i = i - m;

            for(size_t j = 0; j < c2;  ++j){
                T value = 0;

                for(size_t k = 0; k < k2; ++k){
                    value += input(i, j + k) * kernel(k_i, k);
                }

                conv(m,j) += value;
            }
        }
    }

    // Main steps
    for(size_t i = k1 - 1; i < c1; ++i){
        const auto M = R;

        for(size_t m = 0; m < M; ++m){
            const auto c_i = n1 - 1 - m - i;

            for(size_t j = 0; j < c2;  ++j){
                T value = 0;

                for(size_t k = 0; k < k2; ++k){
                    value += input(i, j + k) * kernel(m, k);
                }

                conv(c_i, j) += value;
            }
        }
    }

    // Secondary steps
    for(size_t i = c1; i < n1; ++i){
        auto M = std::min(n1 - i, R);

        for(size_t m = 0; m < M; ++m){
            const auto c_i = m + i - k1 + 1;
            const auto k_i = M - m - c1 + i;

            for(size_t j = 0; j < c2;  ++j){
                T value = 0;

                for(size_t k = 0; k < k2; ++k){
                    value += input(i, j + k) * kernel(k_i, k);
                }

                conv(c_i, j) += value;
            }
        }
    }
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
