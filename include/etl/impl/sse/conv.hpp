//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_IMPL_SSE_CONVOLUTION_HPP
#define ETL_IMPL_SSE_CONVOLUTION_HPP

#ifdef ETL_VECTORIZE

#ifdef __SSE3__

#include <immintrin.h>

namespace etl {

namespace impl {

namespace sse {

inline void dconv1_valid_micro_kernel(const double* in, const std::size_t n, const double* kernel, std::size_t m, double* out){
    __m128* kernel_reverse = new __m128[m];

    for(std::size_t i=0; i< m; i++){
        kernel_reverse[i] = _mm_load1_pd(kernel + m - i - 1);
    }

    __m128 tmp1;
    __m128 tmp2;
    __m128 res;

    for(std::size_t i=0; i<n - m; i+=2){
        res = _mm_setzero_pd();

        for(std::size_t k=0; k< m; k++){
            tmp1 = _mm_loadu_pd(in + i + k);
            tmp2 = _mm_mul_pd(kernel_reverse[k], tmp1);
            res = _mm_add_pd(res, tmp2);
        }

        _mm_storeu_pd(out+i, res);
    }

    auto i = n - m;
    out[i] = 0.0;
    for(std::size_t k=0; k<m; k++){
        out[i] += in[i+k] * kernel[m - k - 1];
    }

    delete[] kernel_reverse;
}

template<typename I, typename K, typename C>
void dconv1_same(const I& input, const K& kernel, C&& conv){
    std::size_t left = (size(kernel) - 1) / 2;
    std::size_t right = size(kernel) / 2;

    //Left invalid part
    for(std::size_t j = 0 ; j < left ; ++j){
        int l_lo = std::max<int>(0, j - (size(kernel) - 1) / 2);
        int l_hi = std::min<int>(size(input)- 1, j + size(kernel) / 2);

        double temp = 0.0;

        for(std::size_t l = l_lo ; l <= static_cast<std::size_t>(l_hi); ++l){
            temp += input(l) * kernel(j - l + size(kernel) / 2);
        }

        conv(j) = temp;
    }

    //Central part is a 'valid' convolution
    dconv1_valid_micro_kernel(input.memory_start(), size(input), kernel.memory_start(), size(kernel), conv.memory_start() + left);

    //Right invalid part
    for(std::size_t j = size(conv) - right ; j < size(conv); ++j){
        int l_lo = std::max<int>(0, j - (size(kernel) - 1) / 2);
        int l_hi = std::min<int>(size(input)- 1, j + size(kernel) / 2);

        double temp = 0.0;

        for(std::size_t l = l_lo ; l <= static_cast<std::size_t>(l_hi); ++l){
            temp += input(l) * kernel(j - l + size(kernel) / 2);
        }

        conv(j) = temp;
    }
}

template<typename I, typename K, typename C>
void dconv1_valid(const I& input, const K& kernel, C&& conv){
    dconv1_valid_micro_kernel(input.memory_start(), size(input), kernel.memory_start(), size(kernel), conv.memory_start());
}

template<typename I, typename K, typename C>
void sconv1_valid(const I& input, const K& kernel, C&& conv){
    __m128* kernel_reverse = new __m128[etl::size(kernel)];

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    for(std::size_t i=0; i< etl::size(kernel); i++){
        kernel_reverse[i] = _mm_load1_ps(k + etl::size(kernel) - i - 1);
    }

    __m128 tmp1;
    __m128 tmp2;
    __m128 res;

    for(std::size_t i=0; i<size(input)-etl::size(kernel); i+=2){
        res = _mm_setzero_ps();

        for(std::size_t k=0; k<size(kernel); k++){
            tmp1 = _mm_loadu_ps(in + i + k);
            tmp2 = _mm_mul_ps(kernel_reverse[k], tmp1);
            res = _mm_add_ps(res, tmp2);
        }

        _mm_storeu_ps(out+i, res);
    }

    auto i = size(input) - size(kernel);
    out[i] = 0.0;
    for(std::size_t k=0; k< size(kernel); k++){
        out[i] += input[i+k] * kernel[size(kernel) - k - 1];
    }

    delete[] kernel_reverse;
}

} //end of namespace std
} //end of namespace impl
} //end of namespace etl

#endif //__SSE3__

#endif //ETL_VECTORIZE

#endif
