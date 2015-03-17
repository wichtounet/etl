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

template<typename I, typename K, typename C>
void sconv1_valid(const I& input, const K& kernel, C&& conv){
    __m128* kernel_reverse = new __m128[etl::size(kernel)];

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    for(std::size_t i=0; i< etl::size(kernel); i++){
        kernel_reverse[i] = _mm_load1_pd(k + etl::size(kernel) - i - 1);
    }

    __m128 tmp1;
    __m128 tmp2;
    __m128 res;

    for(std::size_t i=0; i<size(input)-etl::size(kernel); i+=2){
        res = _mm_setzero_pd();

        for(std::size_t k=0; k< size(kernel); k++){
            tmp1 = _mm_loadu_pd(in + i + k);
            tmp2 = _mm_mul_pd(kernel_reverse[k], tmp1);
            res = _mm_add_pd(res, tmp2);
        }

        _mm_storeu_pd(out+i, res);
    }

    auto i = size(input) - size(kernel);
    conv[i] = 0.0;
    for(std::size_t k=0; k<etl::size(kernel); k++){
        conv[i] += input[i+k] * kernel[size(kernel) - k - 1];
    }

    delete[] kernel_reverse;
}

template<typename I, typename K, typename C>
void dconv1_valid(const I& input, const K& kernel, C&& conv){
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
    conv[i] = 0.0;
    for(std::size_t k=0; k< size(kernel); k++){
        conv[i] += input[i+k] * kernel[size(kernel) - k - 1];
    }

    delete[] kernel_reverse;
}

} //end of namespace std
} //end of namespace impl
} //end of namespace etl

#endif //__SSE3__

#endif //ETL_VECTORIZE

#endif
