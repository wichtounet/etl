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

    //Reverse the kernel

    for(std::size_t i=0; i< m; i++){
        kernel_reverse[i] = _mm_load1_pd(kernel + m - i - 1);
    }

    __m128 tmp1;
    __m128 tmp2;
    __m128 res;

    //Compute the convolution, 2 doubles at a time

    for(std::size_t i=0; i<n - m; i+=2){
        res = _mm_setzero_pd();

        for(std::size_t k=0; k< m; k++){
            tmp1 = _mm_loadu_pd(in + i + k);
            tmp2 = _mm_mul_pd(kernel_reverse[k], tmp1);
            res = _mm_add_pd(res, tmp2);
        }

        _mm_storeu_pd(out+i, res);
    }

    //If the number of operations is not even, the last case must be
    //computed separatly

    if((n - m + 1) % 2 != 0){
        auto i = n - m;
        out[i] = 0.0;
        for(std::size_t k=0; k<m; k++){
            out[i] += in[i+k] * kernel[m - k - 1];
        }
    } 

    delete[] kernel_reverse;
}

template<typename I, typename K, typename C>
void dconv1_same(const I& input, const K& kernel, C&& conv){
    std::size_t left = (size(kernel) - 1) / 2;
    std::size_t right = size(kernel) / 2;

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    //Left invalid part
    for(std::size_t j = 0 ; j < left ; ++j){
        double temp = 0.0;

        for(std::size_t l = 0 ; l <= j + right; ++l){
            temp += in[l] * k[j - l + right];
        }

        out[j] = temp;
    }

    //Central part is a 'valid' convolution
    dconv1_valid_micro_kernel(in, size(input), kernel.memory_start(), size(kernel), out + left);

    //Right invalid part
    for(std::size_t j = size(conv) - right ; j < size(conv); ++j){
        double temp = 0.0;

        std::size_t hi = std::min<int>(size(input) - 1, j + right);
        for(std::size_t l = j - left ; l <= hi; ++l){
            temp += in[l] * k[j - l + size(kernel) / 2];
        }

        out[j] = temp;
    }
}

template<typename I, typename K, typename C>
void dconv1_valid(const I& input, const K& kernel, C&& conv){
    dconv1_valid_micro_kernel(input.memory_start(), size(input), kernel.memory_start(), size(kernel), conv.memory_start());
}

inline void sconv1_valid_micro_kernel(const float* in, const std::size_t n, const float* kernel, std::size_t m, float* out){
    __m128* kernel_reverse = new __m128[m];

    //Reverse the kernel

    for(std::size_t i=0; i< m; i++){
        kernel_reverse[i] = _mm_load1_ps(kernel + m - i - 1);
    }

    __m128 tmp1;
    __m128 tmp2;
    __m128 res;

    //Compute the convolution 4 floats at a time

    for(std::size_t i=0; i< 4 * ((n - m) / 4); i += 4){
        res = _mm_setzero_ps();

        for(std::size_t k=0; k<m; k++){
            tmp1 = _mm_loadu_ps(in + i + k);
            tmp2 = _mm_mul_ps(kernel_reverse[k], tmp1);
            res = _mm_add_ps(res, tmp2);
        }

        _mm_storeu_ps(out+i, res);
    }

    //Complete the last outputs which are not vectorized

    for(std::size_t i = (n - m + 1) - (n - m + 1) % 4; i < n - m + 1; ++i){
        out[i] = 0.0;
        for(std::size_t k=0; k< m; k++){
            out[i] += in[i+k] * kernel[m - k - 1];
        }
    }

    delete[] kernel_reverse;
}

template<typename I, typename K, typename C>
void sconv1_valid(const I& input, const K& kernel, C&& conv){
    sconv1_valid_micro_kernel(input.memory_start(), size(input), kernel.memory_start(), size(kernel), conv.memory_start());
}

} //end of namespace std
} //end of namespace impl
} //end of namespace etl

#endif //__SSE3__

#endif //ETL_VECTORIZE

#endif
