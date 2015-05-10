//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*
 * AVX implementation of 1D and 2D convolutions
 *
 * Ideas:
 *  * the tmp_res vectors could be avoided by using hadd instructions
 *  * 1D convolution with no memory allocation could probably be worked out (needs to be benchmarked)
 *  * Probably some other AVX2 instructions that could improve performances
 */

#ifndef ETL_IMPL_AVX_CONVOLUTION_HPP
#define ETL_IMPL_AVX_CONVOLUTION_HPP

#ifdef ETL_VECTORIZE_IMPL

#ifdef __AVX__

#include <immintrin.h>

#include "../../allocator.hpp"
#include "../common/conv.hpp"

namespace etl {

namespace impl {

namespace avx {

inline __m256d mm256_reverse_pd(__m256d m1){
#ifdef __AVX2__
    return _mm256_permute4x64_pd(m1, 0b00011011);
#else
    __m256d tmp;
    tmp = _mm256_permute2f128_pd(m1, m1, 1);
    return _mm256_permute_pd(tmp, 5);
#endif
}

inline __m256 mm256_reverse_ps(__m256 m1){
    __m256 tmp;
    tmp = _mm256_permute2f128_ps(m1, m1, 33);
    return _mm256_permute_ps(tmp, 27);
}

inline void dconv1_valid_micro_kernel(const double* in, const std::size_t n, const double* kernel, std::size_t m, double* out){
    auto* kernel_reverse = aligned_allocate<__m256d>(m);

    //Reverse the kernel

    for(std::size_t i=0; i< m; i++){
        kernel_reverse[i] = _mm256_broadcast_sd(kernel + m - i - 1);
    }

    __m256d tmp1;
    __m256d tmp2;
    __m256d res;

    auto c = n - m + 1;

    //Compute the convolution, 2 doubles at a time

    for(std::size_t i=0; i + 3 < c; i+=4){
        res = _mm256_setzero_pd();

        for(std::size_t k=0; k< m; k++){
            tmp1 = _mm256_loadu_pd(in + i + k);
            tmp2 = _mm256_mul_pd(kernel_reverse[k], tmp1);
            res = _mm256_add_pd(res, tmp2);
        }

        _mm256_storeu_pd(out+i, res);
    }

    //If the number of operations is not even, the last case must be
    //computed separatly

    if(c % 4 != 0){
        auto rem = c % 4;
        for(std::size_t i = c - rem; i < c; ++i){
            out[i] = 0.0;
            for(std::size_t k=0; k<m; k++){
                out[i] += in[i+k] * kernel[m - k - 1];
            }
        }
    }

    aligned_release(kernel_reverse);
}

inline void sconv1_valid_micro_kernel(const float* in, const std::size_t n, const float* kernel, std::size_t m, float* out){
    auto* kernel_reverse = aligned_allocate<__m256>(m);

    //Reverse the kernel

    for(std::size_t i=0; i< m; i++){
        kernel_reverse[i] = _mm256_broadcast_ss(kernel + m - i - 1);
    }

    __m256 tmp1;
    __m256 tmp2;
    __m256 res;

    auto c = n - m + 1;

    //Compute the convolution 8 floats at a time

    for(std::size_t i = 0; i + 7 < c; i += 8){
        res = _mm256_setzero_ps();

        for(std::size_t k = 0; k<m; k++){
            tmp1 = _mm256_loadu_ps(in + i + k);
            tmp2 = _mm256_mul_ps(kernel_reverse[k], tmp1);
            res = _mm256_add_ps(res, tmp2);
        }

        _mm256_storeu_ps(out+i, res);
    }

    //Complete the last outputs which are not vectorized

    if(c % 8 != 0){
        auto rem = c % 8;
        for(std::size_t i = c - rem; i < c; ++i){
            out[i] = 0.0;
            for(std::size_t k = 0; k< m; k++){
                out[i] += in[i+k] * kernel[m - k - 1];
            }
        }
    }

    aligned_release(kernel_reverse);
}

template<typename I, typename K, typename C>
void dconv1_full(const I& input, const K& kernel, C&& conv){
    std::size_t left = size(kernel) - 1;

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_full_kernel(in, size(input), k, size(kernel), out);
    etl::impl::common::right_full_kernel(in, size(input), k, size(kernel), out);

    //Central part is a 'valid' convolution
    dconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left);
}

template<typename I, typename K, typename C>
void dconv1_same(const I& input, const K& kernel, C&& conv){
    std::size_t left = (size(kernel) - 1) / 2;

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_same_kernel(in, size(input), k, size(kernel), out);
    etl::impl::common::right_same_kernel(in, size(input), k, size(kernel), out);

    //Central part is a 'valid' convolution
    dconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left);
}

template<typename I, typename K, typename C>
void dconv1_valid(const I& input, const K& kernel, C&& conv){
    dconv1_valid_micro_kernel(input.memory_start(), size(input), kernel.memory_start(), size(kernel), conv.memory_start());
}

template<typename I, typename K, typename C>
void sconv1_full(const I& input, const K& kernel, C&& conv){
    std::size_t left = size(kernel) - 1;

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_full_kernel(in, size(input), k, size(kernel), out);
    etl::impl::common::right_full_kernel(in, size(input), k, size(kernel), out);

    //Central part is a 'valid' convolution
    sconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left);
}

template<typename I, typename K, typename C>
void sconv1_same(const I& input, const K& kernel, C&& conv){
    std::size_t left = (size(kernel) - 1) / 2;

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_same_kernel(in, size(input), k, size(kernel), out);
    etl::impl::common::right_same_kernel(in, size(input), k, size(kernel), out);

    //Central part is a 'valid' convolution
    sconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left);
}

template<typename I, typename K, typename C>
void sconv1_valid(const I& input, const K& kernel, C&& conv){
    sconv1_valid_micro_kernel(input.memory_start(), size(input), kernel.memory_start(), size(kernel), conv.memory_start());
}

inline void dconv2_valid_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out){
    auto c1 = n1 - m1 + 1;
    auto c2 = n2 - m2 + 1;

    __m256d tmp1;
    __m256d tmp2;
    __m256d tmp3;
    __m256d tmp4;
    __m256d res;

    double tmp_res[4] __attribute__ ((aligned (32)));

    for(std::size_t i = 0 ; i < c1 ; ++i){
        for(std::size_t j = 0 ; j < c2 ; ++j){
            res = _mm256_setzero_pd();

            for(std::size_t k = i ; k < i + m1; ++k){
                for(std::size_t l = j; l + 3 < j + m2; l += 4){
                    tmp1 = _mm256_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm256_loadu_pd(kernel + ((i+m1-1-k) * m2 + (j+m2-1-(l+3))));
                    tmp3 = mm256_reverse_pd(tmp2);
                    tmp4 = _mm256_mul_pd(tmp3, tmp1);
                    res = _mm256_add_pd(res, tmp4);
                }
            }

            _mm256_store_pd(tmp_res, res);

            double temp = 0.0;

            if(m2 % 4 != 0){
                auto rem = m2 % 4;
                for(std::size_t k = i ; k < i + m1; ++k){
                    for(std::size_t l = j + m2 - rem; l < j + m2; ++l){
                        temp += in[k * n2 + l] * kernel[(i+m1-1-k) * m2 + (j+m2-1-l)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];
        }
    }
}

template<typename I, typename K, typename C>
void dconv2_valid(const I& input, const K& kernel, C&& conv){
    dconv2_valid_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void dconv2_same_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out){
    auto c1 = n1;
    auto c2 = n2;

    __m256d tmp1;
    __m256d tmp2;
    __m256d tmp3;
    __m256d tmp4;
    __m256d res;

    double tmp_res[4] __attribute__ ((aligned (32)));

    for(std::size_t i = 0 ; i < c1; ++i){
        std::size_t k_lo = std::max<int>(0, i - (m1-1)/2);
        std::size_t k_hi = std::min<int>(n1 - 1, i + m1/2) + 1;

        for(std::size_t j = 0 ; j < c2; ++j){
            std::size_t l_lo = std::max<int>(0, j - (m2-1)/2);
            std::size_t l_hi = std::min<int>(n2 - 1, j + m2/2) + 1;

            res = _mm256_setzero_pd();

            for(std::size_t k = k_lo ; k < k_hi ; ++k){
                for(std::size_t l = l_lo ; l + 3 < l_hi; l += 4){
                    tmp1 = _mm256_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm256_loadu_pd(kernel + (i-k+m1/2) * m2 +(j-(l+3)+m2/2));
                    tmp3 = mm256_reverse_pd(tmp2);
                    tmp4 = _mm256_mul_pd(tmp3, tmp1);
                    res = _mm256_add_pd(res, tmp4);
                }
            }

            _mm256_store_pd(tmp_res, res);

            double temp = 0.0;

            if((l_hi - l_lo) % 4 != 0){
                auto rem = (l_hi - l_lo) % 4;
                for(std::size_t k = k_lo ; k < k_hi ; ++k){
                    for(std::size_t l = l_hi - rem ; l < l_hi; ++l){
                        temp += in[k * n2 + l] * kernel[(i-k+m1/2) * m2 +(j-l+m2/2)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];
        }
    }
}

template<typename I, typename K, typename C>
void dconv2_same(const I& input, const K& kernel, C&& conv){
    dconv2_same_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void dconv2_full_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out){
    auto c1 = n1 + m1 - 1;
    auto c2 = n2 + m2 - 1;

    __m256d tmp1;
    __m256d tmp2;
    __m256d tmp3;
    __m256d tmp4;
    __m256d res;

    double tmp_res[4] __attribute__ ((aligned (32)));

    for(std::size_t i = 0 ; i < c1 ; ++i){
        auto k_lo = std::max<int>(0, i - m1 + 1);
        auto k_hi = std::min(n1 - 1, i) + 1;

        for(std::size_t j = 0 ; j < c2 ; ++j){
            auto l_lo = std::max<int>(0, j - m2 + 1);
            auto l_hi = std::min(n2 - 1 , j) + 1;

            res = _mm256_setzero_pd();

            for(std::size_t k = k_lo ; k < k_hi ; ++k){
                for(std::size_t l = l_lo; l + 3 < l_hi ; l += 4){
                    tmp1 = _mm256_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm256_loadu_pd(kernel + (i - k) * m2 + (j - (l+3)));
                    tmp3 = mm256_reverse_pd(tmp2);
                    tmp4 = _mm256_mul_pd(tmp3, tmp1);
                    res = _mm256_add_pd(res, tmp4);
                }
            }

            _mm256_store_pd(tmp_res, res);

            double temp = 0.0;

            if((l_hi - l_lo) % 4 != 0){
                auto rem = (l_hi - l_lo) % 4;
                for(std::size_t k = k_lo ; k < k_hi ; ++k){
                    for(std::size_t l = l_hi - rem; l < l_hi ; ++l){
                        temp += in[k * n2 + l] * kernel[(i - k) * m2 + (j - l)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];
        }
    }
}

template<typename I, typename K, typename C>
void dconv2_full(const I& input, const K& kernel, C&& conv){
    dconv2_full_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void sconv2_valid_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out){
    auto c1 = n1 - m1 + 1;
    auto c2 = n2 - m2 + 1;

    __m256 tmp1;
    __m256 tmp2;
    __m256 tmp3;
    __m256 tmp4;
    __m256 res;

    float tmp_res[8] __attribute__ ((aligned (32)));

    for(std::size_t i = 0 ; i < c1 ; ++i){
        for(std::size_t j = 0 ; j < c2 ; ++j){
            res = _mm256_setzero_ps();

            for(std::size_t k = i ; k < i + m1; ++k){
                for(std::size_t l = j ; l + 7 < j + m2; l += 8){
                    tmp1 = _mm256_loadu_ps(in + k * n2 + l);
                    tmp2 = _mm256_loadu_ps(kernel + (i+m1-1-k) * m2 + (j+m2-1-(l+7)));
                    tmp3 = mm256_reverse_ps(tmp2);
                    tmp4 = _mm256_mul_ps(tmp3, tmp1);
                    res = _mm256_add_ps(res, tmp4);
                }
            }

            _mm256_store_ps(tmp_res, res);

            float temp = 0.0;

            if(m2 % 8 != 0){
                auto rem = m2 % 8;
                for(std::size_t k = i ; k < i + m1; ++k){
                    for(std::size_t l = j + m2 - rem; l < j + m2; ++l){
                        temp += in[k * n2 + l] * kernel[(i+m1-1-k) * m2 + (j+m2-1-l)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] + tmp_res[5] + tmp_res[6] + tmp_res[7];
        }
    }
}

template<typename I, typename K, typename C>
void sconv2_valid(const I& input, const K& kernel, C&& conv){
    sconv2_valid_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void sconv2_same_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out){
    auto c1 = n1;
    auto c2 = n2;

    __m256 tmp1;
    __m256 tmp2;
    __m256 tmp3;
    __m256 tmp4;
    __m256 res;

    float tmp_res[8] __attribute__ ((aligned (32)));

    for(std::size_t i = 0 ; i < c1; ++i){
        auto k_lo = std::max<int>(0, i - (m1-1)/2);
        std::size_t k_hi = std::min<int>(n1 - 1, i + m1/2) + 1;

        for(std::size_t j = 0 ; j < c2; ++j){
            auto l_lo = std::max<int>(0, j - (m2-1)/2);
            std::size_t l_hi = std::min<int>(n2 - 1, j + m2/2) + 1;

            res = _mm256_setzero_ps();

            for(std::size_t k = k_lo ; k < k_hi ; ++k){
                for(std::size_t l = l_lo ; l + 7 < l_hi; l += 8){
                    tmp1 = _mm256_loadu_ps(in + k * n2 + l);
                    tmp2 = _mm256_loadu_ps(kernel + (i-k+m1/2) * m2 + (j-(l+7)+m2/2));
                    tmp3 = mm256_reverse_ps(tmp2);
                    tmp4 = _mm256_mul_ps(tmp3, tmp1);
                    res = _mm256_add_ps(res, tmp4);
                }
            }

            _mm256_store_ps(tmp_res, res);

            float temp = 0.0;

            if((l_hi - l_lo) % 8 != 0){
                auto rem = (l_hi - l_lo) % 8;
                for(std::size_t k = k_lo ; k < k_hi; ++k){
                    for(std::size_t l = l_hi - rem ; l < l_hi; ++l){
                        temp += in[k * n2 + l] * kernel[(i-k+m1/2) * m2 + (j-l+m2/2)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] + tmp_res[5] + tmp_res[6] + tmp_res[7];
        }
    }
}

template<typename I, typename K, typename C>
void sconv2_same(const I& input, const K& kernel, C&& conv){
    sconv2_same_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void sconv2_full_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out){
    auto c1 = n1 + m1 - 1;
    auto c2 = n2 + m2 - 1;

    __m256 tmp1;
    __m256 tmp2;
    __m256 tmp3;
    __m256 tmp4;
    __m256 res;

    float tmp_res[8] __attribute__ ((aligned (32)));

    for(std::size_t i = 0 ; i < c1 ; ++i){
        auto k_lo = std::max<int>(0, i - m1 + 1);
        auto k_hi = std::min(n1 - 1, i) + 1;

        for(std::size_t j = 0 ; j < c2 ; ++j){
            auto l_lo = std::max<int>(0, j - m2 + 1);
            auto l_hi = std::min(n2 - 1 , j) + 1;

            res = _mm256_setzero_ps();

            for(std::size_t k = k_lo ; k < k_hi; ++k){
                for(std::size_t l = l_lo ; l + 7 < l_hi; l += 8){
                    tmp1 = _mm256_loadu_ps(in + k * n2 + l);
                    tmp2 = _mm256_loadu_ps(kernel + (i - k) * m2 + (j - (l+7)));
                    tmp3 = mm256_reverse_ps(tmp2);
                    tmp4 = _mm256_mul_ps(tmp3, tmp1);
                    res = _mm256_add_ps(res, tmp4);
                }
            }

            _mm256_store_ps(tmp_res, res);

            double temp = 0.0;

            if((l_hi - l_lo) % 8 != 0){
                auto rem = (l_hi - l_lo) % 8;
                for(std::size_t k = k_lo ; k < k_hi; ++k){
                    for(std::size_t l = l_hi - rem ; l < l_hi; ++l){
                        temp += in[k * n2 + l] * kernel[(i - k) * m2 + (j - l)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] + tmp_res[5] + tmp_res[6] + tmp_res[7];
        }
    }
}

template<typename I, typename K, typename C>
void sconv2_full(const I& input, const K& kernel, C&& conv){
    sconv2_full_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

} //end of namespace avx
} //end of namespace impl
} //end of namespace etl

#endif //__SSE3__

#endif //ETL_VECTORIZE_IMPL

#endif
