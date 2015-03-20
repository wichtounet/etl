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
    __m128d* kernel_reverse = new __m128d[m];

    //Reverse the kernel

    for(std::size_t i=0; i< m; i++){
        kernel_reverse[i] = _mm_load1_pd(kernel + m - i - 1);
    }

    __m128d tmp1;
    __m128d tmp2;
    __m128d res;

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

template<typename T>
void left_same_kernel(const T* in, const std::size_t, const T* kernel, std::size_t m, T* out){
    std::size_t left = (m - 1) / 2;
    std::size_t right = m / 2;

    //Left invalid part
    for(std::size_t j = 0 ; j < left ; ++j){
        T temp = 0.0;

        for(std::size_t l = 0 ; l <= j + right; ++l){
            temp += in[l] * kernel[j - l + right];
        }

        out[j] = temp;
    }
}

template<typename T>
void right_same_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out){
    std::size_t left = (m - 1) / 2;
    std::size_t right = m / 2;

    //Right invalid part
    for(std::size_t j = n - right ; j < n; ++j){
        T temp = 0.0;

        std::size_t hi = std::min<int>(n - 1, j + right);
        for(std::size_t l = j - left ; l <= hi; ++l){
            temp += in[l] * kernel[j - l + m / 2];
        }

        out[j] = temp;
    }
}

template<typename T>
void left_full_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out){
    std::size_t left = m - 1;

    //Left invalid part
    for(std::size_t i = 0; i < left; ++i) {
        const auto hi = i < n - 1 ? i : n - 1;

        T temp = 0.0;

        for(std::size_t j = 0; j <= hi; ++j) {
            temp += in[j] * kernel[i - j];
        }

        out[i] = temp;
    }
}

template<typename T>
void right_full_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out){
    std::size_t right = m - 1;

    auto c = n + m - 1;

    //Right invalid part
    for(std::size_t i = c - right; i < c; ++i) {
        const auto lo = i >= m - 1 ? i - (m - 1) : 0;
        const auto hi = i < n - 1 ? i : n - 1;

        T temp = 0.0;

        for(std::size_t j = lo; j <= hi; ++j) {
            temp += in[j] * kernel[i - j];
        }

        out[i] = temp;
    }
}

template<typename I, typename K, typename C>
void dconv1_full(const I& input, const K& kernel, C&& conv){
    std::size_t left = size(kernel) - 1;

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    left_full_kernel(in, size(input), k, size(kernel), out);
    right_full_kernel(in, size(input), k, size(kernel), out);

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
    left_same_kernel(in, size(input), k, size(kernel), out);
    right_same_kernel(in, size(input), k, size(kernel), out);

    //Central part is a 'valid' convolution
    dconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left);
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
void sconv1_full(const I& input, const K& kernel, C&& conv){
    std::size_t left = size(kernel) - 1;

    auto out = conv.memory_start();
    auto in = input.memory_start();
    auto k = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    left_full_kernel(in, size(input), k, size(kernel), out);
    right_full_kernel(in, size(input), k, size(kernel), out);

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
    left_same_kernel(in, size(input), k, size(kernel), out);
    right_same_kernel(in, size(input), k, size(kernel), out);

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

    __m128d tmp1;
    __m128d tmp2;
    __m128d tmp3;
    __m128d tmp4;
    __m128d res;

    double tmp_res[2] __attribute__ ((aligned (16)));

    for(std::size_t i = 0 ; i < c1 ; ++i){
        for(std::size_t j = 0 ; j < c2 ; ++j){
            res = _mm_setzero_pd();

            for(std::size_t k = i ; k < i + m1; ++k){
                for(std::size_t l = j; l < j + m2 - 1; l += 2){
                    tmp1 = _mm_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm_loadu_pd(kernel + ((i+m1-1-k) * m2 + (j+m2-1-(l+1))));
                    tmp3 = _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0, 1));
                    tmp4 = _mm_mul_pd(tmp3, tmp1);
                    res = _mm_add_pd(res, tmp4);
                }
            }

            _mm_store_pd(tmp_res, res);

            double temp = 0.0;

            if(m2 % 2 != 0){
                for(std::size_t k = i ; k < i + m1; ++k){
                    auto l = j + m2 - 1;
                    temp += in[k * n2 + l] * kernel[(i+m1-1-k) * m2 + (j+m2-1-l)];
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1];
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

    __m128d tmp1;
    __m128d tmp2;
    __m128d tmp3;
    __m128d tmp4;
    __m128d res;

    double tmp_res[2] __attribute__ ((aligned (16)));

    for(std::size_t i = 0 ; i < c1; ++i){
        std::size_t k_lo = std::max<int>(0, i - (m1-1)/2);
        std::size_t k_hi = std::min<int>(n1 - 1, i + m1/2);

        for(std::size_t j = 0 ; j < c2; ++j){
            std::size_t l_lo = std::max<int>(0, j - (m2-1)/2);
            std::size_t l_hi = std::min<int>(n2 - 1, j + m2/2);

            res = _mm_setzero_pd();

            for(std::size_t k = k_lo ; k <= k_hi ; ++k){
                for(std::size_t l = l_lo ; l + 1 <= l_hi; l += 2){
                    tmp1 = _mm_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm_loadu_pd(kernel + (i-k+m1/2) * m2 +(j-(l+1)+m2/2));
                    tmp3 = _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0, 1));
                    tmp4 = _mm_mul_pd(tmp3, tmp1);
                    res = _mm_add_pd(res, tmp4);
                }
            }

            _mm_store_pd(tmp_res, res);

            double temp = 0.0;

            if((l_hi - l_lo + 1) % 2 != 0){
                for(std::size_t k = k_lo ; k <= k_hi ; ++k){
                    temp += in[k * n2 + l_hi] * kernel[(i-k+m1/2) * m2 +(j-l_hi+m2/2)];
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1];
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

    __m128d tmp1;
    __m128d tmp2;
    __m128d tmp3;
    __m128d tmp4;
    __m128d res;

    double tmp_res[2] __attribute__ ((aligned (16)));

    for(std::size_t i = 0 ; i < c1 ; ++i){
        auto k_lo = std::max<int>(0, i - m1 + 1);
        auto k_hi = std::min(n1 - 1, i);

        for(std::size_t j = 0 ; j < c2 ; ++j){
            auto l_lo = std::max<int>(0, j - m2 + 1);
            auto l_hi = std::min(n2 - 1 ,j);

            res = _mm_setzero_pd();

            for(std::size_t k = k_lo ; k <= k_hi ; ++k){
                for(std::size_t l = l_lo; l + 1 <= l_hi ; l += 2){
                    tmp1 = _mm_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm_loadu_pd(kernel + (i - k) * m2 + (j - (l+1)));
                    tmp3 = _mm_shuffle_pd(tmp2, tmp2, _MM_SHUFFLE2(0, 1));
                    tmp4 = _mm_mul_pd(tmp3, tmp1);
                    res = _mm_add_pd(res, tmp4);
                }
            }

            _mm_store_pd(tmp_res, res);

            double temp = 0.0;

            if((l_hi - l_lo + 1) % 2 != 0){
                for(std::size_t k = k_lo ; k <= k_hi ; ++k){
                    temp += in[k * n2 + l_hi] * kernel[(i - k) * m2 + (j - l_hi)];
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1];
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

    __m128 tmp1;
    __m128 tmp2;
    __m128 tmp3;
    __m128 tmp4;
    __m128 res;

    float tmp_res[4] __attribute__ ((aligned (16)));

    for(std::size_t i = 0 ; i < c1 ; ++i){
        for(std::size_t j = 0 ; j < c2 ; ++j){
            res = _mm_setzero_ps();

            for(std::size_t k = i ; k < i + m1; ++k){
                for(std::size_t l = j ; l + 3 < j + m2; l += 4){
                    tmp1 = _mm_loadu_ps(in + k * n2 + l);
                    tmp2 = _mm_loadu_ps(kernel + (i+m1-1-k) * m2 + (j+m2-1-(l+3)));
                    tmp3 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(0, 1, 2, 3));
                    tmp4 = _mm_mul_ps(tmp3, tmp1);
                    res = _mm_add_ps(res, tmp4);
                }
            }

            _mm_store_ps(tmp_res, res);

            float temp = 0.0;

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
void sconv2_valid(const I& input, const K& kernel, C&& conv){
    sconv2_valid_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void sconv2_same_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out){
    auto c1 = n1;
    auto c2 = n2;

    __m128 tmp1;
    __m128 tmp2;
    __m128 tmp3;
    __m128 tmp4;
    __m128 res;

    float tmp_res[4] __attribute__ ((aligned (16)));

    for(std::size_t i = 0 ; i < c1; ++i){
        auto k_lo = std::max<int>(0, i - (m1-1)/2);
        auto k_hi = std::min<int>(n1 - 1, i + m1/2) + 1;

        for(std::size_t j = 0 ; j < c2; ++j){
            auto l_lo = std::max<int>(0, j - (m2-1)/2);
            auto l_hi = std::min<int>(n2 - 1, j + m2/2) + 1;

            res = _mm_setzero_ps();

            for(int k = k_lo ; k < k_hi ; ++k){
                for(std::size_t l = l_lo ; l + 3 < static_cast<std::size_t>(l_hi); l += 4){
                    tmp1 = _mm_loadu_ps(in + k * n2 + l);
                    tmp2 = _mm_loadu_ps(kernel + (i-k+m1/2) * m2 + (j-(l+3)+m2/2));
                    tmp3 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(0, 1, 2, 3));
                    tmp4 = _mm_mul_ps(tmp3, tmp1);
                    res = _mm_add_ps(res, tmp4);
                }
            }

            _mm_store_ps(tmp_res, res);

            float temp = 0.0;

            if((l_hi - l_lo) % 4 != 0){
                auto rem = (l_hi - l_lo) % 4;
                for(int k = k_lo ; k < k_hi; ++k){
                    for(std::size_t l = l_hi - rem ; l < static_cast<std::size_t>(l_hi); ++l){
                        temp += in[k * n2 + l] * kernel[(i-k+m1/2) * m2 + (j-l+m2/2)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];
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
    
    __m128 tmp1;
    __m128 tmp2;
    __m128 tmp3;
    __m128 tmp4;
    __m128 res;

    float tmp_res[4] __attribute__ ((aligned (16)));

    for(std::size_t i = 0 ; i < c1 ; ++i){
        auto k_lo = std::max<int>(0, i - m1 + 1);
        auto k_hi = std::min(n1 - 1, i) + 1;

        for(std::size_t j = 0 ; j < c2 ; ++j){
            auto l_lo = std::max<int>(0, j - m2 + 1);
            auto l_hi = std::min(n2 - 1 , j) + 1;

            res = _mm_setzero_ps();

            for(std::size_t k = k_lo ; k < k_hi; ++k){
                for(std::size_t l = l_lo ; l + 3 < l_hi; l += 4){
                    tmp1 = _mm_loadu_ps(in + k * n2 + l);
                    tmp2 = _mm_loadu_ps(kernel + (i - k) * m2 + (j - (l+3)));
                    tmp3 = _mm_shuffle_ps(tmp2, tmp2, _MM_SHUFFLE(0, 1, 2, 3));
                    tmp4 = _mm_mul_ps(tmp3, tmp1);
                    res = _mm_add_ps(res, tmp4);
                }
            }

            _mm_store_ps(tmp_res, res);

            double temp = 0.0;

            if((l_hi - l_lo) % 4 != 0){
                auto rem = (l_hi - l_lo) % 4;
                for(std::size_t k = k_lo ; k < k_hi; ++k){
                    for(std::size_t l = l_hi - rem ; l < l_hi; ++l){
                        temp += in[k * n2 + l] * kernel[(i - k) * m2 + (j - l)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];
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

} //end of namespace std
} //end of namespace impl
} //end of namespace etl

#endif //__SSE3__

#endif //ETL_VECTORIZE

#endif
