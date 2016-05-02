//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*
 * AVX implementation of 1D and 2D convolutions
 *
 * Ideas:
 *  * the tmp_res vectors could be avoided by using hadd instructions
 *  * 1D convolution with no memory allocation could probably be worked out (needs to be benchmarked)
 *  * Probably some other AVX2 instructions that could improve performances
 */

#if defined(ETL_VECTORIZE_IMPL) && defined(__AVX__)
#include <immintrin.h>

#include "etl/allocator.hpp"
#include "etl/impl/common/conv.hpp"

#endif

namespace etl {

namespace impl {

namespace avx {

#if defined(ETL_VECTORIZE_IMPL) && defined(__AVX__)

inline __m256d mm256_reverse_pd(__m256d m1) {
#ifdef __AVX2__
    return _mm256_permute4x64_pd(m1, 0b00011011);
#else
    __m256d tmp;
    tmp = _mm256_permute2f128_pd(m1, m1, 1);
    return _mm256_permute_pd(tmp, 5);
#endif
}

inline __m256 mm256_reverse_ps(__m256 m1) {
    __m256 tmp;
    tmp = _mm256_permute2f128_ps(m1, m1, 33);
    return _mm256_permute_ps(tmp, 27);
}

inline void dconv1_valid_micro_kernel(const double* in, const std::size_t n, const double* kernel, std::size_t m, double* out, std::size_t first, std::size_t last) {
    auto kernel_reverse = aligned_allocate_auto<__m256d>(m);

    //Reverse the kernel

    for (std::size_t i = 0; i < m; i++) {
        kernel_reverse[i] = _mm256_broadcast_sd(kernel + m - i - 1);
    }

    //Compute the convolution, 4 doubles at a time

    auto llast = std::min(n - m + 1, last);

    for (std::size_t i = first; i + 3 < llast; i += 4) {
        __m256d r1 = _mm256_setzero_pd();
        __m256d r2 = _mm256_setzero_pd();
        __m256d r3 = _mm256_setzero_pd();
        __m256d r4 = _mm256_setzero_pd();

        for (std::size_t k = 0; k + 3 < m; k += 4) {
            __m256d t1 = _mm256_loadu_pd(in + i + k);
            __m256d t2 = _mm256_loadu_pd(in + i + k + 1);
            __m256d t3 = _mm256_loadu_pd(in + i + k + 2);
            __m256d t4 = _mm256_loadu_pd(in + i + k + 3);

#ifdef __FMA__
            r1 = _mm256_fmadd_pd(kernel_reverse[k], t1, r1);
            r2 = _mm256_fmadd_pd(kernel_reverse[k + 1], t2, r2);
            r3 = _mm256_fmadd_pd(kernel_reverse[k + 2], t3, r3);
            r4 = _mm256_fmadd_pd(kernel_reverse[k + 3], t4, r4);
#else
            __m256d v1 = _mm256_mul_pd(kernel_reverse[k], t1);
            r1         = _mm256_add_pd(r1, v1);
            __m256d v2 = _mm256_mul_pd(kernel_reverse[k + 1], t2);
            r2         = _mm256_add_pd(r2, v2);
            __m256d v3 = _mm256_mul_pd(kernel_reverse[k + 2], t3);
            r3         = _mm256_add_pd(r3, v3);
            __m256d v4 = _mm256_mul_pd(kernel_reverse[k + 3], t4);
            r4         = _mm256_add_pd(r4, v4);
#endif
        }

        for (std::size_t k = m - m % 4; k < m; ++k) {
            __m256d t1 = _mm256_loadu_pd(in + i + k);

#ifdef __FMA__
            r1 = _mm256_fmadd_pd(kernel_reverse[k], t1, r1);
#else
            __m256d v1 = _mm256_mul_pd(kernel_reverse[k], t1);
            r1         = _mm256_add_pd(r1, v1);
#endif
        }

        __m256d res = _mm256_add_pd(r1, r2);
        res         = _mm256_add_pd(res, r3);
        res         = _mm256_add_pd(res, r4);

        _mm256_storeu_pd(out + i, res);
    }

    //If the number of operations is not even, the last case must be
    //computed separatly

    auto c = llast - first;

    if (c % 4 != 0) {
        auto rem = c % 4;
        for (std::size_t i = c - rem; i < c; ++i) {
            out[i] = 0.0;
            for (std::size_t k = 0; k < m; k++) {
                out[i] += in[i + k] * kernel[m - k - 1];
            }
        }
    }
}

inline void sconv1_valid_micro_kernel(const float* in, const std::size_t n, const float* kernel, std::size_t m, float* out, std::size_t first, std::size_t last) {
    auto kernel_reverse = aligned_allocate_auto<__m256>(m);

    //Reverse the kernel

    for (std::size_t i = 0; i < m; i++) {
        kernel_reverse[i] = _mm256_broadcast_ss(kernel + m - i - 1);
    }

    //Compute the convolution 8 floats at a time

    auto llast = std::min(n - m + 1, last);

    for (std::size_t i = first; i + 7 < llast; i += 8) {
        __m256 r1 = _mm256_setzero_ps();
        __m256 r2 = _mm256_setzero_ps();
        __m256 r3 = _mm256_setzero_ps();
        __m256 r4 = _mm256_setzero_ps();

        for (std::size_t k = 0; k + 3 < m; k += 4) {
            __m256 t1 = _mm256_loadu_ps(in + i + k);
            __m256 t2 = _mm256_loadu_ps(in + i + k + 1);
            __m256 t3 = _mm256_loadu_ps(in + i + k + 2);
            __m256 t4 = _mm256_loadu_ps(in + i + k + 3);

#ifdef __FMA__
            r1 = _mm256_fmadd_ps(kernel_reverse[k], t1, r1);
            r2 = _mm256_fmadd_ps(kernel_reverse[k + 1], t2, r2);
            r3 = _mm256_fmadd_ps(kernel_reverse[k + 2], t3, r3);
            r4 = _mm256_fmadd_ps(kernel_reverse[k + 3], t4, r4);
#else
            __m256 v1  = _mm256_mul_ps(kernel_reverse[k], t1);
            r1         = _mm256_add_ps(r1, v1);
            __m256 v2  = _mm256_mul_ps(kernel_reverse[k + 1], t2);
            r2         = _mm256_add_ps(r2, v2);
            __m256 v3  = _mm256_mul_ps(kernel_reverse[k + 2], t3);
            r3         = _mm256_add_ps(r3, v3);
            __m256 v4  = _mm256_mul_ps(kernel_reverse[k + 3], t4);
            r4         = _mm256_add_ps(r4, v4);
#endif
        }

        for (std::size_t k = m - m % 4; k < m; ++k) {
            __m256 t1 = _mm256_loadu_ps(in + i + k);

#ifdef __FMA__
            r1 = _mm256_fmadd_ps(kernel_reverse[k], t1, r1);
#else
            __m256 v1  = _mm256_mul_ps(kernel_reverse[k], t1);
            r1         = _mm256_add_ps(r1, v1);
#endif
        }

        __m256 res = _mm256_add_ps(r1, r2);
        res        = _mm256_add_ps(res, r3);
        res        = _mm256_add_ps(res, r4);

        _mm256_storeu_ps(out + i, res);
    }

    //Complete the last outputs which are not vectorized

    auto c = llast - first;

    if (c % 8 != 0) {
        auto rem = c % 8;
        for (std::size_t i = c - rem; i < c; ++i) {
            out[i] = 0.0;
            for (std::size_t k = 0; k < m; k++) {
                out[i] += in[i + k] * kernel[m - k - 1];
            }
        }
    }
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_double_precision<I, K, C>::value))>
void conv1_full(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    std::size_t left = size(kernel) - 1;

    double* out      = conv.memory_start();
    const double* in = input.memory_start();
    const double* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_full_kernel(in, size(input), k, size(kernel), out, first, last);
    etl::impl::common::right_full_kernel(in, size(input), k, size(kernel), out, first, last);

    //Central part is a 'valid' convolution
    dconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left, first, last);
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_double_precision<I, K, C>::value))>
void conv1_same(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    std::size_t left = (size(kernel) - 1) / 2;

    double* out      = conv.memory_start();
    const double* in = input.memory_start();
    const double* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_same_kernel(in, size(input), k, size(kernel), out, first, last);
    etl::impl::common::right_same_kernel(in, size(input), k, size(kernel), out, first, last);

    //Central part is a 'valid' convolution
    dconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left, first, last);
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_double_precision<I, K, C>::value))>
void conv1_valid(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    double* out      = conv.memory_start();
    const double* in = input.memory_start();
    const double* k  = kernel.memory_start();

    dconv1_valid_micro_kernel(in, size(input), k, size(kernel), out, first, last);
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_single_precision<I, K, C>::value))>
void conv1_full(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    std::size_t left = size(kernel) - 1;

    float* out      = conv.memory_start();
    const float* in = input.memory_start();
    const float* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_full_kernel(in, size(input), k, size(kernel), out, first, last);
    etl::impl::common::right_full_kernel(in, size(input), k, size(kernel), out, first, last);

    //Central part is a 'valid' convolution
    sconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left, first, last);
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_single_precision<I, K, C>::value))>
void conv1_same(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    std::size_t left = (size(kernel) - 1) / 2;

    float* out      = conv.memory_start();
    const float* in = input.memory_start();
    const float* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_same_kernel(in, size(input), k, size(kernel), out, first, last);
    etl::impl::common::right_same_kernel(in, size(input), k, size(kernel), out, first, last);

    //Central part is a 'valid' convolution
    sconv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left, first, last);
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_single_precision<I, K, C>::value))>
void conv1_valid(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    float* out      = conv.memory_start();
    const float* in = input.memory_start();
    const float* k  = kernel.memory_start();

    sconv1_valid_micro_kernel(in, size(input), k, size(kernel), out, first, last);
}

inline void dconv2_valid_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out) {
    if(m2 < 4){
        etl::impl::sse::conv2_valid_micro_kernel(in, n1, n2, kernel, m1, m2, out);
        return;
    }

    std::size_t c1 = n1 - m1 + 1;
    std::size_t c2 = n2 - m2 + 1;

    __m256d tmp1;
    __m256d tmp2;
    __m256d tmp3;
    __m256d tmp4;
    __m256d res;

    double tmp_res[4] __attribute__((aligned(32)));

    for (std::size_t i = 0; i < c1; ++i) {
        for (std::size_t j = 0; j < c2; ++j) {
            res = _mm256_setzero_pd();

            for (std::size_t k = i; k < i + m1; ++k) {
                for (std::size_t l = j; l + 3 < j + m2; l += 4) {
                    tmp1 = _mm256_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm256_loadu_pd(kernel + ((i + m1 - 1 - k) * m2 + (j + m2 - 1 - (l + 3))));
                    tmp3 = mm256_reverse_pd(tmp2);
                    tmp4 = _mm256_mul_pd(tmp3, tmp1);
                    res  = _mm256_add_pd(res, tmp4);
                }
            }

            _mm256_store_pd(tmp_res, res);

            double temp = 0.0;

            if (m2 % 4 != 0) {
                auto rem = m2 % 4;
                for (std::size_t k = i; k < i + m1; ++k) {
                    for (std::size_t l = j + m2 - rem; l < j + m2; ++l) {
                        temp += in[k * n2 + l] * kernel[(i + m1 - 1 - k) * m2 + (j + m2 - 1 - l)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];
        }
    }
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_double_precision<I, K, C>::value))>
void conv2_valid(const I& input, const K& kernel, C&& conv) {
    dconv2_valid_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void dconv2_same_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out) {
    if(m2 < 4){
        etl::impl::sse::conv2_same_micro_kernel(in, n1, n2, kernel, m1, m2, out);
        return;
    }

    std::size_t c1 = n1;
    std::size_t c2 = n2;

    __m256d tmp1;
    __m256d tmp2;
    __m256d tmp3;
    __m256d tmp4;
    __m256d res;

    double tmp_res[4] __attribute__((aligned(32)));

    for (std::size_t i = 0; i < c1; ++i) {
        std::size_t k_lo = std::max<int>(0, i - (m1 - 1) / 2);
        std::size_t k_hi = std::min<int>(n1 - 1, i + m1 / 2) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            std::size_t l_lo = std::max<int>(0, j - (m2 - 1) / 2);
            std::size_t l_hi = std::min<int>(n2 - 1, j + m2 / 2) + 1;

            res = _mm256_setzero_pd();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 3 < l_hi; l += 4) {
                    tmp1 = _mm256_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm256_loadu_pd(kernel + (i - k + m1 / 2) * m2 + (j - (l + 3) + m2 / 2));
                    tmp3 = mm256_reverse_pd(tmp2);
                    tmp4 = _mm256_mul_pd(tmp3, tmp1);
                    res  = _mm256_add_pd(res, tmp4);
                }
            }

            _mm256_store_pd(tmp_res, res);

            double temp = 0.0;

            if ((l_hi - l_lo) % 4 != 0) {
                auto rem = (l_hi - l_lo) % 4;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                        temp += in[k * n2 + l] * kernel[(i - k + m1 / 2) * m2 + (j - l + m2 / 2)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];
        }
    }
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_double_precision<I, K, C>::value))>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    dconv2_same_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void dconv2_full_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out) {
    if(m2 < 4){
        etl::impl::sse::conv2_full_micro_kernel(in, n1, n2, kernel, m1, m2, out);
        return;
    }

    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    __m256d tmp1;
    __m256d tmp2;
    __m256d tmp3;
    __m256d tmp4;
    __m256d res;

    double tmp_res[4] __attribute__((aligned(32)));

    for (std::size_t i = 0; i < c1; ++i) {
        auto k_lo = std::max<int>(0, i - m1 + 1);
        auto k_hi = std::min(n1 - 1, i) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            auto l_lo = std::max<int>(0, j - m2 + 1);
            auto l_hi = std::min(n2 - 1, j) + 1;

            res = _mm256_setzero_pd();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 3 < l_hi; l += 4) {
                    tmp1 = _mm256_loadu_pd(in + k * n2 + l);
                    tmp2 = _mm256_loadu_pd(kernel + (i - k) * m2 + (j - (l + 3)));
                    tmp3 = mm256_reverse_pd(tmp2);
                    tmp4 = _mm256_mul_pd(tmp3, tmp1);
                    res  = _mm256_add_pd(res, tmp4);
                }
            }

            _mm256_store_pd(tmp_res, res);

            double temp = 0.0;

            if ((l_hi - l_lo) % 4 != 0) {
                auto rem = (l_hi - l_lo) % 4;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                        temp += in[k * n2 + l] * kernel[(i - k) * m2 + (j - l)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3];
        }
    }
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_double_precision<I, K, C>::value))>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    dconv2_full_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void sconv2_valid_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out) {
    if(m2 < 8){
        etl::impl::sse::conv2_valid_micro_kernel(in, n1, n2, kernel, m1, m2, out);
        return;
    }

    auto kernel_reverse = aligned_allocate_auto<float>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    std::size_t c1 = n1 - m1 + 1;
    std::size_t c2 = n2 - m2 + 1;

    float tmp_res[8] __attribute__((aligned(32)));

    for (std::size_t i = 0; i < c1; ++i) {
        for (std::size_t j = 0; j < c2; ++j) {
            __m256 res = _mm256_setzero_ps();

            for (std::size_t k = 0; k < m1; ++k) {
                for (std::size_t l = 0; l + 7 < m2; l += 8) {
                    __m256 tmp1 = _mm256_loadu_ps(in + (i + k) * n2 + j + l);
                    __m256 tmp3 = _mm256_loadu_ps(kernel_reverse.get() + k * m2 + l);
                    __m256 tmp4 = _mm256_mul_ps(tmp1, tmp3);
                    res         = _mm256_add_ps(res, tmp4);
                }
            }

            _mm256_store_ps(tmp_res, res);

            float temp = 0.0;

            if (m2 % 8 != 0) {
                auto rem = m2 % 8;
                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = m2 - rem; l < m2; ++l) {
                        temp += in[(i + k) * n2 + j + l] * kernel_reverse[k * m2 + l];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] + tmp_res[5] + tmp_res[6] + tmp_res[7];
        }
    }
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_single_precision<I, K, C>::value))>
void conv2_valid(const I& input, const K& kernel, C&& conv) {
    sconv2_valid_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void sconv2_same_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out) {
    if(m2 < 8){
        etl::impl::sse::conv2_same_micro_kernel(in, n1, n2, kernel, m1, m2, out);
        return;
    }

    std::size_t c1 = n1;
    std::size_t c2 = n2;

    __m256 tmp1;
    __m256 tmp2;
    __m256 tmp3;
    __m256 tmp4;
    __m256 res;

    float tmp_res[8] __attribute__((aligned(32)));

    for (std::size_t i = 0; i < c1; ++i) {
        auto k_lo        = std::max<int>(0, i - (m1 - 1) / 2);
        std::size_t k_hi = std::min<int>(n1 - 1, i + m1 / 2) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            auto l_lo        = std::max<int>(0, j - (m2 - 1) / 2);
            std::size_t l_hi = std::min<int>(n2 - 1, j + m2 / 2) + 1;

            res = _mm256_setzero_ps();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 7 < l_hi; l += 8) {
                    tmp1 = _mm256_loadu_ps(in + k * n2 + l);
                    tmp2 = _mm256_loadu_ps(kernel + (i - k + m1 / 2) * m2 + (j - (l + 7) + m2 / 2));
                    tmp3 = mm256_reverse_ps(tmp2);
                    tmp4 = _mm256_mul_ps(tmp3, tmp1);
                    res  = _mm256_add_ps(res, tmp4);
                }
            }

            _mm256_store_ps(tmp_res, res);

            float temp = 0.0;

            if ((l_hi - l_lo) % 8 != 0) {
                auto rem = (l_hi - l_lo) % 8;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                        temp += in[k * n2 + l] * kernel[(i - k + m1 / 2) * m2 + (j - l + m2 / 2)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] + tmp_res[5] + tmp_res[6] + tmp_res[7];
        }
    }
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_single_precision<I, K, C>::value))>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    sconv2_same_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

inline void sconv2_full_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out) {
    if(m2 < 8){
        etl::impl::sse::conv2_full_micro_kernel(in, n1, n2, kernel, m1, m2, out);
        return;
    }

    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    __m256 tmp1;
    __m256 tmp2;
    __m256 tmp3;
    __m256 tmp4;
    __m256 res;

    float tmp_res[8] __attribute__((aligned(32)));

    for (std::size_t i = 0; i < c1; ++i) {
        auto k_lo = std::max<int>(0, i - m1 + 1);
        auto k_hi = std::min(n1 - 1, i) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            auto l_lo = std::max<int>(0, j - m2 + 1);
            auto l_hi = std::min(n2 - 1, j) + 1;

            res = _mm256_setzero_ps();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 7 < l_hi; l += 8) {
                    tmp1 = _mm256_loadu_ps(in + k * n2 + l);
                    tmp2 = _mm256_loadu_ps(kernel + (i - k) * m2 + (j - (l + 7)));
                    tmp3 = mm256_reverse_ps(tmp2);
                    tmp4 = _mm256_mul_ps(tmp3, tmp1);
                    res  = _mm256_add_ps(res, tmp4);
                }
            }

            _mm256_store_ps(tmp_res, res);

            double temp = 0.0;

            if ((l_hi - l_lo) % 8 != 0) {
                auto rem = (l_hi - l_lo) % 8;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                        temp += in[k * n2 + l] * kernel[(i - k) * m2 + (j - l)];
                    }
                }
            }

            out[i * c2 + j] = temp + tmp_res[0] + tmp_res[1] + tmp_res[2] + tmp_res[3] + tmp_res[4] + tmp_res[5] + tmp_res[6] + tmp_res[7];
        }
    }
}

template <typename I, typename K, typename C, cpp_enable_if((all_dma<I, K, C>::value && all_single_precision<I, K, C>::value))>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    sconv2_full_micro_kernel(
        input.memory_start(), etl::rows(input), etl::columns(input),
        kernel.memory_start(), etl::rows(kernel), etl::columns(kernel),
        conv.memory_start());
}

//To allow compilation

template <typename I, typename K, typename C, cpp_enable_if(!all_dma<I, K, C>::value)>
void conv1_valid(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/, std::size_t /*first*/, std::size_t /*last*/) {}

template <typename I, typename K, typename C, cpp_enable_if(!all_dma<I, K, C>::value)>
void conv1_same(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/, std::size_t /*first*/, std::size_t /*last*/) {}

template <typename I, typename K, typename C, cpp_enable_if(!all_dma<I, K, C>::value)>
void conv1_full(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/, std::size_t /*first*/, std::size_t /*last*/) {}

template <typename I, typename K, typename C, cpp_enable_if(!all_dma<I, K, C>::value)>
void conv2_valid(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {}

template <typename I, typename K, typename C, cpp_enable_if(!all_dma<I, K, C>::value)>
void conv2_same(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {}

template <typename I, typename K, typename C, cpp_enable_if(!all_dma<I, K, C>::value)>
void conv2_full(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {}

#else

template <typename I, typename K, typename C>
void conv1_full(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/, std::size_t /*first*/, std::size_t /*last*/) {
    cpp_unreachable("AVX not available/enabled");
}

template <typename I, typename K, typename C>
void conv1_same(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/, std::size_t /*first*/, std::size_t /*last*/) {
    cpp_unreachable("AVX not available/enabled");
}

template <typename I, typename K, typename C>
void conv1_valid(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/, std::size_t /*first*/, std::size_t /*last*/) {
    cpp_unreachable("AVX not available/enabled");
}

template <typename I, typename K, typename C>
void conv2_valid(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
    cpp_unreachable("AVX not available/enabled");
}

template <typename I, typename K, typename C>
void conv2_same(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
    cpp_unreachable("AVX not available/enabled");
}

template <typename I, typename K, typename C>
void conv2_full(const I& /*input*/, const K& /*kernel*/, C&& /*conv*/) {
    cpp_unreachable("AVX not available/enabled");
}

#endif

} //end of namespace avx
} //end of namespace impl
} //end of namespace etl
