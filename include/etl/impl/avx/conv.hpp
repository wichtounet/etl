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
 *  * 1D convolution with no memory allocation could probably be worked out (needs to be benchmarked)
 *  * Probably some other AVX2 instructions that could improve performances
 *  * Use FMA for 2D convolutions
 */

#if defined(ETL_VECTORIZE_IMPL) && defined(__AVX__)
#include <immintrin.h>

#include "etl/impl/common/conv.hpp"

#endif

namespace etl {

namespace impl {

namespace avx {

#if defined(ETL_VECTORIZE_IMPL) && defined(__AVX__)

ETL_INLINE(__m256d) mm256_reverse_pd(__m256d m1) {
#ifdef __AVX2__
    return _mm256_permute4x64_pd(m1, 0b00011011);
#else
    __m256d tmp;
    tmp = _mm256_permute2f128_pd(m1, m1, 1);
    return _mm256_permute_pd(tmp, 5);
#endif
}

ETL_INLINE(__m256) mm256_reverse_ps(__m256 m1) {
    __m256 tmp;
    tmp = _mm256_permute2f128_ps(m1, m1, 33);
    return _mm256_permute_ps(tmp, 27);
}

ETL_INLINE(float) mm256_hadd_ss(__m256 in) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(in, 1), _mm256_castps256_ps128(in));
    const __m128 x64  = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32  = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

ETL_INLINE(double) mm256_hadd_sd(__m256d in) {
    const __m256d t1 = _mm256_hadd_pd(in, _mm256_permute2f128_pd(in, in, 1));
    const __m256d t2 = _mm256_hadd_pd(t1, t1);
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(t2));
}

inline void dconv1_valid_micro_kernel(const double* in, const std::size_t n, const double* kernel, std::size_t m, double* out, std::size_t first, std::size_t last) {
    auto llast = std::min(n - m + 1, last);

    auto kernel_reverse = aligned_allocate_auto<double>(m);

    std::reverse_copy(kernel, kernel + m, kernel_reverse.get());

    // Loop over the input, 8 times unrolled

    for (std::size_t j = first; j + 7 < llast; j += 8) {
        __m256d r1 = _mm256_setzero_pd();
        __m256d r2 = _mm256_setzero_pd();
        __m256d r3 = _mm256_setzero_pd();
        __m256d r4 = _mm256_setzero_pd();
        __m256d r5 = _mm256_setzero_pd();
        __m256d r6 = _mm256_setzero_pd();
        __m256d r7 = _mm256_setzero_pd();
        __m256d r8 = _mm256_setzero_pd();

        // Compute the convolution with 4 doubles of the kernel at once

        for (std::size_t l = 0; l + 3 < m; l += 4) {
            __m256d k1 = _mm256_load_pd(kernel_reverse.get() + l);

            __m256d i1 = _mm256_loadu_pd(in + (j + 0) + l);
            __m256d i2 = _mm256_loadu_pd(in + (j + 1) + l);
            __m256d i3 = _mm256_loadu_pd(in + (j + 2) + l);
            __m256d i4 = _mm256_loadu_pd(in + (j + 3) + l);

            __m256d t1  = _mm256_mul_pd(i1, k1);
            __m256d t2  = _mm256_mul_pd(i2, k1);
            __m256d t3  = _mm256_mul_pd(i3, k1);
            __m256d t4  = _mm256_mul_pd(i4, k1);

            __m256d i5 = _mm256_loadu_pd(in + (j + 4) + l);
            __m256d i6 = _mm256_loadu_pd(in + (j + 5) + l);
            __m256d i7 = _mm256_loadu_pd(in + (j + 6) + l);
            __m256d i8 = _mm256_loadu_pd(in + (j + 7) + l);

            __m256d t5  = _mm256_mul_pd(i5, k1);
            __m256d t6  = _mm256_mul_pd(i6, k1);
            __m256d t7  = _mm256_mul_pd(i7, k1);
            __m256d t8  = _mm256_mul_pd(i8, k1);

            r1         = _mm256_add_pd(r1, t1);
            r2         = _mm256_add_pd(r2, t2);
            r3         = _mm256_add_pd(r3, t3);
            r4         = _mm256_add_pd(r4, t4);
            r5         = _mm256_add_pd(r5, t5);
            r6         = _mm256_add_pd(r6, t6);
            r7         = _mm256_add_pd(r7, t7);
            r8         = _mm256_add_pd(r8, t8);
        }

        out[j + 0] = mm256_hadd_sd(r1);
        out[j + 1] = mm256_hadd_sd(r2);
        out[j + 2] = mm256_hadd_sd(r3);
        out[j + 3] = mm256_hadd_sd(r4);
        out[j + 4] = mm256_hadd_sd(r5);
        out[j + 5] = mm256_hadd_sd(r6);
        out[j + 6] = mm256_hadd_sd(r7);
        out[j + 7] = mm256_hadd_sd(r8);

        // Remainder loop

        for (std::size_t l = m - m % 4; l < m; ++l) {
            out[j + 0] += in[(j + 0) + l] * kernel_reverse[l];
            out[j + 1] += in[(j + 1) + l] * kernel_reverse[l];
            out[j + 2] += in[(j + 2) + l] * kernel_reverse[l];
            out[j + 3] += in[(j + 3) + l] * kernel_reverse[l];
            out[j + 4] += in[(j + 4) + l] * kernel_reverse[l];
            out[j + 5] += in[(j + 5) + l] * kernel_reverse[l];
            out[j + 6] += in[(j + 6) + l] * kernel_reverse[l];
            out[j + 7] += in[(j + 7) + l] * kernel_reverse[l];
        }
    }

    // Remainder loop for the inputs

    for (std::size_t j = llast - (llast - first) % 8; j < llast; ++j) {
        __m256d r1 = _mm256_setzero_pd();

        // Compute the convolution with 4 doubles

        for (std::size_t l = 0; l + 3 < m; l += 4) {
            __m256d i1 = _mm256_loadu_pd(in + j + l);
            __m256d k1 = _mm256_load_pd(kernel_reverse.get() + l);

            __m256d t1  = _mm256_mul_pd(i1, k1);
            r1         = _mm256_add_pd(r1, t1);
        }

        out[j + 0] = mm256_hadd_sd(r1);

        // Remainder loop

        for (std::size_t l = m - m % 4; l < m; ++l) {
            out[j] += in[j + l] * kernel_reverse[l];
        }
    }
}

inline void sconv1_valid_micro_kernel(const float* __restrict__ in, const std::size_t n, const float* __restrict__ kernel, std::size_t m, float* __restrict__ out, std::size_t first, std::size_t last) {
    auto llast = std::min(n - m + 1, last);

    auto kernel_reverse = aligned_allocate_auto<float>(m);

    std::reverse_copy(kernel, kernel + m, kernel_reverse.get());

    // Loop over the input, 8 times unrolled

    for (std::size_t j = first; j + 7 < llast; j += 8) {
        __m256 r1 = _mm256_setzero_ps();
        __m256 r2 = _mm256_setzero_ps();
        __m256 r3 = _mm256_setzero_ps();
        __m256 r4 = _mm256_setzero_ps();
        __m256 r5 = _mm256_setzero_ps();
        __m256 r6 = _mm256_setzero_ps();
        __m256 r7 = _mm256_setzero_ps();
        __m256 r8 = _mm256_setzero_ps();

        // Compute the convolution with 8 floats of the kernel at once

        for (std::size_t l = 0; l + 7 < m; l += 8) {
            __m256 k1 = _mm256_load_ps(kernel_reverse.get() + l);

            __m256 i1 = _mm256_loadu_ps(in + (j + 0) + l);
            __m256 i2 = _mm256_loadu_ps(in + (j + 1) + l);
            __m256 i3 = _mm256_loadu_ps(in + (j + 2) + l);
            __m256 i4 = _mm256_loadu_ps(in + (j + 3) + l);

            __m256 t1  = _mm256_mul_ps(i1, k1);
            __m256 t2  = _mm256_mul_ps(i2, k1);
            __m256 t3  = _mm256_mul_ps(i3, k1);
            __m256 t4  = _mm256_mul_ps(i4, k1);

            __m256 i5 = _mm256_loadu_ps(in + (j + 4) + l);
            __m256 i6 = _mm256_loadu_ps(in + (j + 5) + l);
            __m256 i7 = _mm256_loadu_ps(in + (j + 6) + l);
            __m256 i8 = _mm256_loadu_ps(in + (j + 7) + l);

            __m256 t5  = _mm256_mul_ps(i5, k1);
            __m256 t6  = _mm256_mul_ps(i6, k1);
            __m256 t7  = _mm256_mul_ps(i7, k1);
            __m256 t8  = _mm256_mul_ps(i8, k1);

            r1         = _mm256_add_ps(r1, t1);
            r2         = _mm256_add_ps(r2, t2);
            r3         = _mm256_add_ps(r3, t3);
            r4         = _mm256_add_ps(r4, t4);
            r5         = _mm256_add_ps(r5, t5);
            r6         = _mm256_add_ps(r6, t6);
            r7         = _mm256_add_ps(r7, t7);
            r8         = _mm256_add_ps(r8, t8);
        }

        out[j + 0] = mm256_hadd_ss(r1);
        out[j + 1] = mm256_hadd_ss(r2);
        out[j + 2] = mm256_hadd_ss(r3);
        out[j + 3] = mm256_hadd_ss(r4);
        out[j + 4] = mm256_hadd_ss(r5);
        out[j + 5] = mm256_hadd_ss(r6);
        out[j + 6] = mm256_hadd_ss(r7);
        out[j + 7] = mm256_hadd_ss(r8);

        // Remainder loop

        for (std::size_t l = m - m % 8; l < m; ++l) {
            out[j + 0] += in[(j + 0) + l] * kernel_reverse[l];
            out[j + 1] += in[(j + 1) + l] * kernel_reverse[l];
            out[j + 2] += in[(j + 2) + l] * kernel_reverse[l];
            out[j + 3] += in[(j + 3) + l] * kernel_reverse[l];
            out[j + 4] += in[(j + 4) + l] * kernel_reverse[l];
            out[j + 5] += in[(j + 5) + l] * kernel_reverse[l];
            out[j + 6] += in[(j + 6) + l] * kernel_reverse[l];
            out[j + 7] += in[(j + 7) + l] * kernel_reverse[l];
        }
    }

    // Remainder loop for the inputs

    for (std::size_t j = llast - (llast - first) % 8; j < llast; ++j) {
        __m256 r1 = _mm256_setzero_ps();

        // Compute the convolution with 8 floats

        for (std::size_t l = 0; l + 7 < m; l += 8) {
            __m256 i1 = _mm256_loadu_ps(in + j + l);
            __m256 k1 = _mm256_load_ps(kernel_reverse.get() + l);

            __m256 t1  = _mm256_mul_ps(i1, k1);
            r1         = _mm256_add_ps(r1, t1);
        }

        out[j + 0] = mm256_hadd_ss(r1);

        // Remainder loop

        for (std::size_t l = m - m % 8; l < m; ++l) {
            out[j] += in[j + l] * kernel_reverse[l];
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

    auto kernel_reverse = aligned_allocate_auto<double>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    std::size_t c1 = n1 - m1 + 1;
    std::size_t c2 = n2 - m2 + 1;

    for (std::size_t i = 0; i < c1; ++i) {
        for (std::size_t j = 0; j + 7 < c2; j += 8) {
            __m256d r1 = _mm256_setzero_pd();
            __m256d r2 = _mm256_setzero_pd();
            __m256d r3 = _mm256_setzero_pd();
            __m256d r4 = _mm256_setzero_pd();

            __m256d r5 = _mm256_setzero_pd();
            __m256d r6 = _mm256_setzero_pd();
            __m256d r7 = _mm256_setzero_pd();
            __m256d r8 = _mm256_setzero_pd();

            for (std::size_t k = 0; k < m1; ++k) {
                for (std::size_t l = 0; l + 3 < m2; l += 4) {
                    __m256d k1 = _mm256_loadu_pd(kernel_reverse.get() + k * m2 + l);

                    __m256d i1 = _mm256_loadu_pd(in + (i + k) * n2 + j + l + 0);
                    __m256d i2 = _mm256_loadu_pd(in + (i + k) * n2 + j + l + 1);
                    __m256d i3 = _mm256_loadu_pd(in + (i + k) * n2 + j + l + 2);
                    __m256d i4 = _mm256_loadu_pd(in + (i + k) * n2 + j + l + 3);

                    __m256d t1 = _mm256_mul_pd(i1, k1);
                    __m256d t2 = _mm256_mul_pd(i2, k1);
                    __m256d t3 = _mm256_mul_pd(i3, k1);
                    __m256d t4 = _mm256_mul_pd(i4, k1);

                    __m256d i5 = _mm256_loadu_pd(in + (i + k) * n2 + j + l + 4);
                    __m256d i6 = _mm256_loadu_pd(in + (i + k) * n2 + j + l + 5);
                    __m256d i7 = _mm256_loadu_pd(in + (i + k) * n2 + j + l + 6);
                    __m256d i8 = _mm256_loadu_pd(in + (i + k) * n2 + j + l + 7);

                    r1         = _mm256_add_pd(r1, t1);
                    r2         = _mm256_add_pd(r2, t2);
                    r3         = _mm256_add_pd(r3, t3);
                    r4         = _mm256_add_pd(r4, t4);

                    __m256d t5 = _mm256_mul_pd(i5, k1);
                    __m256d t6 = _mm256_mul_pd(i6, k1);
                    __m256d t7 = _mm256_mul_pd(i7, k1);
                    __m256d t8 = _mm256_mul_pd(i8, k1);

                    r5         = _mm256_add_pd(r5, t5);
                    r6         = _mm256_add_pd(r6, t6);
                    r7         = _mm256_add_pd(r7, t7);
                    r8         = _mm256_add_pd(r8, t8);
                }
            }

            out[i * c2 + j + 0] = mm256_hadd_sd(r1);
            out[i * c2 + j + 1] = mm256_hadd_sd(r2);
            out[i * c2 + j + 2] = mm256_hadd_sd(r3);
            out[i * c2 + j + 3] = mm256_hadd_sd(r4);
            out[i * c2 + j + 4] = mm256_hadd_sd(r5);
            out[i * c2 + j + 5] = mm256_hadd_sd(r6);
            out[i * c2 + j + 6] = mm256_hadd_sd(r7);
            out[i * c2 + j + 7] = mm256_hadd_sd(r8);
        }

        for (std::size_t j = c2 - c2 % 8; j < c2; ++j) {
            __m256d r1 = _mm256_setzero_pd();

            for (std::size_t k = 0; k < m1; ++k) {
                for (std::size_t l = 0; l + 3 < m2; l += 4) {
                    __m256d k1 = _mm256_loadu_pd(kernel_reverse.get() + k * m2 + l);

                    __m256d i1 = _mm256_loadu_pd(in + (i + k) * n2 + j + l);
                    __m256d t1 = _mm256_mul_pd(i1, k1);
                    r1         = _mm256_add_pd(r1, t1);
                }
            }

            out[i * c2 + j] = mm256_hadd_sd(r1);
        }
    }

    if (m2 % 4 != 0) {
        for (std::size_t i = 0; i < c1; ++i) {
            for (std::size_t j = 0; j < c2; ++j) {
                double temp = 0.0;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = m2 - m2 % 4; l < m2; ++l) {
                        temp += in[(k + i) * n2 + l + j] * kernel_reverse[k * m2 + l];
                    }
                }

                out[i * c2 + j] += temp;
            }
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

    for (std::size_t i = 0; i < c1; ++i) {
        std::size_t k_lo = std::max<int>(0, i - (m1 - 1) / 2);
        std::size_t k_hi = std::min<int>(n1 - 1, i + m1 / 2) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            std::size_t l_lo = std::max<int>(0, j - (m2 - 1) / 2);
            std::size_t l_hi = std::min<int>(n2 - 1, j + m2 / 2) + 1;

            __m256d r1 = _mm256_setzero_pd();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 3 < l_hi; l += 4) {
                    __m256d t2 = _mm256_loadu_pd(kernel + (i - k + m1 / 2) * m2 + (j - (l + 3) + m2 / 2));
                    __m256d i1 = _mm256_loadu_pd(in + k * n2 + l);
                    __m256d k1 = mm256_reverse_pd(t2);
                    __m256d t1 = _mm256_mul_pd(k1, i1);
                    r1         = _mm256_add_pd(r1, t1);
                }
            }

            out[i * c2 + j] = mm256_hadd_sd(r1);

            double temp = 0.0;

            if ((l_hi - l_lo) % 4 != 0) {
                auto rem = (l_hi - l_lo) % 4;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                        temp += in[k * n2 + l] * kernel[(i - k + m1 / 2) * m2 + (j - l + m2 / 2)];
                    }
                }
            }

            out[i * c2 + j] += temp;
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

    for (std::size_t i = 0; i < c1; ++i) {
        auto k_lo = std::max<int>(0, i - m1 + 1);
        auto k_hi = std::min(n1 - 1, i) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            auto l_lo = std::max<int>(0, j - m2 + 1);
            auto l_hi = std::min(n2 - 1, j) + 1;

            __m256d r1 = _mm256_setzero_pd();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 3 < l_hi; l += 4) {
                    __m256d i1 = _mm256_loadu_pd(in + k * n2 + l);
                    __m256d t2 = _mm256_loadu_pd(kernel + (i - k) * m2 + (j - (l + 3)));
                    __m256d k1 = mm256_reverse_pd(t2);
                    __m256d t1 = _mm256_mul_pd(k1, i1);
                    r1         = _mm256_add_pd(r1, t1);
                }
            }

            out[i * c2 + j] = mm256_hadd_sd(r1);

            double temp = 0.0;

            if ((l_hi - l_lo) % 4 != 0) {
                auto rem = (l_hi - l_lo) % 4;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                        temp += in[k * n2 + l] * kernel[(i - k) * m2 + (j - l)];
                    }
                }
            }

            out[i * c2 + j] += temp;
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

    for (std::size_t i = 0; i < c1; ++i) {
        for (std::size_t j = 0; j + 7 < c2; j += 8) {
            __m256 r1 = _mm256_setzero_ps();
            __m256 r2 = _mm256_setzero_ps();
            __m256 r3 = _mm256_setzero_ps();
            __m256 r4 = _mm256_setzero_ps();
            __m256 r5 = _mm256_setzero_ps();
            __m256 r6 = _mm256_setzero_ps();
            __m256 r7 = _mm256_setzero_ps();
            __m256 r8 = _mm256_setzero_ps();

            for (std::size_t k = 0; k < m1; ++k) {
                for (std::size_t l = 0; l + 7 < m2; l += 8) {
                    __m256 k1 = _mm256_loadu_ps(kernel_reverse.get() + k * m2 + l);

                    __m256 i1 = _mm256_loadu_ps(in + (i + k) * n2 + j + 0 + l);
                    __m256 i2 = _mm256_loadu_ps(in + (i + k) * n2 + j + 1 + l);
                    __m256 i3 = _mm256_loadu_ps(in + (i + k) * n2 + j + 2 + l);
                    __m256 i4 = _mm256_loadu_ps(in + (i + k) * n2 + j + 3 + l);

#ifdef __FMA__
                    r1 = _mm256_fmadd_ps(i1, k1, r1);
                    r2 = _mm256_fmadd_ps(i2, k1, r2);
                    r3 = _mm256_fmadd_ps(i3, k1, r3);
                    r4 = _mm256_fmadd_ps(i4, k1, r4);
#else
                    __m256 t1 = _mm256_mul_ps(i1, k1);
                    __m256 t2 = _mm256_mul_ps(i2, k1);
                    __m256 t3 = _mm256_mul_ps(i3, k1);
                    __m256 t4 = _mm256_mul_ps(i4, k1);

                    r1 = _mm256_add_ps(r1, t1);
                    r2 = _mm256_add_ps(r2, t2);
                    r3 = _mm256_add_ps(r3, t3);
                    r4 = _mm256_add_ps(r4, t4);
#endif

                    __m256 i5 = _mm256_loadu_ps(in + (i + k) * n2 + j + 4 + l);
                    __m256 i6 = _mm256_loadu_ps(in + (i + k) * n2 + j + 5 + l);
                    __m256 i7 = _mm256_loadu_ps(in + (i + k) * n2 + j + 6 + l);
                    __m256 i8 = _mm256_loadu_ps(in + (i + k) * n2 + j + 7 + l);

#ifdef __FMA__
                    r5 = _mm256_fmadd_ps(i5, k1, r5);
                    r6 = _mm256_fmadd_ps(i6, k1, r6);
                    r7 = _mm256_fmadd_ps(i7, k1, r7);
                    r8 = _mm256_fmadd_ps(i8, k1, r8);
#else
                    __m256 t5 = _mm256_mul_ps(i5, k1);
                    __m256 t6 = _mm256_mul_ps(i6, k1);
                    __m256 t7 = _mm256_mul_ps(i7, k1);
                    __m256 t8 = _mm256_mul_ps(i8, k1);

                    r5 = _mm256_add_ps(r5, t5);
                    r6 = _mm256_add_ps(r6, t6);
                    r7 = _mm256_add_ps(r7, t7);
                    r8 = _mm256_add_ps(r8, t8);
#endif
                }
            }

            out[i * c2 + j + 0] = mm256_hadd_ss(r1);
            out[i * c2 + j + 1] = mm256_hadd_ss(r2);
            out[i * c2 + j + 2] = mm256_hadd_ss(r3);
            out[i * c2 + j + 3] = mm256_hadd_ss(r4);
            out[i * c2 + j + 4] = mm256_hadd_ss(r5);
            out[i * c2 + j + 5] = mm256_hadd_ss(r6);
            out[i * c2 + j + 6] = mm256_hadd_ss(r7);
            out[i * c2 + j + 7] = mm256_hadd_ss(r8);
        }

        for (std::size_t j = c2 - c2 % 8; j < c2; ++j) {
            __m256 r1 = _mm256_setzero_ps();

            for (std::size_t k = 0; k < m1; ++k) {
                for (std::size_t l = 0; l + 7 < m2; l += 8) {
                    __m256 tmp1 = _mm256_loadu_ps(in + (i + k) * n2 + j + l);
                    __m256 tmp3 = _mm256_loadu_ps(kernel_reverse.get() + k * m2 + l);
                    __m256 tmp4 = _mm256_mul_ps(tmp1, tmp3);
                    r1          = _mm256_add_ps(r1, tmp4);
                }
            }

            out[i * c2 + j] = mm256_hadd_ss(r1);
        }
    }

    if (m2 % 8 != 0) {
        auto rem = m2 % 8;
        for (std::size_t i = 0; i < c1; ++i) {
            for (std::size_t j = 0; j < c2; ++j) {
                float temp = 0.0;
                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = m2 - rem; l < m2; ++l) {
                        temp += in[(i + k) * n2 + j + l] * kernel_reverse[k * m2 + l];
                    }
                }
                out[i * c2 + j] += temp;
            }
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

    for (std::size_t i = 0; i < c1; ++i) {
        auto k_lo        = std::max<int>(0, i - (m1 - 1) / 2);
        std::size_t k_hi = std::min<int>(n1 - 1, i + m1 / 2) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            auto l_lo        = std::max<int>(0, j - (m2 - 1) / 2);
            std::size_t l_hi = std::min<int>(n2 - 1, j + m2 / 2) + 1;

            __m256 r1 = _mm256_setzero_ps();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 7 < l_hi; l += 8) {
                    __m256 i1 = _mm256_loadu_ps(in + k * n2 + l);
                    __m256 t2 = _mm256_loadu_ps(kernel + (i - k + m1 / 2) * m2 + (j - (l + 7) + m2 / 2));
                    __m256 k1 = mm256_reverse_ps(t2);
                    __m256 t1 = _mm256_mul_ps(i1, k1);
                    r1        = _mm256_add_ps(r1, t1);
                }
            }

            out[i * c2 + j] = mm256_hadd_ss(r1);

            float temp = 0.0;

            if ((l_hi - l_lo) % 8 != 0) {
                auto rem = (l_hi - l_lo) % 8;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                        temp += in[k * n2 + l] * kernel[(i - k + m1 / 2) * m2 + (j - l + m2 / 2)];
                    }
                }
            }

            out[i * c2 + j] += temp;
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

    for (std::size_t i = 0; i < c1; ++i) {
        auto k_lo = std::max<int>(0, i - m1 + 1);
        auto k_hi = std::min(n1 - 1, i) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            auto l_lo = std::max<int>(0, j - m2 + 1);
            auto l_hi = std::min(n2 - 1, j) + 1;

            __m256 r1 = _mm256_setzero_ps();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 7 < l_hi; l += 8) {
                    __m256 i1 = _mm256_loadu_ps(in + k * n2 + l);
                    __m256 t2 = _mm256_loadu_ps(kernel + (i - k) * m2 + (j - (l + 7)));
                    __m256 k1 = mm256_reverse_ps(t2);
                    __m256 t1 = _mm256_mul_ps(k1, i1);
                    r1        = _mm256_add_ps(r1, t1);
                }
            }

            out[i * c2 + j] = mm256_hadd_ss(r1);

            double temp = 0.0;

            if ((l_hi - l_lo) % 8 != 0) {
                auto rem = (l_hi - l_lo) % 8;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                        temp += in[k * n2 + l] * kernel[(i - k) * m2 + (j - l)];
                    }
                }
            }

            out[i * c2 + j] += temp;
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

//COVERAGE_EXCLUDE_BEGIN

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

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace avx
} //end of namespace impl
} //end of namespace etl
