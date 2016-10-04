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

inline void conv1_valid_micro_kernel(const double* in, const std::size_t n, const double* kernel, std::size_t m, double* out, std::size_t first, std::size_t last) {
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

inline void conv1_valid_micro_kernel(const float* __restrict__ in, const std::size_t n, const float* __restrict__ kernel, std::size_t m, float* __restrict__ out, std::size_t first, std::size_t last) {
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

template <typename I, typename K, typename C>
void conv1_full(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    std::size_t left = size(kernel) - 1;

    auto* out      = conv.memory_start();
    const auto* in = input.memory_start();
    const auto* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_full_kernel(in, size(input), k, size(kernel), out, first, last);
    etl::impl::common::right_full_kernel(in, size(input), k, size(kernel), out, first, last);

    //Central part is a 'valid' convolution
    conv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left, first, last);
}

template <typename I, typename K, typename C>
void conv1_same(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    std::size_t left = (size(kernel) - 1) / 2;

    auto* out      = conv.memory_start();
    const auto* in = input.memory_start();
    const auto* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_same_kernel(in, size(input), k, size(kernel), out, first, last);
    etl::impl::common::right_same_kernel(in, size(input), k, size(kernel), out, first, last);

    //Central part is a 'valid' convolution
    conv1_valid_micro_kernel(in, size(input), k, size(kernel), out + left, first, last);
}

template <typename I, typename K, typename C>
void conv1_valid(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    auto* out      = conv.memory_start();
    const auto* in = input.memory_start();
    const auto* k  = kernel.memory_start();

    conv1_valid_micro_kernel(in, size(input), k, size(kernel), out, first, last);
}

inline void conv2_valid_flipped_border(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out, double beta, std::size_t i, std::size_t j, size_t s1, size_t s2, size_t p1, size_t p2) {
    const std::size_t c2 = (n2 - m2 + 2 * p2) / s2 + 1;

    double temp = 0.0;

    const auto s_i = i * s1;
    const auto s_j = j * s2;

    for (std::size_t k = 0; k < m1; ++k) {
        for (std::size_t l = 0; l < m2; ++l) {
            if(s_i + k >= p1 && (s_i + k) - p1 < n1 && s_j + l >= p2 && (s_j + l) - p2 < n2){
                const size_t i_i = (s_i + k) - p1;
                const size_t i_j = (s_j + l) - p2;

                temp += in[i_i * n2 + i_j] * kernel[k * m2 + l];
            }
        }
    }

    if(beta == 0.0){
        out[i * c2 + j] = temp;
    } else {
        out[i * c2 + j] = beta * out[i * c2 + j] + temp;
    }
}

inline void conv2_valid_flipped_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out, double beta, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(m2 < 4){
        etl::impl::sse::conv2_valid_flipped_micro_kernel(in, n1, n2, kernel, m1, m2, out, beta, s1, s2, p1, p2);
        return;
    }

    const std::size_t c1 = (n1 - m1 + 2 * p1) / s1 + 1;
    const std::size_t c2 = (n2 - m2 + 2 * p2) / s2 + 1;

    if(p1 || p2){
        for (std::size_t i = 0; i < p1; ++i) {
            for (std::size_t j = 0; j < c2; ++j) {
                conv2_valid_flipped_border(in, n1, n2, kernel, m1, m2, out, beta, i, j, s1, s2, p1, p2);
            }
        }

        for (std::size_t i = c1 - p1; i < c1; ++i) {
            for (std::size_t j = 0; j < c2; ++j) {
                conv2_valid_flipped_border(in, n1, n2, kernel, m1, m2, out, beta, i, j, s1, s2, p1, p2);
            }
        }

        for (std::size_t j = 0; j < p2; ++j) {
            for (std::size_t i = p1; i < c1 - p1; ++i) {
                conv2_valid_flipped_border(in, n1, n2, kernel, m1, m2, out, beta, i, j, s1, s2, p1, p2);
            }
        }

        for (std::size_t j = c2 - p2; j < c2; ++j) {
            for (std::size_t i = p1; i < c1 - p1; ++i) {
                conv2_valid_flipped_border(in, n1, n2, kernel, m1, m2, out, beta, i, j, s1, s2, p1, p2);
            }
        }
    }

    if(beta == 0.0){
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 7 < c2 - p2; j += 8) {
                __m256d r1 = _mm256_setzero_pd();
                __m256d r2 = _mm256_setzero_pd();
                __m256d r3 = _mm256_setzero_pd();
                __m256d r4 = _mm256_setzero_pd();

                __m256d r5 = _mm256_setzero_pd();
                __m256d r6 = _mm256_setzero_pd();
                __m256d r7 = _mm256_setzero_pd();
                __m256d r8 = _mm256_setzero_pd();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m256d k1 = _mm256_loadu_pd(kernel + k * m2 + l);

                        __m256d i1 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 0);
                        __m256d i2 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 1);
                        __m256d i3 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 2);
                        __m256d i4 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 3);

                        __m256d t1 = _mm256_mul_pd(i1, k1);
                        __m256d t2 = _mm256_mul_pd(i2, k1);
                        __m256d t3 = _mm256_mul_pd(i3, k1);
                        __m256d t4 = _mm256_mul_pd(i4, k1);

                        __m256d i5 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 4);
                        __m256d i6 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 5);
                        __m256d i7 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 6);
                        __m256d i8 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 7);

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

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 8; j < c2 - p2; ++j) {
                __m256d r1 = _mm256_setzero_pd();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m256d k1 = _mm256_loadu_pd(kernel + k * m2 + l);

                        __m256d i1 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l);
                        __m256d t1 = _mm256_mul_pd(i1, k1);
                        r1         = _mm256_add_pd(r1, t1);
                    }
                }

                out[i * c2 + j] = mm256_hadd_sd(r1);
            }
        }
    } else {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 7 < c2 - p2; j += 8) {
                __m256d r1 = _mm256_setzero_pd();
                __m256d r2 = _mm256_setzero_pd();
                __m256d r3 = _mm256_setzero_pd();
                __m256d r4 = _mm256_setzero_pd();

                __m256d r5 = _mm256_setzero_pd();
                __m256d r6 = _mm256_setzero_pd();
                __m256d r7 = _mm256_setzero_pd();
                __m256d r8 = _mm256_setzero_pd();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m256d k1 = _mm256_loadu_pd(kernel + k * m2 + l);

                        __m256d i1 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 0);
                        __m256d i2 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 1);
                        __m256d i3 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 2);
                        __m256d i4 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 3);

                        __m256d t1 = _mm256_mul_pd(i1, k1);
                        __m256d t2 = _mm256_mul_pd(i2, k1);
                        __m256d t3 = _mm256_mul_pd(i3, k1);
                        __m256d t4 = _mm256_mul_pd(i4, k1);

                        __m256d i5 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 4);
                        __m256d i6 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 5);
                        __m256d i7 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 6);
                        __m256d i8 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l + 7);

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

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + mm256_hadd_sd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + mm256_hadd_sd(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + mm256_hadd_sd(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + mm256_hadd_sd(r4);
                out[i * c2 + j + 4] = beta * out[i * c2 + j + 4] + mm256_hadd_sd(r5);
                out[i * c2 + j + 5] = beta * out[i * c2 + j + 5] + mm256_hadd_sd(r6);
                out[i * c2 + j + 6] = beta * out[i * c2 + j + 6] + mm256_hadd_sd(r7);
                out[i * c2 + j + 7] = beta * out[i * c2 + j + 7] + mm256_hadd_sd(r8);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 8; j < c2 - p2; ++j) {
                __m256d r1 = _mm256_setzero_pd();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m256d k1 = _mm256_loadu_pd(kernel + k * m2 + l);

                        __m256d i1 = _mm256_loadu_pd(in + (i_i + k) * n2 + i_j + l);
                        __m256d t1 = _mm256_mul_pd(i1, k1);
                        r1         = _mm256_add_pd(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm256_hadd_sd(r1);
            }
        }
    }

    if (m2 % 4 != 0) {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j < c2 - p2; ++j) {
                double temp = 0.0;

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = m2 - m2 % 4; l < m2; ++l) {
                        temp += in[(k + i_i) * n2 + l + i_j] * kernel[k * m2 + l];
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    }
}

inline void conv2_valid_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out, double beta, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(m2 < 4){
        etl::impl::sse::conv2_valid_micro_kernel(in, n1, n2, kernel, m1, m2, out, beta, s1, s2, p1, p2);
        return;
    }

    auto kernel_reverse = aligned_allocate_auto<double>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_valid_flipped_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out, beta, s1, s2, p1, p2);
}

inline void conv2_same_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out) {
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

inline void conv2_same_flipped_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out) {
    auto kernel_reverse = aligned_allocate_auto<double>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_same_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out);
}

inline void conv2_full_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out, double beta) {
    if(m2 < 4){
        etl::impl::sse::conv2_full_micro_kernel(in, n1, n2, kernel, m1, m2, out, beta);
        return;
    }

    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    if(beta == 0.0){
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
    } else {
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

                out[i * c2 + j] = beta * out[i * c2 + j] + mm256_hadd_sd(r1);

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
}

inline void conv2_full_flipped_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out, double beta) {
    if(m2 < 4){
        etl::impl::sse::conv2_full_flipped_micro_kernel(in, n1, n2, kernel, m1, m2, out, beta);
        return;
    }

    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    if (beta == 0.0) {
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
                        __m256d k1 = _mm256_loadu_pd(kernel + (m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l)));
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
                            temp += in[k * n2 + l] * kernel[(m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l))];
                        }
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    } else {
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
                        __m256d k1 = _mm256_loadu_pd(kernel + (m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l)));
                        __m256d t1 = _mm256_mul_pd(k1, i1);
                        r1         = _mm256_add_pd(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm256_hadd_sd(r1);

                double temp = 0.0;

                if ((l_hi - l_lo) % 4 != 0) {
                    auto rem = (l_hi - l_lo) % 4;
                    for (std::size_t k = k_lo; k < k_hi; ++k) {
                        for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                            temp += in[k * n2 + l] * kernel[(m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l))];
                        }
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    }
}

inline void conv2_valid_flipped_border(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out, float beta, std::size_t i, std::size_t j, size_t s1, size_t s2, size_t p1, size_t p2) {
    const std::size_t c2 = (n2 - m2 + 2 * p2) / s2 + 1;

    float temp = 0.0f;

    const auto s_i = i * s1;
    const auto s_j = j * s2;

    for (std::size_t k = 0; k < m1; ++k) {
        for (std::size_t l = 0; l < m2; ++l) {
            if(s_i + k >= p1 && (s_i + k) - p1 < n1 && s_j + l >= p2 && (s_j + l) - p2 < n2){
                const size_t i_i = (s_i + k) - p1;
                const size_t i_j = (s_j + l) - p2;

                temp += in[i_i * n2 + i_j] * kernel[k * m2 + l];
            }
        }
    }

    if(beta == 0.0f){
        out[i * c2 + j] = temp;
    } else {
        out[i * c2 + j] = beta * out[i * c2 + j] + temp;
    }
}

inline void conv2_valid_flipped_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out, float beta, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(m2 < 8){
        etl::impl::sse::conv2_valid_flipped_micro_kernel(in, n1, n2, kernel, m1, m2, out, beta, s1, s2, p1, p2);
        return;
    }

    const std::size_t c1 = (n1 - m1 + 2 * p1) / s1 + 1;
    const std::size_t c2 = (n2 - m2 + 2 * p2) / s2 + 1;

    if(p1 || p2){
        for (std::size_t i = 0; i < p1; ++i) {
            for (std::size_t j = 0; j < c2; ++j) {
                conv2_valid_flipped_border(in, n1, n2, kernel, m1, m2, out, beta, i, j, s1, s2, p1, p2);
            }
        }

        for (std::size_t i = c1 - p1; i < c1; ++i) {
            for (std::size_t j = 0; j < c2; ++j) {
                conv2_valid_flipped_border(in, n1, n2, kernel, m1, m2, out, beta, i, j, s1, s2, p1, p2);
            }
        }

        for (std::size_t j = 0; j < p2; ++j) {
            for (std::size_t i = p1; i < c1 - p1; ++i) {
                conv2_valid_flipped_border(in, n1, n2, kernel, m1, m2, out, beta, i, j, s1, s2, p1, p2);
            }
        }

        for (std::size_t j = c2 - p2; j < c2; ++j) {
            for (std::size_t i = p1; i < c1 - p1; ++i) {
                conv2_valid_flipped_border(in, n1, n2, kernel, m1, m2, out, beta, i, j, s1, s2, p1, p2);
            }
        }
    }

    // TODO The unrolling factor of 8 is too much for several architectures

    if(beta == 0.0f){
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 7 < c2 - p2; j += 8) {
                __m256 r1 = _mm256_setzero_ps();
                __m256 r2 = _mm256_setzero_ps();
                __m256 r3 = _mm256_setzero_ps();
                __m256 r4 = _mm256_setzero_ps();
                __m256 r5 = _mm256_setzero_ps();
                __m256 r6 = _mm256_setzero_ps();
                __m256 r7 = _mm256_setzero_ps();
                __m256 r8 = _mm256_setzero_ps();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 7 < m2; l += 8) {
                        __m256 k1 = _mm256_loadu_ps(kernel + k * m2 + l);

                        __m256 i1 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 0 + l);
                        __m256 i2 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 1 + l);
                        __m256 i3 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 2 + l);
                        __m256 i4 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 3 + l);

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

                        __m256 i5 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 4 + l);
                        __m256 i6 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 5 + l);
                        __m256 i7 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 6 + l);
                        __m256 i8 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 7 + l);

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

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 8; j < c2 - p2; ++j) {
                __m256 r1 = _mm256_setzero_ps();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 7 < m2; l += 8) {
                        __m256 tmp1 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + l);
                        __m256 tmp3 = _mm256_loadu_ps(kernel + k * m2 + l);
                        __m256 tmp4 = _mm256_mul_ps(tmp1, tmp3);
                        r1          = _mm256_add_ps(r1, tmp4);
                    }
                }

                out[i * c2 + j] = mm256_hadd_ss(r1);
            }
        }
    } else {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 7 < c2 - p2; j += 8) {
                __m256 r1 = _mm256_setzero_ps();
                __m256 r2 = _mm256_setzero_ps();
                __m256 r3 = _mm256_setzero_ps();
                __m256 r4 = _mm256_setzero_ps();
                __m256 r5 = _mm256_setzero_ps();
                __m256 r6 = _mm256_setzero_ps();
                __m256 r7 = _mm256_setzero_ps();
                __m256 r8 = _mm256_setzero_ps();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 7 < m2; l += 8) {
                        __m256 k1 = _mm256_loadu_ps(kernel + k * m2 + l);

                        __m256 i1 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 0 + l);
                        __m256 i2 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 1 + l);
                        __m256 i3 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 2 + l);
                        __m256 i4 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 3 + l);

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

                        __m256 i5 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 4 + l);
                        __m256 i6 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 5 + l);
                        __m256 i7 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 6 + l);
                        __m256 i8 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + 7 + l);

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

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + mm256_hadd_ss(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + mm256_hadd_ss(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + mm256_hadd_ss(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + mm256_hadd_ss(r4);
                out[i * c2 + j + 4] = beta * out[i * c2 + j + 4] + mm256_hadd_ss(r5);
                out[i * c2 + j + 5] = beta * out[i * c2 + j + 5] + mm256_hadd_ss(r6);
                out[i * c2 + j + 6] = beta * out[i * c2 + j + 6] + mm256_hadd_ss(r7);
                out[i * c2 + j + 7] = beta * out[i * c2 + j + 7] + mm256_hadd_ss(r8);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 8; j < c2 - p2; ++j) {
                __m256 r1 = _mm256_setzero_ps();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 7 < m2; l += 8) {
                        __m256 tmp1 = _mm256_loadu_ps(in + (i_i + k) * n2 + i_j + l);
                        __m256 tmp3 = _mm256_loadu_ps(kernel + k * m2 + l);
                        __m256 tmp4 = _mm256_mul_ps(tmp1, tmp3);
                        r1          = _mm256_add_ps(r1, tmp4);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm256_hadd_ss(r1);
            }
        }
    }

    if (m2 % 8 != 0) {
        auto rem = m2 % 8;
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j < c2 - p2; ++j) {
                float temp = 0.0;

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = m2 - rem; l < m2; ++l) {
                        temp += in[(i_i + k) * n2 + i_j + l] * kernel[k * m2 + l];
                    }
                }
                out[i * c2 + j] += temp;
            }
        }
    }
}

inline void conv2_valid_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out, float beta, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(m2 < 8){
        etl::impl::sse::conv2_valid_micro_kernel(in, n1, n2, kernel, m1, m2, out, beta, s1, s2, p1, p2);
        return;
    }

    auto kernel_reverse = aligned_allocate_auto<float>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_valid_flipped_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out, beta, s1, s2, p1, p2);
}

inline void conv2_same_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out) {
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

inline void conv2_same_flipped_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out) {
    auto kernel_reverse = aligned_allocate_auto<float>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_same_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out);
}

inline void conv2_full_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out, float beta) {
    if(m2 < 8){
        etl::impl::sse::conv2_full_micro_kernel(in, n1, n2, kernel, m1, m2, out, beta);
        return;
    }

    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    if(beta == 0.0f){
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
    } else {
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

                out[i * c2 + j] = beta * out[i * c2 + j] + mm256_hadd_ss(r1);

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
}

inline void conv2_full_flipped_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out, float beta) {
    if (m2 < 8) {
        etl::impl::sse::conv2_full_flipped_micro_kernel(in, n1, n2, kernel, m1, m2, out, beta);
        return;
    }

    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    if (beta == 0.0f) {
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
                        __m256 k1 = _mm256_loadu_ps(kernel + (m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l)));
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
                            temp += in[k * n2 + l] * kernel[(m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l))];
                        }
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    } else {
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
                        __m256 k1 = _mm256_loadu_ps(kernel + (m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l)));
                        __m256 t1 = _mm256_mul_ps(k1, i1);
                        r1        = _mm256_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm256_hadd_ss(r1);

                double temp = 0.0;

                if ((l_hi - l_lo) % 8 != 0) {
                    auto rem = (l_hi - l_lo) % 8;
                    for (std::size_t k = k_lo; k < k_hi; ++k) {
                        for (std::size_t l = l_hi - rem; l < l_hi; ++l) {
                            temp += in[k * n2 + l] * kernel[(m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l))];
                        }
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    }
}

template <typename T>
void pad_2d_input(const opaque_memory<T, 2>& in, opaque_memory<T, 2>& out, size_t p1, size_t p2) {
    auto in_m = in.memory_start();
    auto out_m = out.memory_start();

    for (std::size_t i = 0; i < in.template dim<0>(); ++i) {
        direct_copy_n(in_m + i * in.template dim<1>(), out_m + (i + p1) * out.template dim<1>() + p2, in.template dim<1>());
    }
}

template <typename T>
void conv2_valid(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(p1 || p2){
        const auto ws_h = input.template dim<0>() + 2 * p1;
        const auto ws_w = input.template dim<1>() + 2 * p2;

        if(ws_h * ws_w * sizeof(T) < max_workspace){
            etl::dyn_matrix<T, 2> workspace(ws_h, ws_w, T(0));
            auto ws_direct = workspace.direct();

            pad_2d_input(input, ws_direct, p1, p2);

            conv2_valid_micro_kernel(
                workspace.memory_start(), ws_h, ws_w,
                kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
                conv.memory_start(), 0.0, s1, s2, 0, 0);

            return;
        }
    }

    conv2_valid_micro_kernel(
        input.memory_start(), input.template dim<0>(), input.template dim<1>(),
        kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
        conv.memory_start(), 0.0, s1, s2, p1, p2);
}

template <typename T>
void conv2_valid_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    conv2_valid_flipped_micro_kernel(
        input.memory_start(), input.template dim<0>(), input.template dim<1>(),
        kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
        conv.memory_start(), 0.0, s1, s2, p1, p2);
}

template <typename T>
void conv2_valid_multi(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    for(std::size_t k = 0; k < kernel.template dim<0>(); ++k){
        auto kk = kernel.template dim<1>() * kernel.template dim<2>();
        auto cc = conv.template dim<1>() * conv.template dim<2>();

        conv2_valid_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
            conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
    }
}

template <typename T>
void conv2_valid_multi_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    for(std::size_t k = 0; k < kernel.template dim<0>(); ++k){
        auto kk = kernel.template dim<1>() * kernel.template dim<2>();
        auto cc = conv.template dim<1>() * conv.template dim<2>();

        conv2_valid_flipped_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
            conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
    }
}

template <typename T>
void conv4_valid(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(kernel.dim(1) > 0){
        auto conv_i_inc = conv.dim(1) * conv.dim(2) * conv.dim(3);
        auto conv_k_inc = conv.dim(2) * conv.dim(3);

        auto kernel_k_inc = kernel.dim(1) * kernel.dim(2) * kernel.dim(3);
        auto kernel_c_inc = kernel.dim(2) * kernel.dim(3);

        auto input_i_inc = input.dim(1) * input.dim(2) * input.dim(3);
        auto input_c_inc = input.dim(2) * input.dim(3);

        for(std::size_t i = 0; i < input.dim(0); ++i){
            for(std::size_t k = 0; k < kernel.dim(0); ++k){
                conv2_valid_micro_kernel(
                    input.memory_start() + i * input_i_inc, input.dim(2), input.dim(3),
                    kernel.memory_start() + k * kernel_k_inc, kernel.dim(2), kernel.dim(3),
                    conv.memory_start() + i * conv_i_inc + k * conv_k_inc, 0.0, s1, s2, p1, p2);

                for(std::size_t c = 1; c < kernel.dim(1); ++c){
                    conv2_valid_micro_kernel(
                        input.memory_start() + i * input_i_inc + c * input_c_inc, input.dim(2), input.dim(3),
                        kernel.memory_start() + k * kernel_k_inc + c * kernel_c_inc, kernel.dim(2), kernel.dim(3),
                        conv.memory_start() + i * conv_i_inc + k * conv_k_inc, 1.0, s1, s2, p1, p2);
                }
            }
        }
    }
}

template <typename T>
void conv4_valid_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(kernel.dim(1) > 0){
        auto conv_i_inc = conv.dim(1) * conv.dim(2) * conv.dim(3);
        auto conv_k_inc = conv.dim(2) * conv.dim(3);

        auto kernel_k_inc = kernel.dim(1) * kernel.dim(2) * kernel.dim(3);
        auto kernel_c_inc = kernel.dim(2) * kernel.dim(3);

        auto input_i_inc = input.dim(1) * input.dim(2) * input.dim(3);
        auto input_c_inc = input.dim(2) * input.dim(3);

        for(std::size_t i = 0; i < input.dim(0); ++i){
            for(std::size_t k = 0; k < kernel.dim(0); ++k){
                conv2_valid_flipped_micro_kernel(
                    input.memory_start() + i * input_i_inc, input.dim(2), input.dim(3),
                    kernel.memory_start() + k * kernel_k_inc, kernel.dim(2), kernel.dim(3),
                    conv.memory_start() + i * conv_i_inc + k * conv_k_inc, 0.0, s1, s2, p1, p2);

                for(std::size_t c = 1; c < kernel.dim(1); ++c){
                    conv2_valid_flipped_micro_kernel(
                        input.memory_start() + i * input_i_inc + c * input_c_inc, input.dim(2), input.dim(3),
                        kernel.memory_start() + k * kernel_k_inc + c * kernel_c_inc, kernel.dim(2), kernel.dim(3),
                        conv.memory_start() + i * conv_i_inc + k * conv_k_inc, 1.0, s1, s2, p1, p2);
                }
            }
        }
    }
}

template <typename T>
void conv4_valid_filter(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    if (input.dim(0) > 0) {
        auto conv_k_inc = conv.dim(1) * conv.dim(2) * conv.dim(3);
        auto conv_c_inc = conv.dim(2) * conv.dim(3);

        auto kernel_i_inc = kernel.dim(1) * kernel.dim(2) * kernel.dim(3);
        auto kernel_k_inc = kernel.dim(2) * kernel.dim(3);

        auto input_i_inc = input.dim(1) * input.dim(2) * input.dim(3);
        auto input_c_inc = input.dim(2) * input.dim(3);

        //i = 0
        for (std::size_t k = 0; k < kernel.dim(1); ++k) {
            for(std::size_t c = 0; c < input.dim(1); ++c){
                conv2_valid_micro_kernel(
                    input.memory_start() + 0 * input_i_inc + c * input_c_inc, input.dim(2), input.dim(3),
                    kernel.memory_start() + 0 * kernel_i_inc + k * kernel_k_inc, kernel.dim(2), kernel.dim(3),
                    conv.memory_start() + k * conv_k_inc + c * conv_c_inc, 0.0, s1, s2, p1, p2);
            }
        }

        for (std::size_t i = 1; i < input.dim(0); ++i) {
            for (std::size_t k = 0; k < kernel.dim(1); ++k) {
                for(std::size_t c = 0; c < input.dim(1); ++c){
                    conv2_valid_micro_kernel(
                        input.memory_start() + i * input_i_inc + c * input_c_inc, input.dim(2), input.dim(3),
                        kernel.memory_start() + i * kernel_i_inc + k * kernel_k_inc, kernel.dim(2), kernel.dim(3),
                        conv.memory_start() + k * conv_k_inc + c * conv_c_inc, 1.0, s1, s2, p1, p2);
                }
            }
        }
    }
}

template <typename T>
void conv4_valid_filter_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    if (input.dim(0) > 0) {
        auto conv_k_inc = conv.dim(1) * conv.dim(2) * conv.dim(3);
        auto conv_c_inc = conv.dim(2) * conv.dim(3);

        auto kernel_i_inc = kernel.dim(1) * kernel.dim(2) * kernel.dim(3);
        auto kernel_k_inc = kernel.dim(2) * kernel.dim(3);

        auto input_i_inc = input.dim(1) * input.dim(2) * input.dim(3);
        auto input_c_inc = input.dim(2) * input.dim(3);

        //i = 0
        for (std::size_t k = 0; k < kernel.dim(1); ++k) {
            for(std::size_t c = 0; c < input.dim(1); ++c){
                conv2_valid_flipped_micro_kernel(
                    input.memory_start() + 0 * input_i_inc + c * input_c_inc, input.dim(2), input.dim(3),
                    kernel.memory_start() + 0 * kernel_i_inc + k * kernel_k_inc, kernel.dim(2), kernel.dim(3),
                    conv.memory_start() + k * conv_k_inc + c * conv_c_inc, 0.0, s1, s2, p1, p2);
            }
        }

        for (std::size_t i = 1; i < input.dim(0); ++i) {
            for (std::size_t k = 0; k < kernel.dim(1); ++k) {
                for(std::size_t c = 0; c < input.dim(1); ++c){
                    conv2_valid_flipped_micro_kernel(
                        input.memory_start() + i * input_i_inc + c * input_c_inc, input.dim(2), input.dim(3),
                        kernel.memory_start() + i * kernel_i_inc + k * kernel_k_inc, kernel.dim(2), kernel.dim(3),
                        conv.memory_start() + k * conv_k_inc + c * conv_c_inc, 1.0, s1, s2, p1, p2);
                }
            }
        }
    }
}

template <typename T>
void conv4_full(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv) {
    if (kernel.dim(1) > 0) {
        auto conv_i_inc = conv.dim(1) * conv.dim(2) * conv.dim(3);
        auto conv_c_inc = conv.dim(2) * conv.dim(3);

        auto kernel_k_inc = kernel.dim(1) * kernel.dim(2) * kernel.dim(3);
        auto kernel_c_inc = kernel.dim(2) * kernel.dim(3);

        auto input_i_inc = input.dim(1) * input.dim(2) * input.dim(3);
        auto input_k_inc = input.dim(2) * input.dim(3);

        for (std::size_t i = 0; i < input.dim(0); ++i) {
            //k = 0
            for (std::size_t c = 0; c < kernel.dim(1); ++c) {
                conv2_full_micro_kernel(
                    input.memory_start() + i * input_i_inc + 0 * input_k_inc, input.dim(2), input.dim(3),
                    kernel.memory_start() + 0 * kernel_k_inc + c * kernel_c_inc, kernel.dim(2), kernel.dim(3),
                    conv.memory_start() + i * conv_i_inc + c * conv_c_inc, 0.0);
            }

            for (std::size_t k = 1; k < kernel.dim(0); ++k) {
                for (std::size_t c = 0; c < kernel.dim(1); ++c) {
                    conv2_full_micro_kernel(
                        input.memory_start() + i * input_i_inc + k * input_k_inc, input.dim(2), input.dim(3),
                        kernel.memory_start() + k * kernel_k_inc + c * kernel_c_inc, kernel.dim(2), kernel.dim(3),
                        conv.memory_start() + i * conv_i_inc + c * conv_c_inc, 1.0);
                }
            }
        }
    }
}

template <typename T>
void conv4_full_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv) {
    if (kernel.dim(1) > 0) {
        auto conv_i_inc = conv.dim(1) * conv.dim(2) * conv.dim(3);
        auto conv_c_inc = conv.dim(2) * conv.dim(3);

        auto kernel_k_inc = kernel.dim(1) * kernel.dim(2) * kernel.dim(3);
        auto kernel_c_inc = kernel.dim(2) * kernel.dim(3);

        auto input_i_inc = input.dim(1) * input.dim(2) * input.dim(3);
        auto input_k_inc = input.dim(2) * input.dim(3);

        for (std::size_t i = 0; i < input.dim(0); ++i) {
            //k = 0
            for (std::size_t c = 0; c < kernel.dim(1); ++c) {
                conv2_full_flipped_micro_kernel(
                    input.memory_start() + i * input_i_inc + 0 * input_k_inc, input.dim(2), input.dim(3),
                    kernel.memory_start() + 0 * kernel_k_inc + c * kernel_c_inc, kernel.dim(2), kernel.dim(3),
                    conv.memory_start() + i * conv_i_inc + c * conv_c_inc, 0.0);
            }

            for (std::size_t k = 1; k < kernel.dim(0); ++k) {
                for (std::size_t c = 0; c < kernel.dim(1); ++c) {
                    conv2_full_flipped_micro_kernel(
                        input.memory_start() + i * input_i_inc + k * input_k_inc, input.dim(2), input.dim(3),
                        kernel.memory_start() + k * kernel_k_inc + c * kernel_c_inc, kernel.dim(2), kernel.dim(3),
                        conv.memory_start() + i * conv_i_inc + c * conv_c_inc, 1.0);
                }
            }
        }
    }
}

template <typename T>
void conv2_same(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv) {
    conv2_same_micro_kernel(
        input.memory_start(), input.template dim<0>(), input.template dim<1>(),
        kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
        conv.memory_start());
}

template <typename T>
void conv2_same_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv) {
    conv2_same_flipped_micro_kernel(
        input.memory_start(), input.template dim<0>(), input.template dim<1>(),
        kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
        conv.memory_start());
}

template <typename T>
void conv2_same_multi(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv) {
    for(std::size_t k = 0; k < kernel.template dim<0>(); ++k){
        auto kk = kernel.template dim<1>() * kernel.template dim<2>();
        auto cc = conv.template dim<1>() * conv.template dim<2>();

        conv2_same_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
            conv.memory_start() + k * cc);
    }
}

template <typename T>
void conv2_same_multi_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv) {
    for(std::size_t k = 0; k < kernel.template dim<0>(); ++k){
        auto kk = kernel.template dim<1>() * kernel.template dim<2>();
        auto cc = conv.template dim<1>() * conv.template dim<2>();

        conv2_same_flipped_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
            conv.memory_start() + k * cc);
    }
}

template <typename T>
void conv2_full(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv) {
    conv2_full_micro_kernel(
        input.memory_start(), input.template dim<0>(), input.template dim<1>(),
        kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
        conv.memory_start(), 0.0);
}

template <typename T>
void conv2_full_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv) {
    conv2_full_flipped_micro_kernel(
        input.memory_start(), input.template dim<0>(), input.template dim<1>(),
        kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
        conv.memory_start(), 0.0);
}

template <typename T>
void conv2_full_multi(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv) {
    for(std::size_t k = 0; k < kernel.template dim<0>(); ++k){
        auto kk = kernel.template dim<1>() * kernel.template dim<2>();
        auto cc = conv.template dim<1>() * conv.template dim<2>();

        conv2_full_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
            conv.memory_start() + k * cc, 0.0);
    }
}

template <typename T>
void conv2_full_multi_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv) {
    for(std::size_t k = 0; k < kernel.template dim<0>(); ++k){
        auto kk = kernel.template dim<1>() * kernel.template dim<2>();
        auto cc = conv.template dim<1>() * conv.template dim<2>();

        conv2_full_flipped_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
            conv.memory_start() + k * cc, 0.0);
    }
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief AVX implementation of a 1D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_full(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(first);
    cpp_unused(last);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 1D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_same(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(first);
    cpp_unused(last);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 1D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_valid(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(first);
    cpp_unused(last);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_multi(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'full' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 2D 'full' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_valid(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv, size_t s1, size_t s2, size_t p1, size_t p2){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 4D 'valid' convolution C = I * K, with flipped weights
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_valid_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv, size_t s1, size_t s2, size_t p1, size_t p2){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 4D 'valid' convolution C = I * K, where the output are considered to be kernels
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_valid_filter(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv, size_t s1, size_t s2, size_t p1, size_t p2){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 4D 'valid' convolution C = I * K, where the output
 * are considered to be kernels, with flipped weights
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_valid_filter_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv, size_t s1, size_t s2, size_t p1, size_t p2){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_full(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

/*!
 * \brief AVX implementation of a 4D 'full' convolution C = I * K, with flipped weights
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_full_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("AVX not available/enabled");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace avx
} //end of namespace impl
} //end of namespace etl
