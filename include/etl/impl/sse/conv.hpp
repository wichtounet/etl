//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*
 * SSE implementation of 1D and 2D convolutions
 *
 * Ideas:
 *  * the tmp_res vectors could be avoided by using hadd instructions
 *  * 1D convolution with no memory allocation could probably be worked out (needs to be benchmarked)
 *
 *  Notes:
 *  * FMA for the 1D convolution is making is slower for some reason
 */

#if defined(ETL_VECTORIZE_IMPL) && defined(__SSE3__)
#include <immintrin.h>

#include "etl/impl/common/conv.hpp"

#endif

namespace etl {

namespace impl {

namespace sse {

#if defined(ETL_VECTORIZE_IMPL) && defined(__SSE3__)

#ifdef __clang__
#define _mm_undefined_ps _mm_setzero_ps
#endif

ETL_INLINE(double) mm_hadd_sd(__m128d in) {
    __m128 undef   = _mm_undefined_ps();
    __m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(in));
    __m128d shuf = _mm_castps_pd(shuftmp);
    return _mm_cvtsd_f64(_mm_add_sd(in, shuf));
}

ETL_INLINE(float) mm_hadd_ss(__m128 in) {
    __m128 shuf = _mm_movehdup_ps(in);
    __m128 sums = _mm_add_ps(in, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline void conv1_valid_micro_kernel(const double* in, const std::size_t n, const double* kernel, std::size_t m, double* out, std::size_t first, std::size_t last) {
    auto kernel_reverse = aligned_allocate_auto<__m128d>(m);

    //Reverse the kernel

    for (std::size_t i = 0; i < m; i++) {
        kernel_reverse[i] = _mm_load1_pd(kernel + m - i - 1);
    }

    //Compute the convolution, 2 doubles at a time

    auto llast = std::min(n - m + 1, last);

    for (std::size_t i = first; i + 1 < llast; i += 2) {
        __m128d r1 = _mm_setzero_pd();
        __m128d r2 = _mm_setzero_pd();
        __m128d r3 = _mm_setzero_pd();
        __m128d r4 = _mm_setzero_pd();

        for (std::size_t k = 0; k + 3 < m; k += 4) {
            __m128d t1 = _mm_loadu_pd(in + i + k);
            __m128d t2 = _mm_loadu_pd(in + i + k + 1);
            __m128d t3 = _mm_loadu_pd(in + i + k + 2);
            __m128d t4 = _mm_loadu_pd(in + i + k + 3);

            __m128d v1 = _mm_mul_pd(kernel_reverse[k], t1);
            r1         = _mm_add_pd(r1, v1);
            __m128d v2 = _mm_mul_pd(kernel_reverse[k + 1], t2);
            r2         = _mm_add_pd(r2, v2);
            __m128d v3 = _mm_mul_pd(kernel_reverse[k + 2], t3);
            r3         = _mm_add_pd(r3, v3);
            __m128d v4 = _mm_mul_pd(kernel_reverse[k + 3], t4);
            r4         = _mm_add_pd(r4, v4);
        }

        for (std::size_t k = m - m % 4; k < m; k++) {
            __m128d t1 = _mm_loadu_pd(in + i + k);
            __m128d v1 = _mm_mul_pd(kernel_reverse[k], t1);
            r1         = _mm_add_pd(r1, v1);
        }

        __m128d res = _mm_add_pd(r1, r2);
        res         = _mm_add_pd(res, r3);
        res         = _mm_add_pd(res, r4);

        _mm_storeu_pd(out + i, res);
    }

    //If the number of operations is not even, the last case must be
    //computed separatly

    std::size_t c = llast - first;
    if (c % 2 != 0) {
        auto i = llast - 1;
        out[i] = 0.0;
        for (std::size_t k = 0; k < m; k++) {
            out[i] += in[i + k] * kernel[m - k - 1];
        }
    }
}

inline void conv1_valid_micro_kernel(const float* in, const std::size_t n, const float* kernel, std::size_t m, float* out, std::size_t first, std::size_t last) {
    auto kernel_reverse = aligned_allocate_auto<__m128>(m);

    //Reverse the kernel

    for (std::size_t i = 0; i < m; i++) {
        kernel_reverse[i] = _mm_load1_ps(kernel + m - i - 1);
    }

    //Compute the convolution 4 floats at a time

    auto llast = std::min(n - m + 1, last);

    for (std::size_t i = first; i + 3 < llast; i += 4) {
        __m128 r1 = _mm_setzero_ps();
        __m128 r2 = _mm_setzero_ps();
        __m128 r3 = _mm_setzero_ps();
        __m128 r4 = _mm_setzero_ps();

        for (std::size_t k = 0; k + 3 < m; k += 4) {
            __m128 t1 = _mm_loadu_ps(in + i + k);
            __m128 t2 = _mm_loadu_ps(in + i + k + 1);
            __m128 t3 = _mm_loadu_ps(in + i + k + 2);
            __m128 t4 = _mm_loadu_ps(in + i + k + 3);

            __m128 v1 = _mm_mul_ps(kernel_reverse[k], t1);
            r1        = _mm_add_ps(r1, v1);
            __m128 v2 = _mm_mul_ps(kernel_reverse[k + 1], t2);
            r2        = _mm_add_ps(r2, v2);
            __m128 v3 = _mm_mul_ps(kernel_reverse[k + 2], t3);
            r3        = _mm_add_ps(r3, v3);
            __m128 v4 = _mm_mul_ps(kernel_reverse[k + 3], t4);
            r4        = _mm_add_ps(r4, v4);
        }

        for (std::size_t k = m - m % 4; k < m; k++) {
            __m128 t1 = _mm_loadu_ps(in + i + k);
            __m128 v1 = _mm_mul_ps(kernel_reverse[k], t1);
            r1        = _mm_add_ps(r1, v1);
        }

        __m128 res = _mm_add_ps(r1, r2);
        res        = _mm_add_ps(res, r3);
        res        = _mm_add_ps(res, r4);

        _mm_storeu_ps(out + i, res);
    }

    //Complete the last outputs which are not vectorized

    auto c = llast - first;

    if (c % 4 != 0) {
        auto rem = c % 4;
        for (std::size_t i = llast - rem; i < llast; ++i) {
            out[i] = 0.0;
            for (std::size_t k = 0; k < m; k++) {
                out[i] += in[i + k] * kernel[m - k - 1];
            }
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
            for (std::size_t j = p2; j + 3 < c2 - p2; j += 4) {
                __m128d r1 = _mm_setzero_pd();
                __m128d r2 = _mm_setzero_pd();
                __m128d r3 = _mm_setzero_pd();
                __m128d r4 = _mm_setzero_pd();

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 1 < m2; l += 2) {
                        __m128d k1 = _mm_loadu_pd(kernel + k * m2 + l);

                        __m128d i1 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + 0 + l);
                        __m128d i2 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + 1 + l);
                        __m128d i3 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + 2 + l);
                        __m128d i4 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + 3 + l);

                        __m128d t1 = _mm_mul_pd(k1, i1);
                        __m128d t2 = _mm_mul_pd(k1, i2);
                        __m128d t3 = _mm_mul_pd(k1, i3);
                        __m128d t4 = _mm_mul_pd(k1, i4);

                        r1  = _mm_add_pd(r1, t1);
                        r2  = _mm_add_pd(r2, t2);
                        r3  = _mm_add_pd(r3, t3);
                        r4  = _mm_add_pd(r4, t4);
                    }
                }

                out[i * c2 + j + 0] = mm_hadd_sd(r1);
                out[i * c2 + j + 1] = mm_hadd_sd(r2);
                out[i * c2 + j + 2] = mm_hadd_sd(r3);
                out[i * c2 + j + 3] = mm_hadd_sd(r4);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 4; j < c2 - p2; ++j) {
                __m128d r1 = _mm_setzero_pd();

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 1 < m2; l += 2) {
                        __m128d tmp1 = _mm_loadu_pd(in + (i_i + k -  p1) * n2 + i_j - p2 + l);
                        __m128d tmp2 = _mm_loadu_pd(kernel + k * m2 + l);
                        __m128d tmp4 = _mm_mul_pd(tmp2, tmp1);
                        r1  = _mm_add_pd(r1, tmp4);
                    }
                }

                out[i * c2 + j] = mm_hadd_sd(r1);
            }
        }
    } else {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 3 < c2 - p2; j += 4) {
                __m128d r1 = _mm_setzero_pd();
                __m128d r2 = _mm_setzero_pd();
                __m128d r3 = _mm_setzero_pd();
                __m128d r4 = _mm_setzero_pd();

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 1 < m2; l += 2) {
                        __m128d k1 = _mm_loadu_pd(kernel + k * m2 + l);

                        __m128d i1 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + 0 + l);
                        __m128d i2 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + 1 + l);
                        __m128d i3 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + 2 + l);
                        __m128d i4 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + 3 + l);

                        __m128d t1 = _mm_mul_pd(k1, i1);
                        __m128d t2 = _mm_mul_pd(k1, i2);
                        __m128d t3 = _mm_mul_pd(k1, i3);
                        __m128d t4 = _mm_mul_pd(k1, i4);

                        r1  = _mm_add_pd(r1, t1);
                        r2  = _mm_add_pd(r2, t2);
                        r3  = _mm_add_pd(r3, t3);
                        r4  = _mm_add_pd(r4, t4);
                    }
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + mm_hadd_sd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + mm_hadd_sd(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + mm_hadd_sd(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + mm_hadd_sd(r4);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 4; j < c2 - p2; ++j) {
                __m128d r1 = _mm_setzero_pd();

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 1 < m2; l += 2) {
                        __m128d tmp1 = _mm_loadu_pd(in + (i_i + k - p1) * n2 + i_j - p2 + l);
                        __m128d tmp2 = _mm_loadu_pd(kernel + k * m2 + l);
                        __m128d tmp4 = _mm_mul_pd(tmp2, tmp1);
                        r1  = _mm_add_pd(r1, tmp4);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm_hadd_sd(r1);
            }
        }
    }

    if (m2 % 2 != 0) {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j < c2 - p2; ++j) {
                double temp = 0.0;

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    const auto l = m2 - 1;
                    temp += in[(i_i - p1 + k) * n2 + i_j - p2 + l] * kernel[k * m2 + l];
                }

                out[i * c2 + j] += temp;
            }
        }
    }
}

inline void conv2_valid_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out, double beta, size_t s1, size_t s2, size_t p1, size_t p2) {
    auto kernel_reverse = aligned_allocate_auto<double>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_valid_flipped_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out, beta, s1, s2, p1, p2);
}

inline void conv2_same_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out) {
    std::size_t c1 = n1;
    std::size_t c2 = n2;

    for (std::size_t i = 0; i < c1; ++i) {
        std::size_t k_lo = std::max<int>(0, i - (m1 - 1) / 2);
        std::size_t k_hi = std::min<int>(n1 - 1, i + m1 / 2) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            std::size_t l_lo = std::max<int>(0, j - (m2 - 1) / 2);
            std::size_t l_hi = std::min<int>(n2 - 1, j + m2 / 2) + 1;

            __m128d r1 = _mm_setzero_pd();

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 1 < l_hi; l += 2) {
                    __m128d i1 = _mm_loadu_pd(in + k * n2 + l);

                    __m128d t2 = _mm_loadu_pd(kernel + (i - k + m1 / 2) * m2 + (j - (l + 1) + m2 / 2));
                    __m128d k1 = _mm_shuffle_pd(t2, t2, _MM_SHUFFLE2(0, 1));

                    __m128d t1 = _mm_mul_pd(k1, i1);
                    r1  = _mm_add_pd(r1, t1);
                }
            }

            out[i * c2 + j] = mm_hadd_sd(r1);

            double temp = 0.0;

            if ((l_hi - l_lo) % 2 != 0) {
                auto rem = (l_hi - l_lo) % 2;
                auto l = l_hi - rem;
                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    temp += in[k * n2 + l] * kernel[(i - k + m1 / 2) * m2 + (j - l + m2 / 2)];
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
    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    if(beta == 0.0){
        for (std::size_t i = 0; i < c1; ++i) {
            auto k_lo = std::max<int>(0, i - m1 + 1);
            auto k_hi = std::min(n1 - 1, i) + 1;

            for (std::size_t j = 0; j < c2; ++j) {
                auto l_lo = std::max<int>(0, j - m2 + 1);
                auto l_hi = std::min(n2 - 1, j) + 1;

                __m128d r1 = _mm_setzero_pd();

                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_lo; l + 1 < l_hi; l += 2) {
                        __m128d i1 = _mm_loadu_pd(in + k * n2 + l);
                        __m128d t2 = _mm_loadu_pd(kernel + (i - k) * m2 + (j - (l + 1)));
                        __m128d k1 = _mm_shuffle_pd(t2, t2, _MM_SHUFFLE2(0, 1));
                        __m128d t1 = _mm_mul_pd(k1, i1);
                        r1  = _mm_add_pd(r1, t1);
                    }
                }

                out[i * c2 + j] = mm_hadd_sd(r1);

                double temp = 0.0;

                if ((l_hi - l_lo) % 2 != 0) {
                    for (std::size_t k = k_lo; k < k_hi; ++k) {
                        temp += in[k * n2 + l_hi - 1] * kernel[(i - k) * m2 + (j - (l_hi - 1))];
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

                __m128d r1 = _mm_setzero_pd();

                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_lo; l + 1 < l_hi; l += 2) {
                        __m128d i1 = _mm_loadu_pd(in + k * n2 + l);
                        __m128d t2 = _mm_loadu_pd(kernel + (i - k) * m2 + (j - (l + 1)));
                        __m128d k1 = _mm_shuffle_pd(t2, t2, _MM_SHUFFLE2(0, 1));
                        __m128d t1 = _mm_mul_pd(k1, i1);
                        r1  = _mm_add_pd(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm_hadd_sd(r1);

                double temp = 0.0;

                if ((l_hi - l_lo) % 2 != 0) {
                    for (std::size_t k = k_lo; k < k_hi; ++k) {
                        temp += in[k * n2 + l_hi - 1] * kernel[(i - k) * m2 + (j - (l_hi - 1))];
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    }
}

inline void conv2_full_flipped_micro_kernel(const double* in, std::size_t n1, std::size_t n2, const double* kernel, std::size_t m1, std::size_t m2, double* out, double beta) {
    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    if (beta == 0.0) {
        for (std::size_t i = 0; i < c1; ++i) {
            auto k_lo = std::max<int>(0, i - m1 + 1);
            auto k_hi = std::min(n1 - 1, i) + 1;

            for (std::size_t j = 0; j < c2; ++j) {
                auto l_lo = std::max<int>(0, j - m2 + 1);
                auto l_hi = std::min(n2 - 1, j) + 1;

                __m128d r1 = _mm_setzero_pd();

                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_lo; l + 1 < l_hi; l += 2) {
                        __m128d i1 = _mm_loadu_pd(in + k * n2 + l);
                        __m128d k1 = _mm_loadu_pd(kernel + (m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l)));
                        __m128d t1 = _mm_mul_pd(k1, i1);
                        r1         = _mm_add_pd(r1, t1);
                    }
                }

                out[i * c2 + j] = mm_hadd_sd(r1);

                double temp = 0.0;

                if ((l_hi - l_lo) % 2 != 0) {
                    for (std::size_t k = k_lo; k < k_hi; ++k) {
                        temp += in[k * n2 + l_hi - 1] * kernel[(m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - (l_hi - 1)))];
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

                __m128d r1 = _mm_setzero_pd();

                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_lo; l + 1 < l_hi; l += 2) {
                        __m128d i1 = _mm_loadu_pd(in + k * n2 + l);
                        __m128d k1 = _mm_loadu_pd(kernel + (m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l)));
                        __m128d t1 = _mm_mul_pd(k1, i1);
                        r1         = _mm_add_pd(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm_hadd_sd(r1);

                double temp = 0.0;

                if ((l_hi - l_lo) % 2 != 0) {
                    for (std::size_t k = k_lo; k < k_hi; ++k) {
                        temp += in[k * n2 + l_hi - 1] * kernel[(m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - (l_hi - 1)))];
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

    if(beta == 0.0f){
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 3 < c2 - p2; j += 4) {
                __m128 r1 = _mm_setzero_ps();
                __m128 r2 = _mm_setzero_ps();
                __m128 r3 = _mm_setzero_ps();
                __m128 r4 = _mm_setzero_ps();

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m128 k1 = _mm_loadu_ps(kernel + k * m2 + l);

                        __m128 i1 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2 + 0);
                        __m128 i2 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2 + 1);
                        __m128 i3 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2 + 2);
                        __m128 i4 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2 + 3);

                        __m128 t1 = _mm_mul_ps(k1, i1);
                        __m128 t2 = _mm_mul_ps(k1, i2);
                        __m128 t3 = _mm_mul_ps(k1, i3);
                        __m128 t4 = _mm_mul_ps(k1, i4);

                        r1  = _mm_add_ps(r1, t1);
                        r2  = _mm_add_ps(r2, t2);
                        r3  = _mm_add_ps(r3, t3);
                        r4  = _mm_add_ps(r4, t4);
                    }
                }

                out[i * c2 + j + 0] = mm_hadd_ss(r1);
                out[i * c2 + j + 1] = mm_hadd_ss(r2);
                out[i * c2 + j + 2] = mm_hadd_ss(r3);
                out[i * c2 + j + 3] = mm_hadd_ss(r4);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 4; j < c2 - p2; ++j) {
                __m128 r1 = _mm_setzero_ps();

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m128 k1 = _mm_loadu_ps(kernel + k * m2 + l);

                        __m128 i1 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2);

                        __m128 t1 = _mm_mul_ps(k1, i1);

                        r1  = _mm_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = mm_hadd_ss(r1);
            }
        }
    } else {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 3 < c2 - p2; j += 4) {
                __m128 r1 = _mm_setzero_ps();
                __m128 r2 = _mm_setzero_ps();
                __m128 r3 = _mm_setzero_ps();
                __m128 r4 = _mm_setzero_ps();

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m128 k1 = _mm_loadu_ps(kernel + k * m2 + l);

                        __m128 i1 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2 + 0);
                        __m128 i2 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2 + 1);
                        __m128 i3 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2 + 2);
                        __m128 i4 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2 + 3);

                        __m128 t1 = _mm_mul_ps(k1, i1);
                        __m128 t2 = _mm_mul_ps(k1, i2);
                        __m128 t3 = _mm_mul_ps(k1, i3);
                        __m128 t4 = _mm_mul_ps(k1, i4);

                        r1  = _mm_add_ps(r1, t1);
                        r2  = _mm_add_ps(r2, t2);
                        r3  = _mm_add_ps(r3, t3);
                        r4  = _mm_add_ps(r4, t4);
                    }
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + mm_hadd_ss(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + mm_hadd_ss(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + mm_hadd_ss(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + mm_hadd_ss(r4);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 4; j < c2 - p2; ++j) {
                __m128 r1 = _mm_setzero_ps();

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m128 k1 = _mm_loadu_ps(kernel + k * m2 + l);

                        __m128 i1 = _mm_loadu_ps(in + (k + i_i - p1) * n2 + l + i_j - p2);

                        __m128 t1 = _mm_mul_ps(k1, i1);

                        r1  = _mm_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm_hadd_ss(r1);
            }
        }
    }

    if (m2 % 4 != 0) {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j < c2 - p2; ++j) {
                float temp = 0.0;

                const auto i_i = i * s1;
                const auto i_j = j * s2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = m2 - m2 % 4; l < m2; ++l) {
                        temp += in[(k + i_i - p1) * n2 + l + i_j - p2] * kernel[k * m2 + l];
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    }
}

inline void conv2_valid_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out, float beta, size_t s1, size_t s2, size_t p1, size_t p2) {
    auto kernel_reverse = aligned_allocate_auto<float>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_valid_flipped_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out, beta, s1, s2, p1, p2);
}

inline void conv2_same_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out) {
    std::size_t c1 = n1;
    std::size_t c2 = n2;

    for (std::size_t i = 0; i < c1; ++i) {
        auto k_lo = std::max<int>(0, i - (m1 - 1) / 2);
        auto k_hi = std::min<int>(n1 - 1, i + m1 / 2) + 1;

        for (std::size_t j = 0; j < c2; ++j) {
            auto l_lo = std::max<int>(0, j - (m2 - 1) / 2);
            auto l_hi = std::min<int>(n2 - 1, j + m2 / 2) + 1;

            __m128 r1 = _mm_setzero_ps();

            for (int k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l + 3 < static_cast<std::size_t>(l_hi); l += 4) {
                    __m128 i1 = _mm_loadu_ps(in + k * n2 + l);
                    __m128 t2 = _mm_loadu_ps(kernel + (i - k + m1 / 2) * m2 + (j - (l + 3) + m2 / 2));
                    __m128 k1 = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(0, 1, 2, 3));

                    __m128 t1 = _mm_mul_ps(k1, i1);
                    r1  = _mm_add_ps(r1, t1);
                }
            }

            out[i * c2 + j] = mm_hadd_ss(r1);

            float temp = 0.0;

            if ((l_hi - l_lo) % 4 != 0) {
                auto rem = (l_hi - l_lo) % 4;
                for (int k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_hi - rem; l < static_cast<std::size_t>(l_hi); ++l) {
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
    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    if(beta == 0.0){
        for (std::size_t i = 0; i < c1; ++i) {
            auto k_lo = std::max<int>(0, i - m1 + 1);
            auto k_hi = std::min(n1 - 1, i) + 1;

            for (std::size_t j = 0; j < c2; ++j) {
                auto l_lo = std::max<int>(0, j - m2 + 1);
                auto l_hi = std::min(n2 - 1, j) + 1;

                __m128 r1 = _mm_setzero_ps();

                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_lo; l + 3 < l_hi; l += 4) {
                        __m128 i1 = _mm_loadu_ps(in + k * n2 + l);
                        __m128 t2 = _mm_loadu_ps(kernel + (i - k) * m2 + (j - (l + 3)));
                        __m128 k1 = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(0, 1, 2, 3));
                        __m128 t1 = _mm_mul_ps(k1, i1);
                        r1        = _mm_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = mm_hadd_ss(r1);

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

                __m128 r1 = _mm_setzero_ps();

                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_lo; l + 3 < l_hi; l += 4) {
                        __m128 i1 = _mm_loadu_ps(in + k * n2 + l);
                        __m128 t2 = _mm_loadu_ps(kernel + (i - k) * m2 + (j - (l + 3)));
                        __m128 k1 = _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(0, 1, 2, 3));
                        __m128 t1 = _mm_mul_ps(k1, i1);
                        r1        = _mm_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm_hadd_ss(r1);

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

inline void conv2_full_flipped_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out, float beta) {
    std::size_t c1 = n1 + m1 - 1;
    std::size_t c2 = n2 + m2 - 1;

    if(beta == 0.0){
        for (std::size_t i = 0; i < c1; ++i) {
            auto k_lo = std::max<int>(0, i - m1 + 1);
            auto k_hi = std::min(n1 - 1, i) + 1;

            for (std::size_t j = 0; j < c2; ++j) {
                auto l_lo = std::max<int>(0, j - m2 + 1);
                auto l_hi = std::min(n2 - 1, j) + 1;

                __m128 r1 = _mm_setzero_ps();

                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_lo; l + 3 < l_hi; l += 4) {
                        __m128 i1 = _mm_loadu_ps(in + k * n2 + l);
                        __m128 k1 = _mm_loadu_ps(kernel + (m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l)));
                        __m128 t1 = _mm_mul_ps(k1, i1);
                        r1        = _mm_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = mm_hadd_ss(r1);

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

                __m128 r1 = _mm_setzero_ps();

                for (std::size_t k = k_lo; k < k_hi; ++k) {
                    for (std::size_t l = l_lo; l + 3 < l_hi; l += 4) {
                        __m128 i1 = _mm_loadu_ps(in + k * n2 + l);
                        __m128 k1 = _mm_loadu_ps(kernel + (m1 - 1 - (i - k)) * m2 + (m2 - 1 - (j - l)));
                        __m128 t1 = _mm_mul_ps(k1, i1);
                        r1        = _mm_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + mm_hadd_ss(r1);

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

template <typename T>
void conv2_valid(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
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
                //c = 0
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
void conv4_valid_filter_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv) {
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
                    conv.memory_start() + k * conv_k_inc + c * conv_c_inc, 0.0, 1, 1, 0, 0);
            }
        }

        for (std::size_t i = 1; i < input.dim(0); ++i) {
            for (std::size_t k = 0; k < kernel.dim(1); ++k) {
                for(std::size_t c = 0; c < input.dim(1); ++c){
                    conv2_valid_flipped_micro_kernel(
                        input.memory_start() + i * input_i_inc + c * input_c_inc, input.dim(2), input.dim(3),
                        kernel.memory_start() + i * kernel_i_inc + k * kernel_k_inc, kernel.dim(2), kernel.dim(3),
                        conv.memory_start() + k * conv_k_inc + c * conv_c_inc, 1.0, 1, 1, 0, 0);
                }
            }
        }
    }
}

template <typename T>
void conv4_valid_filter(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv) {
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
                    conv.memory_start() + k * conv_k_inc + c * conv_c_inc, 0.0, 1, 1, 0, 0);
            }
        }

        for (std::size_t i = 1; i < input.dim(0); ++i) {
            for (std::size_t k = 0; k < kernel.dim(1); ++k) {
                for(std::size_t c = 0; c < input.dim(1); ++c){
                    conv2_valid_micro_kernel(
                        input.memory_start() + i * input_i_inc + c * input_c_inc, input.dim(2), input.dim(3),
                        kernel.memory_start() + i * kernel_i_inc + k * kernel_k_inc, kernel.dim(2), kernel.dim(3),
                        conv.memory_start() + k * conv_k_inc + c * conv_c_inc, 1.0, 1, 1, 0, 0);
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

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief SSE implementation of a 1D 'full' convolution C = I * K
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
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 1D 'same' convolution C = I * K
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
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 1D 'valid' convolution C = I * K
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
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'valid' convolution C = I * K
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
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'valid' convolution C = I * K
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
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid_multi(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename K, typename C>
void conv2_valid_multi_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_multi(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K, with multiple
 * kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K, with multiple
 * flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K
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
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K, with flipped weights
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
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K, where the output are considered to be kernels
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_valid_filter(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K, where the output
 * are considered to be kernels, with flipped weights
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_valid_filter_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_full(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

/*!
 * \brief SSE implementation of a 4D 'full' convolution C = I * K,
 * with flipped weights
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename T>
void conv4_full_flipped(const opaque_memory<T, 4>& input, const opaque_memory<T, 4>& kernel, const opaque_memory<T, 4>& conv){
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);
    cpp_unreachable("SSE not available/enabled");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace sse
} //end of namespace impl
} //end of namespace etl
