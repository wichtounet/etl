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

#include "common.hpp"
#include "etl/impl/common/conv.hpp"

#endif

namespace etl {

namespace impl {

namespace avx {

#if defined(ETL_VECTORIZE_IMPL) && defined(__AVX__)

namespace detail {

template<typename T>
constexpr bool prefer_sse(const size_t n){
    return
          std::is_same<T, float>::value
        ? (n % 4 < n % 8)
        : (n % 2 < n % 4);
}

template <typename T>
void pad_2d_input(const opaque_memory<T, 2>& in, opaque_memory<T, 2>& out, size_t p1, size_t p2) {
    auto in_m = in.memory_start();
    auto out_m = out.memory_start();

    for (std::size_t i = 0; i < in.template dim<0>(); ++i) {
        direct_copy_n(in_m + i * in.template dim<1>(), out_m + (i + p1) * out.template dim<1>() + p2, in.template dim<1>());
    }
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

    if (!padding_impl && m2 % 4 != 0) {
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
    auto kernel_reverse = aligned_allocate_auto<double>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_valid_flipped_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out, beta, s1, s2, p1, p2);
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

    if(cpp_unlikely(p1 || p2)){
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
            std::size_t j = p2;
            for (; j + 7 < c2 - p2; j += 8) {
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

            for (; j < c2 - p2; ++j) {
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
            std::size_t j = p2;

            for (; j + 7 < c2 - p2; j += 8) {
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

            for (; j < c2 - p2; ++j) {
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

    if (!padding_impl && m2 % 8 != 0) {
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

    auto kernel_reverse = aligned_allocate_auto<float>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_valid_flipped_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out, beta, s1, s2, p1, p2);
}

} // end of namespace detail

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
template <typename T>
void conv2_valid(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(cpp_unlikely(p1 || p2)){
        const auto ws_h = input.template dim<0>() + 2 * p1;
        const auto ws_w = input.template dim<1>() + 2 * p2;

        if(ws_h * ws_w * sizeof(T) < max_workspace){
            etl::dyn_matrix<T, 2> workspace(ws_h, ws_w, T(0));
            auto ws_direct = workspace.direct();

            detail::pad_2d_input(input, ws_direct, p1, p2);

            conv2_valid(workspace.direct(), kernel, conv, s1, s2, 0, 0);

            return;
        }
    }

    const auto k2 = kernel.dim(1);

    if(padding_impl){
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if(k2 < SS || k2 % AS > 0){
            const auto pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right_flip(kernel, pad);

            if(detail::prefer_sse<T>(k2 + pad)){
                etl::impl::sse::conv2_valid_flipped_micro_kernel(
                    padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                    padded_kernel.memory_start(), padded_kernel.template dim<0>(), padded_kernel.template dim<1>(),
                    conv.memory_start(), 0.0, s1, s2, p1, p2);
            } else {
                detail::conv2_valid_flipped_micro_kernel(
                    padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                    padded_kernel.memory_start(), padded_kernel.template dim<0>(), padded_kernel.template dim<1>(),
                    conv.memory_start(), 0.0, s1, s2, p1, p2);
            }

            return;
        }
    }

    if(detail::prefer_sse<T>(k2)){
        etl::impl::sse::conv2_valid_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
            conv.memory_start(), 0.0, s1, s2, p1, p2);
    } else {
        detail::conv2_valid_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
            conv.memory_start(), 0.0, s1, s2, p1, p2);
    }
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
template <typename T>
void conv2_valid_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto k2 = kernel.dim(1);

    if(padding_impl){
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if(k2 < SS || k2 % AS > 0){
            const auto pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right(kernel, pad);

            if(detail::prefer_sse<T>(k2 + pad)){
                etl::impl::sse::conv2_valid_flipped_micro_kernel(
                    padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                    padded_kernel.memory_start(), padded_kernel.template dim<0>(), padded_kernel.template dim<1>(),
                    conv.memory_start(), 0.0, s1, s2, p1, p2);
            } else {
                detail::conv2_valid_flipped_micro_kernel(
                    padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                    padded_kernel.memory_start(), padded_kernel.template dim<0>(), padded_kernel.template dim<1>(),
                    conv.memory_start(), 0.0, s1, s2, p1, p2);
            }

            return;
        }
    }

    if(detail::prefer_sse<T>(k2)){
        etl::impl::sse::conv2_valid_flipped_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
            conv.memory_start(), 0.0, s1, s2, p1, p2);
    } else {
        detail::conv2_valid_flipped_micro_kernel(
            input.memory_start(), input.template dim<0>(), input.template dim<1>(),
            kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
            conv.memory_start(), 0.0, s1, s2, p1, p2);
    }
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
template <typename T>
void conv2_valid_multi(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto K = kernel.template dim<0>();
    const auto k2 = kernel.dim(2);

    if(padding_impl){
        static constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        static constexpr size_t SS = AS / 2;

        if(k2 < SS || k2 % AS > 0){
            const auto pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

            // TODO Test if it is better to do the padding of the kernel inside each thread

            if(detail::prefer_sse<T>(k2 + pad)){
                auto fun_k = [&](const size_t first, const size_t last) {
                    for (std::size_t k = first; k < last; ++k) {
                        auto kk = padded_kernel.template dim<1>() * padded_kernel.template dim<2>();
                        auto cc = conv.template dim<1>() * conv.template dim<2>();

                        impl::sse::conv2_valid_flipped_micro_kernel(
                            padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                            padded_kernel.memory_start() + k * kk, padded_kernel.template dim<1>(), padded_kernel.template dim<2>(),
                            conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
                    }
                };

                dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
            } else {
                auto fun_k = [&](const size_t first, const size_t last) {
                    for (std::size_t k = first; k < last; ++k) {
                        auto kk = padded_kernel.template dim<1>() * padded_kernel.template dim<2>();
                        auto cc = conv.template dim<1>() * conv.template dim<2>();

                        detail::conv2_valid_flipped_micro_kernel(
                            padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                            padded_kernel.memory_start() + k * kk, padded_kernel.template dim<1>(), padded_kernel.template dim<2>(),
                            conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
                    }
                };

                dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
            }

            return;
        }
    }

    if(detail::prefer_sse<T>(kernel.dim(2))){
        auto fun_k = [&](const size_t first, const size_t last) {
            for (std::size_t k = first; k < last; ++k) {
                auto kk = kernel.template dim<1>() * kernel.template dim<2>();
                auto cc = conv.template dim<1>() * conv.template dim<2>();

                impl::sse::conv2_valid_micro_kernel(
                    input.memory_start(), input.template dim<0>(), input.template dim<1>(),
                    kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
                    conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
            }
        };

        dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
    } else {
        auto fun_k = [&](const size_t first, const size_t last) {
            for (std::size_t k = first; k < last; ++k) {
                auto kk = kernel.template dim<1>() * kernel.template dim<2>();
                auto cc = conv.template dim<1>() * conv.template dim<2>();

                detail::conv2_valid_micro_kernel(
                    input.memory_start(), input.template dim<0>(), input.template dim<1>(),
                    kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
                    conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
            }
        };

        dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
    }
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
template <typename T>
void conv2_valid_multi_flipped(const opaque_memory<T, 2>& input, const opaque_memory<T, 3>& kernel, const opaque_memory<T, 3>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto K = kernel.template dim<0>();
    const auto k2 = kernel.dim(2);

    if(padding_impl){
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if(k2 < SS || k2 % AS > 0){
            const auto pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right_multi(kernel, pad);

            // TODO Test if it is better to do the padding of the kernel inside each thread

            if(detail::prefer_sse<T>(k2 + pad)){
                auto fun_k = [&](const size_t first, const size_t last) {
                    for (std::size_t k = first; k < last; ++k) {
                        auto kk = padded_kernel.template dim<1>() * padded_kernel.template dim<2>();
                        auto cc = conv.template dim<1>() * conv.template dim<2>();

                        impl::sse::conv2_valid_flipped_micro_kernel(
                            padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                            padded_kernel.memory_start() + k * kk, padded_kernel.template dim<1>(), padded_kernel.template dim<2>(),
                            conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
                    }
                };

                dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
            } else {
                auto fun_k = [&](const size_t first, const size_t last) {
                    for (std::size_t k = first; k < last; ++k) {
                        auto kk = padded_kernel.template dim<1>() * padded_kernel.template dim<2>();
                        auto cc = conv.template dim<1>() * conv.template dim<2>();

                        detail::conv2_valid_flipped_micro_kernel(
                            padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                            padded_kernel.memory_start() + k * kk, padded_kernel.template dim<1>(), padded_kernel.template dim<2>(),
                            conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
                    }
                };

                dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
            }

            return;
        }
    }

    if(detail::prefer_sse<T>(k2)){
        auto fun_k = [&](const size_t first, const size_t last) {
            for (std::size_t k = first; k < last; ++k) {
                auto kk = kernel.template dim<1>() * kernel.template dim<2>();
                auto cc = conv.template dim<1>() * conv.template dim<2>();

                impl::sse::conv2_valid_flipped_micro_kernel(
                    input.memory_start(), input.template dim<0>(), input.template dim<1>(),
                    kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
                    conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
            }
        };

        dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
    } else {
        auto fun_k = [&](const size_t first, const size_t last) {
            for (std::size_t k = first; k < last; ++k) {
                auto kk = kernel.template dim<1>() * kernel.template dim<2>();
                auto cc = conv.template dim<1>() * conv.template dim<2>();

                detail::conv2_valid_flipped_micro_kernel(
                    input.memory_start(), input.template dim<0>(), input.template dim<1>(),
                    kernel.memory_start() + k * kk, kernel.template dim<1>(), kernel.template dim<2>(),
                    conv.memory_start() + k * cc, 0.0, s1, s2, p1, p2);
            }
        };

        dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
    }
}

#else

//COVERAGE_EXCLUDE_BEGIN

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
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
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
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("AVX not available/enabled");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace avx
} //end of namespace impl
} //end of namespace etl
