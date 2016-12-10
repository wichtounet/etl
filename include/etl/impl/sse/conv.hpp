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

#include "common.hpp"
#include "etl/impl/common/conv.hpp"

#endif

namespace etl {

namespace impl {

namespace sse {

#if defined(ETL_VECTORIZE_IMPL) && defined(__SSE3__)

#ifdef __clang__
#define _mm_undefined_ps _mm_setzero_ps
#endif

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

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 1 < m2; l += 2) {
                        __m128d k1 = _mm_loadu_pd(kernel + k * m2 + l);

                        __m128d i1 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + 0 + l);
                        __m128d i2 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + 1 + l);
                        __m128d i3 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + 2 + l);
                        __m128d i4 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + 3 + l);

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

                out[i * c2 + j + 0] = detail::mm_hadd_sd(r1);
                out[i * c2 + j + 1] = detail::mm_hadd_sd(r2);
                out[i * c2 + j + 2] = detail::mm_hadd_sd(r3);
                out[i * c2 + j + 3] = detail::mm_hadd_sd(r4);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 4; j < c2 - p2; ++j) {
                __m128d r1 = _mm_setzero_pd();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 1 < m2; l += 2) {
                        __m128d tmp1 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + l);
                        __m128d tmp2 = _mm_loadu_pd(kernel + k * m2 + l);
                        __m128d tmp4 = _mm_mul_pd(tmp2, tmp1);
                        r1  = _mm_add_pd(r1, tmp4);
                    }
                }

                out[i * c2 + j] = detail::mm_hadd_sd(r1);
            }
        }
    } else {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 3 < c2 - p2; j += 4) {
                __m128d r1 = _mm_setzero_pd();
                __m128d r2 = _mm_setzero_pd();
                __m128d r3 = _mm_setzero_pd();
                __m128d r4 = _mm_setzero_pd();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 1 < m2; l += 2) {
                        __m128d k1 = _mm_loadu_pd(kernel + k * m2 + l);

                        __m128d i1 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + 0 + l);
                        __m128d i2 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + 1 + l);
                        __m128d i3 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + 2 + l);
                        __m128d i4 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + 3 + l);

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

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + detail::mm_hadd_sd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + detail::mm_hadd_sd(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + detail::mm_hadd_sd(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + detail::mm_hadd_sd(r4);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 4; j < c2 - p2; ++j) {
                __m128d r1 = _mm_setzero_pd();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 1 < m2; l += 2) {
                        __m128d tmp1 = _mm_loadu_pd(in + (i_i + k) * n2 + i_j + l);
                        __m128d tmp2 = _mm_loadu_pd(kernel + k * m2 + l);
                        __m128d tmp4 = _mm_mul_pd(tmp2, tmp1);
                        r1  = _mm_add_pd(r1, tmp4);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + detail::mm_hadd_sd(r1);
            }
        }
    }

    if (!padding_impl && m2 % 2 != 0) {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j < c2 - p2; ++j) {
                double temp = 0.0;

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    const auto l = m2 - 1;
                    temp += in[(i_i + k) * n2 + i_j + l] * kernel[k * m2 + l];
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

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m128 k1 = _mm_loadu_ps(kernel + k * m2 + l);

                        __m128 i1 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j + 0);
                        __m128 i2 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j + 1);
                        __m128 i3 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j + 2);
                        __m128 i4 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j + 3);

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

                out[i * c2 + j + 0] = detail::mm_hadd_ss(r1);
                out[i * c2 + j + 1] = detail::mm_hadd_ss(r2);
                out[i * c2 + j + 2] = detail::mm_hadd_ss(r3);
                out[i * c2 + j + 3] = detail::mm_hadd_ss(r4);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 4; j < c2 - p2; ++j) {
                __m128 r1 = _mm_setzero_ps();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m128 k1 = _mm_loadu_ps(kernel + k * m2 + l);

                        __m128 i1 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j);

                        __m128 t1 = _mm_mul_ps(k1, i1);

                        r1  = _mm_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = detail::mm_hadd_ss(r1);
            }
        }
    } else {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j + 3 < c2 - p2; j += 4) {
                __m128 r1 = _mm_setzero_ps();
                __m128 r2 = _mm_setzero_ps();
                __m128 r3 = _mm_setzero_ps();
                __m128 r4 = _mm_setzero_ps();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m128 k1 = _mm_loadu_ps(kernel + k * m2 + l);

                        __m128 i1 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j + 0);
                        __m128 i2 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j + 1);
                        __m128 i3 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j + 2);
                        __m128 i4 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j + 3);

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

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + detail::mm_hadd_ss(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + detail::mm_hadd_ss(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + detail::mm_hadd_ss(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + detail::mm_hadd_ss(r4);
            }

            for (std::size_t j = (c2 - p2) - (c2 - 2 * p2) % 4; j < c2 - p2; ++j) {
                __m128 r1 = _mm_setzero_ps();

                const auto i_i = i * s1 - p1;
                const auto i_j = j * s2 - p2;

                for (std::size_t k = 0; k < m1; ++k) {
                    for (std::size_t l = 0; l + 3 < m2; l += 4) {
                        __m128 k1 = _mm_loadu_ps(kernel + k * m2 + l);

                        __m128 i1 = _mm_loadu_ps(in + (k + i_i) * n2 + l + i_j);

                        __m128 t1 = _mm_mul_ps(k1, i1);

                        r1  = _mm_add_ps(r1, t1);
                    }
                }

                out[i * c2 + j] = beta * out[i * c2 + j] + detail::mm_hadd_ss(r1);
            }
        }
    }

    if (!padding_impl && m2 % 4 != 0) {
        for (std::size_t i = p1; i < c1 - p1; ++i) {
            for (std::size_t j = p2; j < c2 - p2; ++j) {
                float temp = 0.0;

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

inline void conv2_valid_micro_kernel(const float* in, std::size_t n1, std::size_t n2, const float* kernel, std::size_t m1, std::size_t m2, float* out, float beta, size_t s1, size_t s2, size_t p1, size_t p2) {
    auto kernel_reverse = aligned_allocate_auto<float>(m1 * m2);

    std::reverse_copy(kernel, kernel + m1 * m2, kernel_reverse.get());

    conv2_valid_flipped_micro_kernel(in, n1, n2, kernel_reverse.get(), m1, m2, out, beta, s1, s2, p1, p2);
}

template <typename T>
void pad_2d_input(const opaque_memory<T, 2>& in, opaque_memory<T, 2>& out, size_t p1, size_t p2) {
    auto in_m = in.memory_start();
    auto out_m = out.memory_start();

    for (std::size_t i = 0; i < in.template dim<0>(); ++i) {
        direct_copy_n(in_m + i * in.template dim<1>(), out_m + (i + p1) * out.template dim<1>() + p2, in.template dim<1>());
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
void conv2_valid(const opaque_memory<T, 2>& input, const opaque_memory<T, 2>& kernel, const opaque_memory<T, 2>& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    if(cpp_unlikely(p1 || p2)){
        const auto ws_h = input.template dim<0>() + 2 * p1;
        const auto ws_w = input.template dim<1>() + 2 * p2;

        if(ws_h * ws_w * sizeof(T) < max_workspace){
            etl::dyn_matrix<T, 2> workspace(ws_h, ws_w, T(0));
            auto ws_direct = workspace.direct();

            pad_2d_input(input, ws_direct, p1, p2);

            conv2_valid(workspace.direct(), kernel, conv, s1, s2, 0, 0);

            return;
        }
    }

    const auto k2 = kernel.dim(1);

    if(padding_impl){
        constexpr size_t SS = std::is_same<T, float>::value ? 4 : 2;

        if(k2 % SS > 0){
            const auto pad = SS - k2 % SS;

            auto padded_input = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right_flip(kernel, pad);

            conv2_valid_flipped_micro_kernel(
                padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                padded_kernel.memory_start(), padded_kernel.template dim<0>(), padded_kernel.template dim<1>(),
                conv.memory_start(), 0.0, s1, s2, p1, p2);

            return;
        }
    }

    conv2_valid_micro_kernel(
        input.memory_start(), input.template dim<0>(), input.template dim<1>(),
        kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
        conv.memory_start(), 0.0, s1, s2, p1, p2);
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
        constexpr size_t SS = std::is_same<T, float>::value ? 4 : 2;

        if(k2 % SS > 0){
            const auto pad = SS - k2 % SS;

            auto padded_input = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right(kernel, pad);

            conv2_valid_flipped_micro_kernel(
                padded_input.memory_start(), padded_input.template dim<0>(), padded_input.template dim<1>(),
                padded_kernel.memory_start(), padded_kernel.template dim<0>(), padded_kernel.template dim<1>(),
                conv.memory_start(), 0.0, s1, s2, p1, p2);

            return;
        }
    }

    conv2_valid_flipped_micro_kernel(
        input.memory_start(), input.template dim<0>(), input.template dim<1>(),
        kernel.memory_start(), kernel.template dim<0>(), kernel.template dim<1>(),
        conv.memory_start(), 0.0, s1, s2, p1, p2);
}

#else

//COVERAGE_EXCLUDE_BEGIN

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

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace sse
} //end of namespace impl
} //end of namespace etl
