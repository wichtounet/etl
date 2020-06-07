//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Kernels for row-major matrix - row-major matrix multiplication and
 * assignment to a row-major matrix
 */

#pragma once

namespace etl::impl::vec {

// Optimizations opportunities
// For alpha=1, specific kernels could be done to avoid the multiplication

// The 8-times unrolled loop is poorly handled by clang (3.9, 4.0)
#ifndef ETL_GEMM_SMALL_RR_R_UNROLL_8
#ifndef __clang__
#define ETL_GEMM_SMALL_RR_R_UNROLL_8
#endif
#endif

/*!
 * \brief Optimized version of small GEMM for row major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_small_kernel_rr_to_r(const T* a, const T* b, T* ETL_RESTRICT c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const auto j_end = N & (size_t(-vec_size));

    size_t j = 0;

    auto alpha_vec = vec_type::set(alpha);

    // As an optimization, we directly do the first iteration of the K-loop
    // since K cannot be zero. This avoids having to preset the vector with zero

#ifdef ETL_GEMM_SMALL_RR_R_UNROLL_8
    // Vectorized loop unrolled eight times
    for (; j + vec_size * 7 < j_end; j += vec_size * 8) {
        for (size_t i = 0; i < M; ++i) {
            size_t k = 0;

            auto a1 = vec_type::set(a[i * K + k]);

            auto r1 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 0));
            auto r2 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 1));
            auto r3 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 2));
            auto r4 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 3));
            auto r5 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 4));
            auto r6 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 5));
            auto r7 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 6));
            auto r8 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 7));

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[i * K + k]);

                r1 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 0), r1);
                r2 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 1), r2);
                r3 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 2), r3);
                r4 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 3), r4);
                r5 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 4), r5);
                r6 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 5), r6);
                r7 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 6), r7);
                r8 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 7), r8);
            }

            vec_type::storeu(c + i * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r1));
            vec_type::storeu(c + i * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r2));
            vec_type::storeu(c + i * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r3));
            vec_type::storeu(c + i * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r4));
            vec_type::storeu(c + i * N + j + 4 * vec_size, vec_type::mul(alpha_vec, r5));
            vec_type::storeu(c + i * N + j + 5 * vec_size, vec_type::mul(alpha_vec, r6));
            vec_type::storeu(c + i * N + j + 6 * vec_size, vec_type::mul(alpha_vec, r7));
            vec_type::storeu(c + i * N + j + 7 * vec_size, vec_type::mul(alpha_vec, r8));
        }
    }
#endif

    // Vectorized loop unrolled five times
    // This should max out the number of registers better than four
    for (; j + vec_size * 4 < j_end; j += 5 * vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);
            auto a2 = vec_type::set(a[(i + 1) * K + k]);

            auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
            auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
            auto b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
            auto b4 = vec_type::loadu(b + k * N + j + vec_size * 3);
            auto b5 = vec_type::loadu(b + k * N + j + vec_size * 4);

            auto r11 = vec_type::mul(a1, b1);
            auto r12 = vec_type::mul(a2, b1);

            auto r21 = vec_type::mul(a1, b2);
            auto r22 = vec_type::mul(a2, b2);

            auto r31 = vec_type::mul(a1, b3);
            auto r32 = vec_type::mul(a2, b3);

            auto r41 = vec_type::mul(a1, b4);
            auto r42 = vec_type::mul(a2, b4);

            auto r51 = vec_type::mul(a1, b5);
            auto r52 = vec_type::mul(a2, b5);

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);
                a2 = vec_type::set(a[(i + 1) * K + k]);

                b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
                b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
                b4 = vec_type::loadu(b + k * N + j + vec_size * 3);
                b5 = vec_type::loadu(b + k * N + j + vec_size * 4);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);

                r31 = vec_type::fmadd(a1, b3, r31);
                r32 = vec_type::fmadd(a2, b3, r32);

                r41 = vec_type::fmadd(a1, b4, r41);
                r42 = vec_type::fmadd(a2, b4, r42);

                r51 = vec_type::fmadd(a1, b5, r51);
                r52 = vec_type::fmadd(a2, b5, r52);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + (i + 0) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r31));
            vec_type::storeu(c + (i + 0) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r41));
            vec_type::storeu(c + (i + 0) * N + j + 4 * vec_size, vec_type::mul(alpha_vec, r51));

            vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));
            vec_type::storeu(c + (i + 1) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r32));
            vec_type::storeu(c + (i + 1) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r42));
            vec_type::storeu(c + (i + 1) * N + j + 4 * vec_size, vec_type::mul(alpha_vec, r52));
        }

        if (i < M) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);

            auto r11 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 0));
            auto r21 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 1));
            auto r31 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 2));
            auto r41 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 3));
            auto r51 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 4));

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);

                r11 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 0), r11);
                r21 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 1), r21);
                r31 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 2), r31);
                r41 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 3), r41);
                r51 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 4), r51);
            }

            vec_type::storeu(c + i * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + i * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r31));
            vec_type::storeu(c + i * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r41));
            vec_type::storeu(c + i * N + j + 4 * vec_size, vec_type::mul(alpha_vec, r51));
        }
    }

    // Vectorized loop unrolled four times
    for (; j + vec_size * 3 < j_end; j += 4 * vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);
            auto a2 = vec_type::set(a[(i + 1) * K + k]);

            auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
            auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
            auto b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
            auto b4 = vec_type::loadu(b + k * N + j + vec_size * 3);

            auto r11 = vec_type::mul(a1, b1);
            auto r12 = vec_type::mul(a2, b1);

            auto r21 = vec_type::mul(a1, b2);
            auto r22 = vec_type::mul(a2, b2);

            auto r31 = vec_type::mul(a1, b3);
            auto r32 = vec_type::mul(a2, b3);

            auto r41 = vec_type::mul(a1, b4);
            auto r42 = vec_type::mul(a2, b4);

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);
                a2 = vec_type::set(a[(i + 1) * K + k]);

                b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
                b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
                b4 = vec_type::loadu(b + k * N + j + vec_size * 3);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);

                r31 = vec_type::fmadd(a1, b3, r31);
                r32 = vec_type::fmadd(a2, b3, r32);

                r41 = vec_type::fmadd(a1, b4, r41);
                r42 = vec_type::fmadd(a2, b4, r42);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + (i + 0) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r31));
            vec_type::storeu(c + (i + 0) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r41));

            vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));
            vec_type::storeu(c + (i + 1) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r32));
            vec_type::storeu(c + (i + 1) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r42));
        }

        if (i < M) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);

            auto r11 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 0));
            auto r21 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 1));
            auto r31 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 2));
            auto r41 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 3));

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);

                r11 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 0), r11);
                r21 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 1), r21);
                r31 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 2), r31);
                r41 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 3), r41);
            }

            vec_type::storeu(c + i * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + i * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r31));
            vec_type::storeu(c + i * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r41));
        }
    }

    // Vectorized loop unrolled three times
    for (; j + vec_size * 2 < j_end; j += 3 * vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);
            auto a2 = vec_type::set(a[(i + 1) * K + k]);

            auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
            auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
            auto b3 = vec_type::loadu(b + k * N + j + vec_size * 2);

            auto r11 = vec_type::mul(a1, b1);
            auto r12 = vec_type::mul(a2, b1);

            auto r21 = vec_type::mul(a1, b2);
            auto r22 = vec_type::mul(a2, b2);

            auto r31 = vec_type::mul(a1, b3);
            auto r32 = vec_type::mul(a2, b3);

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);
                a2 = vec_type::set(a[(i + 1) * K + k]);

                b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
                b3 = vec_type::loadu(b + k * N + j + vec_size * 2);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);

                r31 = vec_type::fmadd(a1, b3, r31);
                r32 = vec_type::fmadd(a2, b3, r32);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + (i + 0) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r31));

            vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));
            vec_type::storeu(c + (i + 1) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r32));
        }

        if (i < M) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);

            auto r11 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 0));
            auto r21 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 1));
            auto r31 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 2));

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);

                r11 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 0), r11);
                r21 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 1), r21);
                r31 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 2), r31);
            }

            vec_type::storeu(c + i * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + i * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r31));
        }
    }

    // Vectorized loop unrolled twice
    for (; j + vec_size < j_end; j += 2 * vec_size) {
        size_t i = 0;

        for (; i + 3 < M; i += 4) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);
            auto a2 = vec_type::set(a[(i + 1) * K + k]);
            auto a3 = vec_type::set(a[(i + 2) * K + k]);
            auto a4 = vec_type::set(a[(i + 3) * K + k]);

            auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
            auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

            auto r11 = vec_type::mul(a1, b1);
            auto r12 = vec_type::mul(a2, b1);
            auto r13 = vec_type::mul(a3, b1);
            auto r14 = vec_type::mul(a4, b1);

            auto r21 = vec_type::mul(a1, b2);
            auto r22 = vec_type::mul(a2, b2);
            auto r23 = vec_type::mul(a3, b2);
            auto r24 = vec_type::mul(a4, b2);

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);
                a2 = vec_type::set(a[(i + 1) * K + k]);
                a3 = vec_type::set(a[(i + 2) * K + k]);
                a4 = vec_type::set(a[(i + 3) * K + k]);

                b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);
                r13 = vec_type::fmadd(a3, b1, r13);
                r14 = vec_type::fmadd(a4, b1, r14);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);
                r23 = vec_type::fmadd(a3, b2, r23);
                r24 = vec_type::fmadd(a4, b2, r24);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));

            vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));

            vec_type::storeu(c + (i + 2) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r13));
            vec_type::storeu(c + (i + 2) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r23));

            vec_type::storeu(c + (i + 3) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r14));
            vec_type::storeu(c + (i + 3) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r24));
        }

        for (; i + 1 < M; i += 2) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);
            auto a2 = vec_type::set(a[(i + 1) * K + k]);

            auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
            auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

            auto r11 = vec_type::mul(a1, b1);
            auto r12 = vec_type::mul(a2, b1);

            auto r21 = vec_type::mul(a1, b2);
            auto r22 = vec_type::mul(a2, b2);

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);
                a2 = vec_type::set(a[(i + 1) * K + k]);

                b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));

            vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));
        }

        if (i < M) {
            size_t k = 0;

            auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
            auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

            auto a1 = vec_type::set(a[(i + 0) * K + k]);

            auto r11 = vec_type::mul(a1, b1);
            auto r21 = vec_type::mul(a1, b2);

            for (++k; k < K; ++k) {
                b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

                a1 = vec_type::set(a[(i + 0) * K + k]);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a1, b2, r21);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r21));
        }
    }

    // Vectorized loop
    for (; j < j_end; j += vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);
            auto a2 = vec_type::set(a[(i + 1) * K + k]);

            auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);

            auto r1 = vec_type::mul(a1, b1);
            auto r2 = vec_type::mul(a2, b1);

            for (++k; k < K; ++k) {
                b1 = vec_type::loadu(b + k * N + j + vec_size * 0);

                a1 = vec_type::set(a[(i + 0) * K + k]);
                a2 = vec_type::set(a[(i + 1) * K + k]);

                r1 = vec_type::fmadd(a1, b1, r1);
                r2 = vec_type::fmadd(a2, b1, r2);
            }

            vec_type::storeu(c + (i + 0) * N + j, vec_type::mul(alpha_vec, r1));
            vec_type::storeu(c + (i + 1) * N + j, vec_type::mul(alpha_vec, r2));
        }

        if (i < M) {
            size_t k = 0;

            auto a1 = vec_type::set(a[(i + 0) * K + k]);

            auto r1 = vec_type::mul(a1, vec_type::loadu(b + k * N + j + vec_size * 0));

            for (++k; k < K; ++k) {
                a1 = vec_type::set(a[(i + 0) * K + k]);

                r1 = vec_type::fmadd(a1, vec_type::loadu(b + k * N + j + vec_size * 0), r1);
            }

            vec_type::storeu(c + (i + 0) * N + j, vec_type::mul(alpha_vec, r1));
        }
    }

    // Remainder Loop unrolled by 2
    for (; j + 1 < N; j += 2) {
        const size_t j1 = j + 0;
        const size_t j2 = j + 1;

        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            size_t k = 0;

            auto r11 = a[(i + 0) * K + k] * b[k * N + j1];
            auto r21 = a[(i + 0) * K + k] * b[k * N + j2];
            auto r12 = a[(i + 1) * K + k] * b[k * N + j1];
            auto r22 = a[(i + 1) * K + k] * b[k * N + j2];

            for (++k; k < K; ++k) {
                r11 += a[(i + 0) * K + k] * b[k * N + j1];
                r21 += a[(i + 0) * K + k] * b[k * N + j2];
                r12 += a[(i + 1) * K + k] * b[k * N + j1];
                r22 += a[(i + 1) * K + k] * b[k * N + j2];
            }

            c[(i + 0) * N + j1] = alpha * r11;
            c[(i + 0) * N + j2] = alpha * r21;
            c[(i + 1) * N + j1] = alpha * r12;
            c[(i + 1) * N + j2] = alpha * r22;
        }

        if (i < M) {
            size_t k = 0;

            auto r1 = a[i * K + k] * b[k * N + j1];
            auto r2 = a[i * K + k] * b[k * N + j2];

            for (++k; k < K; ++k) {
                r1 += a[i * K + k] * b[k * N + j1];
                r2 += a[i * K + k] * b[k * N + j2];
            }

            c[i * N + j1] = alpha * r1;
            c[i * N + j2] = alpha * r2;
        }
    }

    // Final remainder loop iteration
    if (j < N) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            size_t k = 0;

            auto r1 = a[(i + 0) * K + k] * b[k * N + j];
            auto r2 = a[(i + 1) * K + k] * b[k * N + j];

            for (++k; k < K; ++k) {
                r1 += a[(i + 0) * K + k] * b[k * N + j];
                r2 += a[(i + 1) * K + k] * b[k * N + j];
            }

            c[(i + 0) * N + j] = alpha * r1;
            c[(i + 1) * N + j] = alpha * r2;
        }

        if (i < M) {
            size_t k = 0;

            auto r1 = a[i * K + k] * b[k * N + j];

            for (++k; k < K; ++k) {
                r1 += a[i * K + k] * b[k * N + j];
            }

            c[i * N + j] = alpha * r1;
        }
    }
}

/*!
 * \brief Optimized version of large GEMM for row major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 * \param beta The multipliying of the previous value
 */
template <typename V, typename T>
void gemm_large_kernel_rr_to_r(const T* a, const T* b, T* ETL_RESTRICT c, size_t M, size_t N, size_t K, T alpha, T beta) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t n_block_size = 128;
    const size_t m_block_size = 64;
    const size_t k_block_size = 128;

    auto alpha_vec = vec_type::set(alpha);

    // Note: There is a small benefit to parallelize this
    // However, most of the parallel benefit is in larger matrices and the
    // larger algorithm

    for (size_t block_j = 0; block_j < N; block_j += n_block_size) {
        const size_t j_end = std::min(block_j + n_block_size, N);

        for (size_t block_i = 0; block_i < M; block_i += m_block_size) {
            const size_t i_end = std::min(block_i + m_block_size, M);

            if (beta == T(0.0)) {
                for (size_t i = block_i; i < i_end; ++i) {
                    for (size_t j = block_j; j < j_end; ++j) {
                        c[i * N + j] = 0;
                    }
                }
            } else {
                for (size_t i = block_i; i < i_end; ++i) {
                    for (size_t j = block_j; j < j_end; ++j) {
                        c[i * N + j] = beta * c[i * N + j];
                    }
                }
            }

            for (size_t block_k = 0; block_k < K; block_k += k_block_size) {
                const size_t k_end = std::min(block_k + k_block_size, K);

                size_t j = block_j;

                for (; j + vec_size * 4 - 1 < j_end; j += vec_size * 4) {
                    const size_t j1 = j + vec_size * 1;
                    const size_t j2 = j + vec_size * 2;
                    const size_t j3 = j + vec_size * 3;

                    size_t i = block_i;

                    for (; i + 1 < i_end; i += 2) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j);
                        auto r12 = vec_type::loadu(c + (i + 0) * N + j1);
                        auto r13 = vec_type::loadu(c + (i + 0) * N + j2);
                        auto r14 = vec_type::loadu(c + (i + 0) * N + j3);

                        auto r21 = vec_type::loadu(c + (i + 1) * N + j);
                        auto r22 = vec_type::loadu(c + (i + 1) * N + j1);
                        auto r23 = vec_type::loadu(c + (i + 1) * N + j2);
                        auto r24 = vec_type::loadu(c + (i + 1) * N + j3);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) * K + k]);
                            auto a2 = vec_type::set(a[(i + 1) * K + k]);

                            auto b1 = vec_type::loadu(b + k * N + j);
                            auto b2 = vec_type::loadu(b + k * N + j1);
                            auto b3 = vec_type::loadu(b + k * N + j2);
                            auto b4 = vec_type::loadu(b + k * N + j3);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);
                            r13 = vec_type::fmadd(a1, b3, r13);
                            r14 = vec_type::fmadd(a1, b4, r14);

                            r21 = vec_type::fmadd(a2, b1, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                            r23 = vec_type::fmadd(a2, b3, r23);
                            r24 = vec_type::fmadd(a2, b4, r24);
                        }

                        vec_type::storeu(c + (i + 0) * N + j, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + (i + 0) * N + j1, vec_type::mul(alpha_vec, r12));
                        vec_type::storeu(c + (i + 0) * N + j2, vec_type::mul(alpha_vec, r13));
                        vec_type::storeu(c + (i + 0) * N + j3, vec_type::mul(alpha_vec, r14));
                        vec_type::storeu(c + (i + 1) * N + j, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + (i + 1) * N + j1, vec_type::mul(alpha_vec, r22));
                        vec_type::storeu(c + (i + 1) * N + j2, vec_type::mul(alpha_vec, r23));
                        vec_type::storeu(c + (i + 1) * N + j3, vec_type::mul(alpha_vec, r24));
                    }

                    if (i < i_end) {
                        auto r1 = vec_type::loadu(c + (i + 0) * N + j);
                        auto r2 = vec_type::loadu(c + (i + 0) * N + j1);
                        auto r3 = vec_type::loadu(c + (i + 0) * N + j2);
                        auto r4 = vec_type::loadu(c + (i + 0) * N + j3);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) * K + k]);

                            auto b1 = vec_type::loadu(b + k * N + j);
                            auto b2 = vec_type::loadu(b + k * N + j1);
                            auto b3 = vec_type::loadu(b + k * N + j2);
                            auto b4 = vec_type::loadu(b + k * N + j3);

                            r1 = vec_type::fmadd(a1, b1, r1);
                            r2 = vec_type::fmadd(a1, b2, r2);
                            r3 = vec_type::fmadd(a1, b3, r3);
                            r4 = vec_type::fmadd(a1, b4, r4);
                        }

                        vec_type::storeu(c + (i + 0) * N + j, vec_type::mul(alpha_vec, r1));
                        vec_type::storeu(c + (i + 0) * N + j1, vec_type::mul(alpha_vec, r2));
                        vec_type::storeu(c + (i + 0) * N + j2, vec_type::mul(alpha_vec, r3));
                        vec_type::storeu(c + (i + 0) * N + j3, vec_type::mul(alpha_vec, r4));
                    }
                }

                for (; j + vec_size * 2 - 1 < j_end; j += vec_size * 2) {
                    const size_t j1(j + vec_size);

                    size_t i = block_i;

                    for (; i + 3 < i_end; i += 4) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j);
                        auto r12 = vec_type::loadu(c + (i + 0) * N + j1);

                        auto r21 = vec_type::loadu(c + (i + 1) * N + j);
                        auto r22 = vec_type::loadu(c + (i + 1) * N + j1);

                        auto r31 = vec_type::loadu(c + (i + 2) * N + j);
                        auto r32 = vec_type::loadu(c + (i + 2) * N + j1);

                        auto r41 = vec_type::loadu(c + (i + 3) * N + j);
                        auto r42 = vec_type::loadu(c + (i + 3) * N + j1);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) * K + k]);
                            auto a2 = vec_type::set(a[(i + 1) * K + k]);
                            auto a3 = vec_type::set(a[(i + 2) * K + k]);
                            auto a4 = vec_type::set(a[(i + 3) * K + k]);

                            auto b1 = vec_type::loadu(b + k * N + j);
                            auto b2 = vec_type::loadu(b + k * N + j1);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);

                            r21 = vec_type::fmadd(a2, b1, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);

                            r31 = vec_type::fmadd(a3, b1, r31);
                            r32 = vec_type::fmadd(a3, b2, r32);

                            r41 = vec_type::fmadd(a4, b1, r41);
                            r42 = vec_type::fmadd(a4, b2, r42);
                        }

                        vec_type::storeu(c + (i + 0) * N + j, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + (i + 0) * N + j1, vec_type::mul(alpha_vec, r12));
                        vec_type::storeu(c + (i + 1) * N + j, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + (i + 1) * N + j1, vec_type::mul(alpha_vec, r22));
                        vec_type::storeu(c + (i + 2) * N + j, vec_type::mul(alpha_vec, r31));
                        vec_type::storeu(c + (i + 2) * N + j1, vec_type::mul(alpha_vec, r32));
                        vec_type::storeu(c + (i + 3) * N + j, vec_type::mul(alpha_vec, r41));
                        vec_type::storeu(c + (i + 3) * N + j1, vec_type::mul(alpha_vec, r42));
                    }

                    for (; i + 2 - 1 < i_end; i += 2) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j);
                        auto r12 = vec_type::loadu(c + (i + 0) * N + j1);

                        auto r21 = vec_type::loadu(c + (i + 1) * N + j);
                        auto r22 = vec_type::loadu(c + (i + 1) * N + j1);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) * K + k]);
                            auto a2 = vec_type::set(a[(i + 1) * K + k]);

                            auto b1 = vec_type::loadu(b + k * N + j);
                            auto b2 = vec_type::loadu(b + k * N + j1);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);

                            r21 = vec_type::fmadd(a2, b1, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                        }

                        vec_type::storeu(c + (i + 0) * N + j, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + (i + 0) * N + j1, vec_type::mul(alpha_vec, r12));
                        vec_type::storeu(c + (i + 1) * N + j, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + (i + 1) * N + j1, vec_type::mul(alpha_vec, r22));
                    }

                    if (i < i_end) {
                        auto r1 = vec_type::loadu(c + (i + 0) * N + j);
                        auto r2 = vec_type::loadu(c + (i + 0) * N + j1);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) * K + k]);

                            auto b1 = vec_type::loadu(b + k * N + j);
                            auto b2 = vec_type::loadu(b + k * N + j1);

                            r1 = vec_type::fmadd(a1, b1, r1);
                            r2 = vec_type::fmadd(a1, b2, r2);
                        }

                        vec_type::storeu(c + (i + 0) * N + j, vec_type::mul(alpha_vec, r1));
                        vec_type::storeu(c + (i + 0) * N + j1, vec_type::mul(alpha_vec, r2));
                    }
                }

                for (; j + vec_size - 1 < j_end; j += vec_size) {
                    for (size_t i = block_i; i < i_end; ++i) {
                        auto r1 = vec_type::loadu(c + (i + 0) * N + j);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) * K + k]);
                            auto b1 = vec_type::loadu(b + k * N + j);
                            r1      = vec_type::fmadd(a1, b1, r1);
                        }

                        vec_type::storeu(c + (i + 0) * N + j, vec_type::mul(alpha_vec, r1));
                    }
                }

                for (; j < j_end; ++j) {
                    for (size_t i = block_i; i < i_end; ++i) {
                        auto value = c[i * N + j];

                        for (size_t k = block_k; k < k_end; ++k) {
                            value += a[i * K + k] * b[k * N + j];
                        }

                        c[i * N + j] = alpha * value;
                    }
                }
            }
        }
    }
}

template <size_t vec_size>
inline constexpr size_t prev_vec_block(size_t value) noexcept {
    return value - (value % vec_size);
}

/*!
 * \brief Optimized version of large GEMM for row major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 * \param beta The multipliying of the previous value
 */
template <typename V, typename T>
void gemm_large_kernel_rr_to_r_temp(const T* a, const T* b, T* ETL_RESTRICT c, size_t M, size_t N, size_t K, T alpha, T beta) {
    using vec_type = V;

    constexpr size_t vec_size = vec_type::template traits<T>::size;

    constexpr size_t K_BLOCK = 112 * (16 / sizeof(T));
    constexpr size_t J_BLOCK = 96;

    etl::custom_dyn_matrix<T> A(const_cast<T*>(a), M, K);
    etl::custom_dyn_matrix<T> B(const_cast<T*>(b), K, N);
    etl::custom_dyn_matrix<T> C(c, M, N);

    if (beta == T(0)) {
        C = 0;
    } else {
        C = beta * C;
    }

    auto batch_fun_j = [&](const size_t jfirst, const size_t jlast) {
        etl::dyn_matrix_impl<T, order::RowMajor> A2(M, K_BLOCK);
        etl::dyn_matrix_impl<T, order::ColumnMajor> B2(K_BLOCK, J_BLOCK);

        auto * A2M = A2.memory_start();
        auto * B2M = B2.memory_start();

        size_t kblock = 0;
        size_t kk     = 0;

        // Main loop
        for (; kk + vec_size - 1 < K; kk += kblock) {
            kblock = kk + K_BLOCK <= K ? K_BLOCK : prev_vec_block<vec_size>(K - kk);

            if (!kblock) {
                continue;
            }

            // Copy A into A2
            for (size_t iii = 0; iii < M; ++iii) {
                for (size_t kkk = 0; kkk < kblock; ++kkk) {
                    A2(iii, kkk) = A(iii, kkk + kk);
                }
            }

            size_t jj     = jfirst;
            size_t jblock = 0;

            for (; jj < jlast; jj += jblock) {
                jblock = jj + J_BLOCK <= jlast ? J_BLOCK : jlast - jj;

                // Copy B into B2
                for (size_t kkk = 0; kkk < kblock; ++kkk) {
                    for (size_t jjj = 0; jjj < jblock; ++jjj) {
                        B2(kkk, jjj) = B(kkk + kk, jjj + jj);
                    }
                }

                size_t i = 0;

                for (; i + 4 < M; i += 5) {
                    size_t j = 0;

                    for (; j + 1 < jblock; j += 2) {
                        size_t k = 0;

                        auto a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                        auto a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);
                        auto a3 = vec_type::load(A2M + (i + 2) * K_BLOCK + k);
                        auto a4 = vec_type::load(A2M + (i + 3) * K_BLOCK + k);
                        auto a5 = vec_type::load(A2M + (i + 4) * K_BLOCK + k);

                        auto b1 = vec_type::load(B2M + (j + 0) * K_BLOCK + k);
                        auto b2 = vec_type::load(B2M + (j + 1) * K_BLOCK + k);

                        auto xmm1  = vec_type::mul(a1, b1);
                        auto xmm2  = vec_type::mul(a1, b2);
                        auto xmm3  = vec_type::mul(a2, b1);
                        auto xmm4  = vec_type::mul(a2, b2);
                        auto xmm5  = vec_type::mul(a3, b1);
                        auto xmm6  = vec_type::mul(a3, b2);
                        auto xmm7  = vec_type::mul(a4, b1);
                        auto xmm8  = vec_type::mul(a4, b2);
                        auto xmm9  = vec_type::mul(a5, b1);
                        auto xmm10 = vec_type::mul(a5, b2);

                        for (k += vec_size; k < kblock; k += vec_size) {
                            a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                            a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);
                            a3 = vec_type::load(A2M + (i + 2) * K_BLOCK + k);
                            a4 = vec_type::load(A2M + (i + 3) * K_BLOCK + k);
                            a5 = vec_type::load(A2M + (i + 4) * K_BLOCK + k);

                            b1 = vec_type::load(B2M + (j + 0) * K_BLOCK + k);
                            b2 = vec_type::load(B2M + (j + 1) * K_BLOCK + k);

                            xmm1  = vec_type::fmadd(a1, b1, xmm1);
                            xmm2  = vec_type::fmadd(a1, b2, xmm2);
                            xmm3  = vec_type::fmadd(a2, b1, xmm3);
                            xmm4  = vec_type::fmadd(a2, b2, xmm4);
                            xmm5  = vec_type::fmadd(a3, b1, xmm5);
                            xmm6  = vec_type::fmadd(a3, b2, xmm6);
                            xmm7  = vec_type::fmadd(a4, b1, xmm7);
                            xmm8  = vec_type::fmadd(a4, b2, xmm8);
                            xmm9  = vec_type::fmadd(a5, b1, xmm9);
                            xmm10 = vec_type::fmadd(a5, b2, xmm10);
                        }

                        C(i + 0, jj + j + 0) += alpha * vec_type::hadd(xmm1);
                        C(i + 0, jj + j + 1) += alpha * vec_type::hadd(xmm2);
                        C(i + 1, jj + j + 0) += alpha * vec_type::hadd(xmm3);
                        C(i + 1, jj + j + 1) += alpha * vec_type::hadd(xmm4);
                        C(i + 2, jj + j + 0) += alpha * vec_type::hadd(xmm5);
                        C(i + 2, jj + j + 1) += alpha * vec_type::hadd(xmm6);
                        C(i + 3, jj + j + 0) += alpha * vec_type::hadd(xmm7);
                        C(i + 3, jj + j + 1) += alpha * vec_type::hadd(xmm8);
                        C(i + 4, jj + j + 0) += alpha * vec_type::hadd(xmm9);
                        C(i + 4, jj + j + 1) += alpha * vec_type::hadd(xmm10);
                    }

                    if (j < jblock) {
                        size_t k = 0;

                        auto a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                        auto a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);
                        auto a3 = vec_type::load(A2M + (i + 2) * K_BLOCK + k);
                        auto a4 = vec_type::load(A2M + (i + 3) * K_BLOCK + k);
                        auto a5 = vec_type::load(A2M + (i + 4) * K_BLOCK + k);

                        auto b1 = vec_type::load(B2M + j * K_BLOCK + k);

                        auto xmm1 = vec_type::mul(a1, b1);
                        auto xmm2 = vec_type::mul(a2, b1);
                        auto xmm3 = vec_type::mul(a3, b1);
                        auto xmm4 = vec_type::mul(a4, b1);
                        auto xmm5 = vec_type::mul(a5, b1);

                        for (k += vec_size; k < kblock; k += vec_size) {
                            a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                            a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);
                            a3 = vec_type::load(A2M + (i + 2) * K_BLOCK + k);
                            a4 = vec_type::load(A2M + (i + 3) * K_BLOCK + k);
                            a5 = vec_type::load(A2M + (i + 4) * K_BLOCK + k);

                            b1 = vec_type::load(B2M + j * K_BLOCK + k);

                            xmm1 = vec_type::fmadd(a1, b1, xmm1);
                            xmm2 = vec_type::fmadd(a2, b1, xmm2);
                            xmm3 = vec_type::fmadd(a3, b1, xmm3);
                            xmm4 = vec_type::fmadd(a4, b1, xmm4);
                            xmm5 = vec_type::fmadd(a5, b1, xmm5);
                        }

                        C(i + 0, jj + j) += alpha * vec_type::hadd(xmm1);
                        C(i + 1, jj + j) += alpha * vec_type::hadd(xmm2);
                        C(i + 2, jj + j) += alpha * vec_type::hadd(xmm3);
                        C(i + 3, jj + j) += alpha * vec_type::hadd(xmm4);
                        C(i + 4, jj + j) += alpha * vec_type::hadd(xmm5);
                    }
                }

                for (; i + 1 < M; i += 2) {
                    size_t j = 0;

                    for (; j + 3 < jblock; j += 4) {
                        size_t k = 0;

                        auto a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                        auto a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);

                        auto b1 = vec_type::load(B2M + (j + 0) * K_BLOCK + k);
                        auto b2 = vec_type::load(B2M + (j + 1) * K_BLOCK + k);
                        auto b3 = vec_type::load(B2M + (j + 2) * K_BLOCK + k);
                        auto b4 = vec_type::load(B2M + (j + 3) * K_BLOCK + k);

                        auto xmm1 = vec_type::mul(a1, b1);
                        auto xmm2 = vec_type::mul(a1, b2);
                        auto xmm3 = vec_type::mul(a1, b3);
                        auto xmm4 = vec_type::mul(a1, b4);
                        auto xmm5 = vec_type::mul(a2, b1);
                        auto xmm6 = vec_type::mul(a2, b2);
                        auto xmm7 = vec_type::mul(a2, b3);
                        auto xmm8 = vec_type::mul(a2, b4);

                        for (k += vec_size; k < kblock; k += vec_size) {
                            a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                            a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);

                            b1 = vec_type::load(B2M + (j + 0) * K_BLOCK + k);
                            b2 = vec_type::load(B2M + (j + 1) * K_BLOCK + k);
                            b3 = vec_type::load(B2M + (j + 2) * K_BLOCK + k);
                            b4 = vec_type::load(B2M + (j + 3) * K_BLOCK + k);

                            xmm1 = vec_type::fmadd(a1, b1, xmm1);
                            xmm2 = vec_type::fmadd(a1, b2, xmm2);
                            xmm3 = vec_type::fmadd(a1, b3, xmm3);
                            xmm4 = vec_type::fmadd(a1, b4, xmm4);

                            xmm5 = vec_type::fmadd(a2, b1, xmm5);
                            xmm6 = vec_type::fmadd(a2, b2, xmm6);
                            xmm7 = vec_type::fmadd(a2, b3, xmm7);
                            xmm8 = vec_type::fmadd(a2, b2, xmm8);
                        }

                        C(i + 0, jj + j + 0) += alpha * vec_type::hadd(xmm1);
                        C(i + 0, jj + j + 1) += alpha * vec_type::hadd(xmm2);
                        C(i + 0, jj + j + 2) += alpha * vec_type::hadd(xmm3);
                        C(i + 0, jj + j + 3) += alpha * vec_type::hadd(xmm4);

                        C(i + 1, jj + j + 0) += alpha * vec_type::hadd(xmm5);
                        C(i + 1, jj + j + 1) += alpha * vec_type::hadd(xmm6);
                        C(i + 1, jj + j + 2) += alpha * vec_type::hadd(xmm7);
                        C(i + 1, jj + j + 3) += alpha * vec_type::hadd(xmm8);
                    }

                    for (; j + 1 < jblock; j += 2) {
                        size_t k = 0;

                        auto a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                        auto a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);

                        auto b1 = vec_type::load(B2M + (j + 0) * K_BLOCK + k);
                        auto b2 = vec_type::load(B2M + (j + 1) * K_BLOCK + k);

                        auto xmm1 = vec_type::mul(a1, b1);
                        auto xmm2 = vec_type::mul(a1, b2);
                        auto xmm3 = vec_type::mul(a2, b1);
                        auto xmm4 = vec_type::mul(a2, b2);

                        for (k += vec_size; k < kblock; k += vec_size) {
                            a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                            a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);

                            b1 = vec_type::load(B2M + (j + 0) * K_BLOCK + k);
                            b2 = vec_type::load(B2M + (j + 1) * K_BLOCK + k);

                            xmm1 = vec_type::fmadd(a1, b1, xmm1);
                            xmm2 = vec_type::fmadd(a1, b2, xmm2);

                            xmm3 = vec_type::fmadd(a2, b1, xmm3);
                            xmm4 = vec_type::fmadd(a2, b2, xmm4);
                        }

                        C(i + 0, jj + j + 0) += alpha * vec_type::hadd(xmm1);
                        C(i + 0, jj + j + 1) += alpha * vec_type::hadd(xmm2);

                        C(i + 1, jj + j + 0) += alpha * vec_type::hadd(xmm3);
                        C(i + 1, jj + j + 1) += alpha * vec_type::hadd(xmm4);
                    }

                    if (j < jblock) {
                        size_t k = 0;

                        auto a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                        auto a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);

                        auto b1 = vec_type::load(B2M + j * K_BLOCK + k);

                        auto xmm1 = vec_type::mul(a1, b1);
                        auto xmm2 = vec_type::mul(a2, b1);

                        for (k += vec_size; k < kblock; k += vec_size) {
                            a1 = vec_type::load(A2M + (i + 0) * K_BLOCK + k);
                            a2 = vec_type::load(A2M + (i + 1) * K_BLOCK + k);

                            b1 = vec_type::load(B2M + j * K_BLOCK + k);

                            xmm1 = vec_type::fmadd(a1, b1, xmm1);
                            xmm2 = vec_type::fmadd(a2, b1, xmm2);
                        }

                        C(i + 0, jj + j) += alpha * vec_type::hadd(xmm1);
                        C(i + 1, jj + j) += alpha * vec_type::hadd(xmm2);
                    }
                }

                if (i < M) {
                    size_t j = 0;

                    for (; j + 1 < jblock; j += 2) {
                        size_t k = 0;

                        auto a1 = vec_type::load(A2M + i * K_BLOCK + k);

                        auto b1 = vec_type::load(B2M + (j + 0) * K_BLOCK + k);
                        auto b2 = vec_type::load(B2M + (j + 1) * K_BLOCK + k);

                        auto xmm1 = vec_type::mul(a1, b1);
                        auto xmm2 = vec_type::mul(a1, b2);

                        for (k += vec_size; k < kblock; k += vec_size) {
                            a1 = vec_type::load(A2M + i * K_BLOCK + k);

                            b1 = vec_type::load(B2M + (j + 0) * K_BLOCK + k);
                            b2 = vec_type::load(B2M + (j + 1) * K_BLOCK + k);

                            xmm1 = vec_type::fmadd(a1, b1, xmm1);
                            xmm2 = vec_type::fmadd(a1, b2, xmm2);
                        }

                        C(i, jj + j + 0) += alpha * vec_type::hadd(xmm1);
                        C(i, jj + j + 1) += alpha * vec_type::hadd(xmm2);
                    }

                    if (j < jblock) {
                        size_t k = 0;

                        auto a1 = vec_type::load(A2M + i * K_BLOCK + k);

                        auto b1 = vec_type::load(B2M + j * K_BLOCK + k);

                        auto xmm1 = vec_type::mul(a1, b1);

                        for (k += vec_size; k < kblock; k += vec_size) {
                            a1 = vec_type::load(A2M + i * K_BLOCK + k);

                            b1 = vec_type::load(B2M + j * K_BLOCK + k);

                            xmm1 = vec_type::fmadd(a1, b1, xmm1);
                        }

                        C(i, jj + j) += alpha * vec_type::hadd(xmm1);
                    }
                }
            }
        }

        // Remainder loop
        if (kk < K) {
            const size_t kend = K - kk;

            // Copy A into A2
            for (size_t iii = 0; iii < M; ++iii) {
                for (size_t kkk = 0; kkk < kend; ++kkk) {
                    A2(iii, kkk) = A(iii, kkk + kk);
                }
            }

            size_t jj     = 0;
            size_t jblock = 0;

            for (; jj < jlast; jj += jblock) {
                jblock = jj + J_BLOCK <= jlast ? J_BLOCK : jlast - jj;

                // Copy B into B2
                for (size_t kkk = 0; kkk < kend; ++kkk) {
                    for (size_t jjj = 0; jjj < jblock; ++jjj) {
                        B2(kkk, jjj) = B(kkk + kk, jjj + jj);
                    }
                }

                size_t i = 0;

                for (; i + 4 < M; i += 5) {
                    size_t j = 0;

                    for (; j + 1 < jblock; j += 2) {
                        for (size_t k = 0; k < kend; ++k) {
                            C(i + 0, jj + j + 0) += alpha * A2(i + 0, k) * B2(k, j + 0);
                            C(i + 0, jj + j + 1) += alpha * A2(i + 0, k) * B2(k, j + 1);
                            C(i + 1, jj + j + 0) += alpha * A2(i + 1, k) * B2(k, j + 0);
                            C(i + 1, jj + j + 1) += alpha * A2(i + 1, k) * B2(k, j + 1);
                            C(i + 2, jj + j + 0) += alpha * A2(i + 2, k) * B2(k, j + 0);
                            C(i + 2, jj + j + 1) += alpha * A2(i + 2, k) * B2(k, j + 1);
                            C(i + 3, jj + j + 0) += alpha * A2(i + 3, k) * B2(k, j + 0);
                            C(i + 3, jj + j + 1) += alpha * A2(i + 3, k) * B2(k, j + 1);
                            C(i + 4, jj + j + 0) += alpha * A2(i + 4, k) * B2(k, j + 0);
                            C(i + 4, jj + j + 1) += alpha * A2(i + 4, k) * B2(k, j + 1);
                        }
                    }

                    if (j < jblock) {
                        for (size_t k = 0; k < kend; ++k) {
                            C(i + 0, jj + j) += alpha * A2(i + 0, k) * B2(k, j);
                            C(i + 1, jj + j) += alpha * A2(i + 1, k) * B2(k, j);
                            C(i + 2, jj + j) += alpha * A2(i + 2, k) * B2(k, j);
                            C(i + 3, jj + j) += alpha * A2(i + 3, k) * B2(k, j);
                            C(i + 4, jj + j) += alpha * A2(i + 4, k) * B2(k, j);
                        }
                    }
                }

                for (; i + 1 < M; i += 2) {
                    size_t j = 0;

                    for (; j + 1 < jblock; j += 2) {
                        for (size_t k = 0; k < kend; ++k) {
                            C(i + 0, jj + j + 0) += alpha * A2(i + 0, k) * B2(k, j + 0);
                            C(i + 0, jj + j + 1) += alpha * A2(i + 0, k) * B2(k, j + 1);
                            C(i + 1, jj + j + 0) += alpha * A2(i + 1, k) * B2(k, j + 0);
                            C(i + 1, jj + j + 1) += alpha * A2(i + 1, k) * B2(k, j + 1);
                        }
                    }

                    if (j < jblock) {
                        for (size_t k = 0; k < kend; ++k) {
                            C(i + 0, jj + j) += alpha * A2(i + 0, k) * B2(k, j);
                            C(i + 1, jj + j) += alpha * A2(i + 1, k) * B2(k, j);
                        }
                    }
                }

                if (i < M) {
                    size_t j = 0;

                    for (; j + 1 < jblock; j += 2) {
                        for (size_t k = 0; k < kend; ++k) {
                            C(i, jj + j + 0) += alpha * A2(i, k) * B2(k, j + 0);
                            C(i, jj + j + 1) += alpha * A2(i, k) * B2(k, j + 1);
                        }
                    }

                    if (j < jblock) {
                        for (size_t k = 0; k < kend; ++k) {
                            C(i, jj + j) += alpha * A2(i, k) * B2(k, j);
                        }
                    }
                }
            }
        }
    };

    engine_dispatch_1d(batch_fun_j, 0, N, J_BLOCK);
}

/*!
 * \brief Vectorized implementation of row-major matrix - row-major matrix
 * multiplication and assignment into a row-major matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 *
 * \param M The number of rows of the matrix A and rows of the matrix C
 * \param N The number of columns of the matrix B and columns of the matrix C
 * \param K The number of columns of the matrix A and rows of the matrix B
 */
template <typename T>
void gemm_rr_to_r(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    // Dispatch to the best kernel

    if (K * N <= gemm_rr_small_threshold) {
        gemm_small_kernel_rr_to_r<default_vec>(a, b, c, M, N, K, alpha);
    } else if (K * N <= gemm_rr_medium_threshold) {
        gemm_large_kernel_rr_to_r<default_vec>(a, b, c, M, N, K, alpha, T(0));
    } else {
        gemm_large_kernel_rr_to_r_temp<default_vec>(a, b, c, M, N, K, alpha, T(0));
    }
}

} //end of namespace etl::impl::vec
