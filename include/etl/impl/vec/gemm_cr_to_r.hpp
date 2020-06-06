//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Kernels for colum-major matrix - row-major matrix multiplication and
 * assignment to a row-major matrix
 */

#pragma once

namespace etl::impl::vec {

/*!
 * \brief Optimized version of GEMM for assignment of a small
 * Column-Major Matrix - Row Major Matrix to a Row Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_small_kernel_cr_to_r(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    auto alpha_vec = vec_type::set(alpha);

    const auto j_end = N & (size_t(-vec_size));

    size_t j = 0;

    for (; j + 7 * vec_size < j_end; j += 8 * vec_size) {
        size_t i = 0;

        for (; i < M; i++) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();
            auto r13 = vec_type::template zero<T>();
            auto r14 = vec_type::template zero<T>();
            auto r15 = vec_type::template zero<T>();
            auto r16 = vec_type::template zero<T>();
            auto r17 = vec_type::template zero<T>();
            auto r18 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) + k * M]);

                auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);
                auto b3 = vec_type::loadu(b + k * N + j + 2 * vec_size);
                auto b4 = vec_type::loadu(b + k * N + j + 3 * vec_size);
                auto b5 = vec_type::loadu(b + k * N + j + 4 * vec_size);
                auto b6 = vec_type::loadu(b + k * N + j + 5 * vec_size);
                auto b7 = vec_type::loadu(b + k * N + j + 6 * vec_size);
                auto b8 = vec_type::loadu(b + k * N + j + 7 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a1, b2, r12);
                r13 = vec_type::fmadd(a1, b3, r13);
                r14 = vec_type::fmadd(a1, b4, r14);
                r15 = vec_type::fmadd(a1, b5, r15);
                r16 = vec_type::fmadd(a1, b6, r16);
                r17 = vec_type::fmadd(a1, b7, r17);
                r18 = vec_type::fmadd(a1, b8, r18);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 0) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r13));
            vec_type::storeu(c + (i + 0) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r14));
            vec_type::storeu(c + (i + 0) * N + j + 4 * vec_size, vec_type::mul(alpha_vec, r15));
            vec_type::storeu(c + (i + 0) * N + j + 5 * vec_size, vec_type::mul(alpha_vec, r16));
            vec_type::storeu(c + (i + 0) * N + j + 6 * vec_size, vec_type::mul(alpha_vec, r17));
            vec_type::storeu(c + (i + 0) * N + j + 7 * vec_size, vec_type::mul(alpha_vec, r18));
        }
    }

    for (; j + 3 * vec_size < j_end; j += 4 * vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            auto r12 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            auto r13 = vec_type::template zero<T>();
            auto r23 = vec_type::template zero<T>();

            auto r14 = vec_type::template zero<T>();
            auto r24 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) + k * M]);
                auto a2 = vec_type::set(a[(i + 1) + k * M]);

                auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);
                auto b3 = vec_type::loadu(b + k * N + j + 2 * vec_size);
                auto b4 = vec_type::loadu(b + k * N + j + 3 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);

                r12 = vec_type::fmadd(a1, b2, r12);
                r22 = vec_type::fmadd(a2, b2, r22);

                r13 = vec_type::fmadd(a1, b3, r13);
                r23 = vec_type::fmadd(a2, b3, r23);

                r14 = vec_type::fmadd(a1, b4, r14);
                r24 = vec_type::fmadd(a2, b4, r24);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r21));

            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));

            vec_type::storeu(c + (i + 0) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r13));
            vec_type::storeu(c + (i + 1) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r23));

            vec_type::storeu(c + (i + 0) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r14));
            vec_type::storeu(c + (i + 1) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r24));
        }

        for (; i < M; i++) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();
            auto r13 = vec_type::template zero<T>();
            auto r14 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) + k * M]);

                auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);
                auto b3 = vec_type::loadu(b + k * N + j + 2 * vec_size);
                auto b4 = vec_type::loadu(b + k * N + j + 3 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a1, b2, r12);
                r13 = vec_type::fmadd(a1, b3, r13);
                r14 = vec_type::fmadd(a1, b4, r14);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 0) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r13));
            vec_type::storeu(c + (i + 0) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r14));
        }
    }

    for (; j + 1 * vec_size < j_end; j += 2 * vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            auto r12 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) + k * M]);
                auto a2 = vec_type::set(a[(i + 1) + k * M]);

                auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);

                r12 = vec_type::fmadd(a1, b2, r12);
                r22 = vec_type::fmadd(a2, b2, r22);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r21));

            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));
        }

        for (; i < M; i++) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) + k * M]);

                auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a1, b2, r12);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));
        }
    }

    for (; j < j_end; j += vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) + k * M]);
                auto a2 = vec_type::set(a[(i + 1) + k * M]);

                auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);
            }

            vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r21));
        }

        if (i < M) {
            auto r11 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[i + k * M]);

                auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
            }

            vec_type::storeu(c + i * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
        }
    }

    for (; j < N; ++j) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r1 = T();
            auto r2 = T();

            for (size_t k = 0; k < K; ++k) {
                r1 += a[(i + 0) + k * M] * b[k * N + j];
                r2 += a[(i + 1) + k * M] * b[k * N + j];
            }

            c[(i + 0) * N + j] = alpha * r1;
            c[(i + 1) * N + j] = alpha * r2;
        }

        if (i < M) {
            auto r1 = T();

            for (size_t k = 0; k < K; ++k) {
                r1 += a[i + k * M] * b[k * N + j];
            }

            c[i * N + j] = alpha * r1;
        }
    }
}

/*!
 * \brief Optimized version of GEMM for assignment of a large
 * Column-Major Matrix - Row Major Matrix to a Row Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_large_kernel_cr_to_r(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    constexpr size_t n_block_size = 128UL;
    constexpr size_t m_block_size = 64UL;
    constexpr size_t k_block_size = 128UL;

    auto alpha_vec = vec_type::set(alpha);

    for (size_t jj = 0; jj < N; jj += n_block_size) {
        const size_t j_end_a = std::min(jj + n_block_size, N);
        const size_t j_end   = j_end_a & size_t(-vec_size);

        for (size_t ii = 0; ii < M; ii += m_block_size) {
            const size_t i_end = std::min(ii + m_block_size, M);

            for (size_t kk = 0; kk < K; kk += k_block_size) {
                const size_t k_end = std::min(kk + k_block_size, K);

                size_t j = jj;

#ifdef __clang__
                for (; j + 3 * vec_size < j_end; j += 4 * vec_size) {
                    size_t i = ii;

                    for (; i + 1 < i_end; i += 2) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + (i + 0) * N + j + 1 * vec_size);
                        auto r13 = vec_type::loadu(c + (i + 0) * N + j + 2 * vec_size);
                        auto r14 = vec_type::loadu(c + (i + 0) * N + j + 3 * vec_size);

                        auto r21 = vec_type::loadu(c + (i + 1) * N + j + 0 * vec_size);
                        auto r22 = vec_type::loadu(c + (i + 1) * N + j + 1 * vec_size);
                        auto r23 = vec_type::loadu(c + (i + 1) * N + j + 2 * vec_size);
                        auto r24 = vec_type::loadu(c + (i + 1) * N + j + 3 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) + k * M]);
                            auto a2 = vec_type::set(a[(i + 1) + k * M]);

                            auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                            auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);
                            auto b3 = vec_type::loadu(b + k * N + j + 2 * vec_size);
                            auto b4 = vec_type::loadu(b + k * N + j + 3 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);
                            r13 = vec_type::fmadd(a1, b3, r13);
                            r14 = vec_type::fmadd(a1, b4, r14);

                            r21 = vec_type::fmadd(a2, b1, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                            r23 = vec_type::fmadd(a2, b3, r23);
                            r24 = vec_type::fmadd(a2, b4, r24);
                        }

                        vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));
                        vec_type::storeu(c + (i + 0) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r13));
                        vec_type::storeu(c + (i + 0) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r14));

                        vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));
                        vec_type::storeu(c + (i + 1) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r23));
                        vec_type::storeu(c + (i + 1) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r24));
                    }

                    if (i < i_end) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + (i + 0) * N + j + 1 * vec_size);
                        auto r13 = vec_type::loadu(c + (i + 0) * N + j + 2 * vec_size);
                        auto r14 = vec_type::loadu(c + (i + 0) * N + j + 3 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) + k * M]);

                            auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                            auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);
                            auto b3 = vec_type::loadu(b + k * N + j + 2 * vec_size);
                            auto b4 = vec_type::loadu(b + k * N + j + 3 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);
                            r13 = vec_type::fmadd(a1, b3, r13);
                            r14 = vec_type::fmadd(a1, b4, r14);
                        }

                        vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));
                        vec_type::storeu(c + (i + 0) * N + j + 2 * vec_size, vec_type::mul(alpha_vec, r13));
                        vec_type::storeu(c + (i + 0) * N + j + 3 * vec_size, vec_type::mul(alpha_vec, r14));
                    }
                }
#endif

                for (; j + vec_size < j_end; j += 2 * vec_size) {
                    size_t i = ii;

                    for (; i + 3 < i_end; i += 4) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + (i + 0) * N + j + 1 * vec_size);

                        auto r21 = vec_type::loadu(c + (i + 1) * N + j + 0 * vec_size);
                        auto r22 = vec_type::loadu(c + (i + 1) * N + j + 1 * vec_size);

                        auto r31 = vec_type::loadu(c + (i + 2) * N + j + 0 * vec_size);
                        auto r32 = vec_type::loadu(c + (i + 2) * N + j + 1 * vec_size);

                        auto r41 = vec_type::loadu(c + (i + 3) * N + j + 0 * vec_size);
                        auto r42 = vec_type::loadu(c + (i + 3) * N + j + 1 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) + k * M]);
                            auto a2 = vec_type::set(a[(i + 1) + k * M]);
                            auto a3 = vec_type::set(a[(i + 2) + k * M]);
                            auto a4 = vec_type::set(a[(i + 3) + k * M]);

                            auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                            auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);

                            r21 = vec_type::fmadd(a2, b1, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);

                            r31 = vec_type::fmadd(a3, b1, r31);
                            r32 = vec_type::fmadd(a3, b2, r32);

                            r41 = vec_type::fmadd(a4, b1, r41);
                            r42 = vec_type::fmadd(a4, b2, r42);
                        }

                        vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));

                        vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));

                        vec_type::storeu(c + (i + 2) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r31));
                        vec_type::storeu(c + (i + 2) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r32));

                        vec_type::storeu(c + (i + 3) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r41));
                        vec_type::storeu(c + (i + 3) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r42));
                    }

                    for (; i + 1 < i_end; i += 2) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + (i + 0) * N + j + 1 * vec_size);

                        auto r21 = vec_type::loadu(c + (i + 1) * N + j + 0 * vec_size);
                        auto r22 = vec_type::loadu(c + (i + 1) * N + j + 1 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) + k * M]);
                            auto a2 = vec_type::set(a[(i + 1) + k * M]);

                            auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                            auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);

                            r21 = vec_type::fmadd(a2, b1, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                        }

                        vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));

                        vec_type::storeu(c + (i + 1) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + (i + 1) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r22));
                    }

                    if (i < i_end) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + (i + 0) * N + j + 1 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) + k * M]);

                            auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);
                            auto b2 = vec_type::loadu(b + k * N + j + 1 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);
                        }

                        vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + (i + 0) * N + j + 1 * vec_size, vec_type::mul(alpha_vec, r12));
                    }
                }

                for (; j < j_end; j += vec_size) {
                    for (size_t i = ii; i < i_end; ++i) {
                        auto r11 = vec_type::loadu(c + (i + 0) * N + j + 0 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::set(a[(i + 0) + k * M]);

                            auto b1 = vec_type::loadu(b + k * N + j + 0 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                        }

                        vec_type::storeu(c + (i + 0) * N + j + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                    }
                }

                for (; j < j_end_a; ++j) {
                    for (size_t i = ii; i < i_end; ++i) {
                        auto r11 = c[(i + 0) * N + j];

                        for (size_t k = kk; k < k_end; ++k) {
                            r11 += a[(i + 0) + k * M] * b[k * N + j];
                        }

                        c[(i + 0) * N + j] = alpha * r11;
                    }
                }
            }
        }
    }
}

/*!
 * \brief Vectorized implementation of column-major matrix - row-major matrix
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
void gemm_cr_to_r(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    if (M * N <= gemm_rr_small_threshold) {
        gemm_small_kernel_cr_to_r<default_vec>(a, b, c, M, N, K, alpha);
    } else {
        direct_fill_n(c, M * N, T(0));
        gemm_large_kernel_cr_to_r<default_vec>(a, b, c, M, N, K, alpha);
    }
}

} //end of namespace etl::impl::vec
