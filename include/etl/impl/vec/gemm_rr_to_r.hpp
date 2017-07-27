//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace etl {

namespace impl {

namespace vec {

/*!
 * \brief Optimized version of small GEMM for row major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_small_kernel_rr_to_r(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const auto j_end = N & (size_t(-vec_size));

    size_t j = 0;

    for (; j + vec_size * 7 < j_end; j += vec_size * 8) {
        for (size_t i = 0; i < M; ++i) {
            auto r1 = vec_type::template zero<T>();
            auto r2 = vec_type::template zero<T>();
            auto r3 = vec_type::template zero<T>();
            auto r4 = vec_type::template zero<T>();
            auto r5 = vec_type::template zero<T>();
            auto r6 = vec_type::template zero<T>();
            auto r7 = vec_type::template zero<T>();
            auto r8 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[i * K + k]);

                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
                auto b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
                auto b4 = vec_type::loadu(b + k * N + j + vec_size * 3);
                auto b5 = vec_type::loadu(b + k * N + j + vec_size * 4);
                auto b6 = vec_type::loadu(b + k * N + j + vec_size * 5);
                auto b7 = vec_type::loadu(b + k * N + j + vec_size * 6);
                auto b8 = vec_type::loadu(b + k * N + j + vec_size * 7);

                r1 = vec_type::fmadd(a1, b1, r1);
                r2 = vec_type::fmadd(a1, b2, r2);
                r3 = vec_type::fmadd(a1, b3, r3);
                r4 = vec_type::fmadd(a1, b4, r4);
                r5 = vec_type::fmadd(a1, b5, r5);
                r6 = vec_type::fmadd(a1, b6, r6);
                r7 = vec_type::fmadd(a1, b7, r7);
                r8 = vec_type::fmadd(a1, b8, r8);
            }

            vec_type::storeu(c + i * N + j + 0 * vec_size, r1);
            vec_type::storeu(c + i * N + j + 1 * vec_size, r2);
            vec_type::storeu(c + i * N + j + 2 * vec_size, r3);
            vec_type::storeu(c + i * N + j + 3 * vec_size, r4);
            vec_type::storeu(c + i * N + j + 4 * vec_size, r5);
            vec_type::storeu(c + i * N + j + 5 * vec_size, r6);
            vec_type::storeu(c + i * N + j + 6 * vec_size, r7);
            vec_type::storeu(c + i * N + j + 7 * vec_size, r8);
        }
    }

    for (; j + vec_size * 3 < j_end; j += 4 * vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2){
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            auto r31 = vec_type::template zero<T>();
            auto r32 = vec_type::template zero<T>();

            auto r41 = vec_type::template zero<T>();
            auto r42 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) * K + k]);
                auto a2 = vec_type::set(a[(i + 1) * K + k]);

                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
                auto b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
                auto b4 = vec_type::loadu(b + k * N + j + vec_size * 3);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);

                r31 = vec_type::fmadd(a1, b3, r31);
                r32 = vec_type::fmadd(a2, b3, r32);

                r41 = vec_type::fmadd(a1, b4, r41);
                r42 = vec_type::fmadd(a2, b4, r42);
            }

            vec_type::storeu(c + (i+0) * N + j + 0 * vec_size, r11);
            vec_type::storeu(c + (i+0) * N + j + 1 * vec_size, r21);
            vec_type::storeu(c + (i+0) * N + j + 2 * vec_size, r31);
            vec_type::storeu(c + (i+0) * N + j + 3 * vec_size, r41);

            vec_type::storeu(c + (i+1) * N + j + 0 * vec_size, r12);
            vec_type::storeu(c + (i+1) * N + j + 1 * vec_size, r22);
            vec_type::storeu(c + (i+1) * N + j + 2 * vec_size, r32);
            vec_type::storeu(c + (i+1) * N + j + 3 * vec_size, r42);
        }

        if (i < M) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();
            auto r31 = vec_type::template zero<T>();
            auto r41 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) * K + k]);

                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
                auto b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
                auto b4 = vec_type::loadu(b + k * N + j + vec_size * 3);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a1, b2, r21);
                r31 = vec_type::fmadd(a1, b3, r31);
                r41 = vec_type::fmadd(a1, b4, r41);
            }

            vec_type::storeu(c + i * N + j + 0 * vec_size, r11);
            vec_type::storeu(c + i * N + j + 1 * vec_size, r21);
            vec_type::storeu(c + i * N + j + 2 * vec_size, r31);
            vec_type::storeu(c + i * N + j + 3 * vec_size, r41);
        }
    }

    for (; j + vec_size < j_end; j += 2 * vec_size) {
        size_t i = 0;

        for (; i + 3 < M; i += 4) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();
            auto r13 = vec_type::template zero<T>();
            auto r14 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();
            auto r23 = vec_type::template zero<T>();
            auto r24 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) * K + k]);
                auto a2 = vec_type::set(a[(i + 1) * K + k]);
                auto a3 = vec_type::set(a[(i + 2) * K + k]);
                auto a4 = vec_type::set(a[(i + 3) * K + k]);

                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);
                r13 = vec_type::fmadd(a3, b1, r13);
                r14 = vec_type::fmadd(a4, b1, r14);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);
                r23 = vec_type::fmadd(a3, b2, r23);
                r24 = vec_type::fmadd(a4, b2, r24);
            }

            vec_type::storeu(c + (i+0) * N + j + 0 * vec_size, r11);
            vec_type::storeu(c + (i+0) * N + j + 1 * vec_size, r21);

            vec_type::storeu(c + (i+1) * N + j + 0 * vec_size, r12);
            vec_type::storeu(c + (i+1) * N + j + 1 * vec_size, r22);

            vec_type::storeu(c + (i+2) * N + j + 0 * vec_size, r13);
            vec_type::storeu(c + (i+2) * N + j + 1 * vec_size, r23);

            vec_type::storeu(c + (i+3) * N + j + 0 * vec_size, r14);
            vec_type::storeu(c + (i+3) * N + j + 1 * vec_size, r24);
        }

        for (; i + 1 < M; i += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::set(a[(i + 0) * K + k]);
                auto a2 = vec_type::set(a[(i + 1) * K + k]);

                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);
            }

            vec_type::storeu(c + (i+0) * N + j + 0 * vec_size, r11);
            vec_type::storeu(c + (i+0) * N + j + 1 * vec_size, r21);

            vec_type::storeu(c + (i+1) * N + j + 0 * vec_size, r12);
            vec_type::storeu(c + (i+1) * N + j + 1 * vec_size, r22);
        }

        if (i < M) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

                auto a1 = vec_type::set(a[(i + 0) * K + k]);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a1, b2, r21);
            }

            vec_type::storeu(c + (i+0) * N + j + 0 * vec_size, r11);
            vec_type::storeu(c + (i+0) * N + j + 1 * vec_size, r21);
        }
    }

    for (; j < j_end; j += vec_size) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r1 = vec_type::template zero<T>();
            auto r2 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);

                auto a1 = vec_type::set(a[(i + 0) * K + k]);
                auto a2 = vec_type::set(a[(i + 1) * K + k]);

                r1 = vec_type::fmadd(a1, b1, r1);
                r2 = vec_type::fmadd(a2, b1, r2);
            }

            vec_type::storeu(c + (i+0) * N + j, r1);
            vec_type::storeu(c + (i+1) * N + j, r2);
            }

        if (i < M) {
            auto r1 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);

                auto a1 = vec_type::set(a[(i + 0) * K + k]);

                r1 = vec_type::fmadd(a1, b1, r1);
            }

            vec_type::storeu(c + (i+0) * N + j, r1);
        }
    }

    for (; j + 1 < N; j += 2) {
        const size_t j1 = j + 0;
        const size_t j2 = j + 1;

        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r11 = T();
            auto r12 = T();
            auto r21 = T();
            auto r22 = T();

            for (size_t k = 0; k < K; ++k) {
                r11 += a[(i + 0) * K + k] * b[k * N + j1];
                r21 += a[(i + 0) * K + k] * b[k * N + j2];
                r12 += a[(i + 1) * K + k] * b[k * N + j1];
                r22 += a[(i + 1) * K + k] * b[k * N + j2];
            }

            c[(i + 0) * N + j1] = r11;
            c[(i + 0) * N + j2] = r21;
            c[(i + 1) * N + j1] = r12;
            c[(i + 1) * N + j2] = r22;
        }

        if (i < M) {
            auto r1 = T();
            auto r2 = T();

            for (size_t k = 0; k < K; ++k) {
                r1 += a[i * K + k] * b[k * N + j1];
                r2 += a[i * K + k] * b[k * N + j2];
            }

            c[i * N + j1] = r1;
            c[i * N + j2] = r2;
        }
    }

    if (j < N) {
        size_t i = 0;

        for (; i + 1 < M; i += 2) {
            auto r1 = T();
            auto r2 = T();

            for (size_t k = 0; k < K; ++k) {
                r1 += a[(i + 0) * K + k] * b[k * N + j];
                r2 += a[(i + 1) * K + k] * b[k * N + j];
            }

            c[(i + 0) * N + j] = r1;
            c[(i + 1) * N + j] = r2;
        }

        if (i < M) {
            auto r1 = T();

            for (size_t k = 0; k < K; ++k) {
                r1 += a[i * K + k] * b[k * N + j];
            }

            c[i * N + j] = r1;
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
void gemm_large_kernel_rr_to_r(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T beta) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t n_block_size = 128;
    const size_t m_block_size = 64;
    const size_t k_block_size = 128;

    // TODO Ideally, it should be possible to split the workload in thread easily
    // Unfortunately, adding a lambda around the following code makes it twice
    // slower, for some reason

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

                        vec_type::storeu(c + (i + 0) * N + j, r11);
                        vec_type::storeu(c + (i + 0) * N + j1, r12);
                        vec_type::storeu(c + (i + 0) * N + j2, r13);
                        vec_type::storeu(c + (i + 0) * N + j3, r14);
                        vec_type::storeu(c + (i + 1) * N + j, r21);
                        vec_type::storeu(c + (i + 1) * N + j1, r22);
                        vec_type::storeu(c + (i + 1) * N + j2, r23);
                        vec_type::storeu(c + (i + 1) * N + j3, r24);
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

                        vec_type::storeu(c + (i + 0) * N + j, r1);
                        vec_type::storeu(c + (i + 0) * N + j1, r2);
                        vec_type::storeu(c + (i + 0) * N + j2, r3);
                        vec_type::storeu(c + (i + 0) * N + j3, r4);
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

                        vec_type::storeu(c + (i + 0) * N + j, r11);
                        vec_type::storeu(c + (i + 0) * N + j1, r12);
                        vec_type::storeu(c + (i + 1) * N + j, r21);
                        vec_type::storeu(c + (i + 1) * N + j1, r22);
                        vec_type::storeu(c + (i + 2) * N + j, r31);
                        vec_type::storeu(c + (i + 2) * N + j1, r32);
                        vec_type::storeu(c + (i + 3) * N + j, r41);
                        vec_type::storeu(c + (i + 3) * N + j1, r42);
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

                        vec_type::storeu(c + (i + 0) * N + j, r11);
                        vec_type::storeu(c + (i + 0) * N + j1, r12);
                        vec_type::storeu(c + (i + 1) * N + j, r21);
                        vec_type::storeu(c + (i + 1) * N + j1, r22);
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

                        vec_type::storeu(c + (i + 0) * N + j, r1);
                        vec_type::storeu(c + (i + 0) * N + j1, r2);
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

                        vec_type::storeu(c + (i + 0) * N + j, r1);
                    }
                }

                for (; j < j_end; ++j) {
                    for (size_t i = block_i; i < i_end; ++i) {
                        auto value = c[i * N + j];

                        for (size_t k = block_k; k < k_end; ++k) {
                            value += a[i * K + k] * b[k * N + j];
                        }

                        c[i * N + j] = value;
                    }
                }
            }
        }
    }
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
void gemm_rr_to_r(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    // Dispatch to the best kernel

    if(K * N  <= gemm_rr_small_threshold){
        gemm_small_kernel_rr_to_r<default_vec>(a, b, c, M, N, K);
    } else {
        gemm_large_kernel_rr_to_r<default_vec>(a, b, c, M, N, K, T(0));
    }
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
