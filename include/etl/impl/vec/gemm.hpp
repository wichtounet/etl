//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// The idea of the GEMM kernels is largely inspired by the kernels in Blaze by
// Klaus Igleberg

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
void gemm_small_kernel_rr(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    size_t j = 0;

    for (; j + vec_size * 8 - 1 < N; j += vec_size * 8) {
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

    for (; j + (4 * vec_size) - 1 < N; j += 4 * vec_size) {
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
                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
                auto b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
                auto b4 = vec_type::loadu(b + k * N + j + vec_size * 3);

                auto a1 = vec_type::set(a[(i + 0) * K + k]);
                auto a2 = vec_type::set(a[(i + 1) * K + k]);

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
            vec_type::storeu(c + (i+1) * N + j + 0 * vec_size, r12);

            vec_type::storeu(c + (i+0) * N + j + 1 * vec_size, r21);
            vec_type::storeu(c + (i+1) * N + j + 1 * vec_size, r22);

            vec_type::storeu(c + (i+0) * N + j + 2 * vec_size, r31);
            vec_type::storeu(c + (i+1) * N + j + 2 * vec_size, r32);

            vec_type::storeu(c + (i+0) * N + j + 3 * vec_size, r41);
            vec_type::storeu(c + (i+1) * N + j + 3 * vec_size, r42);
        }

        if (i < M) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();
            auto r31 = vec_type::template zero<T>();
            auto r41 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);
                auto b3 = vec_type::loadu(b + k * N + j + vec_size * 2);
                auto b4 = vec_type::loadu(b + k * N + j + vec_size * 3);

                auto a1 = vec_type::set(a[(i + 0) * K + k]);

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

    for (; j + (2 * vec_size) - 1 < N; j += 2 * vec_size) {
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
                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

                auto a1 = vec_type::set(a[(i + 0) * K + k]);
                auto a2 = vec_type::set(a[(i + 1) * K + k]);
                auto a3 = vec_type::set(a[(i + 2) * K + k]);
                auto a4 = vec_type::set(a[(i + 3) * K + k]);

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
            vec_type::storeu(c + (i+1) * N + j + 0 * vec_size, r12);
            vec_type::storeu(c + (i+2) * N + j + 0 * vec_size, r13);
            vec_type::storeu(c + (i+3) * N + j + 0 * vec_size, r14);

            vec_type::storeu(c + (i+0) * N + j + 1 * vec_size, r21);
            vec_type::storeu(c + (i+1) * N + j + 1 * vec_size, r22);
            vec_type::storeu(c + (i+2) * N + j + 1 * vec_size, r23);
            vec_type::storeu(c + (i+3) * N + j + 1 * vec_size, r24);
        }

        for (; i + 1 < M; i += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto b1 = vec_type::loadu(b + k * N + j + vec_size * 0);
                auto b2 = vec_type::loadu(b + k * N + j + vec_size * 1);

                auto a1 = vec_type::set(a[(i + 0) * K + k]);
                auto a2 = vec_type::set(a[(i + 1) * K + k]);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);
            }

            vec_type::storeu(c + (i+0) * N + j + 0 * vec_size, r11);
            vec_type::storeu(c + (i+1) * N + j + 0 * vec_size, r12);

            vec_type::storeu(c + (i+0) * N + j + 1 * vec_size, r21);
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

    for (; j + vec_size - 1 < N; j += vec_size) {
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
void gemm_large_kernel_rr(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T beta) {
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
 * \brief Optimized version of GEMM for row major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemm(A&& a, B&& b, C&& c) {
    using T = value_t<A>;

    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    const size_t M = etl::rows(a);
    const size_t N = etl::columns(b);
    const size_t K = etl::columns(a);

    if(etl::size(b) <= gemm_rr_small_threshold){
        gemm_small_kernel_rr<default_vec>(a.memory_start(), b.memory_start(), c.memory_start(), M, N, K);
    } else {
        gemm_large_kernel_rr<default_vec>(a.memory_start(), b.memory_start(), c.memory_start(), M, N, K, T(0));
    }

    c.invalidate_gpu();
}

/*!
 * \brief Optimized version of small GEMM for column major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_small_kernel_cc(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    size_t i = 0;

    for (; i + 4 * vec_size - 1 < M; i += 4 * vec_size) {
        size_t j = 0;

        for (; (j + 2UL) <= N; j += 2UL) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            auto r31 = vec_type::template zero<T>();
            auto r32 = vec_type::template zero<T>();

            auto r41 = vec_type::template zero<T>();
            auto r42 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a11 = vec_type::loadu(a + (i + vec_size * 0) + k * M);
                auto a21 = vec_type::loadu(a + (i + vec_size * 1) + k * M);
                auto a31 = vec_type::loadu(a + (i + vec_size * 2) + k * M);
                auto a41 = vec_type::loadu(a + (i + vec_size * 3) + k * M);

                auto b1 = vec_type::set(b[k + (j + 0) * K]);
                auto b2 = vec_type::set(b[k + (j + 1) * K]);

                r11 = vec_type::fmadd(a11, b1, r11);
                r12 = vec_type::fmadd(a11, b2, r12);

                r21 = vec_type::fmadd(a21, b1, r21);
                r22 = vec_type::fmadd(a21, b2, r22);

                r31 = vec_type::fmadd(a31, b1, r31);
                r32 = vec_type::fmadd(a31, b2, r32);

                r41 = vec_type::fmadd(a41, b1, r41);
                r42 = vec_type::fmadd(a41, b2, r42);
            }

            vec_type::storeu(c + (i + vec_size * 0) + (j + 0) * M, r11);
            vec_type::storeu(c + (i + vec_size * 0) + (j + 1) * M, r12);

            vec_type::storeu(c + (i + vec_size * 1) + (j + 0) * M, r21);
            vec_type::storeu(c + (i + vec_size * 1) + (j + 1) * M, r22);

            vec_type::storeu(c + (i + vec_size * 2) + (j + 0) * M, r31);
            vec_type::storeu(c + (i + vec_size * 2) + (j + 1) * M, r32);

            vec_type::storeu(c + (i + vec_size * 3) + (j + 0) * M, r41);
            vec_type::storeu(c + (i + vec_size * 3) + (j + 1) * M, r42);
        }

        if (j < N) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();
            auto r31 = vec_type::template zero<T>();
            auto r41 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a11 = vec_type::loadu(a + i + vec_size * 0 + k * M);
                auto a21 = vec_type::loadu(a + i + vec_size * 1 + k * M);
                auto a31 = vec_type::loadu(a + i + vec_size * 2 + k * M);
                auto a41 = vec_type::loadu(a + i + vec_size * 3 + k * M);

                auto b1 = vec_type::set(b[k + j * K]);

                r11 = vec_type::fmadd(a11, b1, r11);
                r21 = vec_type::fmadd(a21, b1, r21);
                r31 = vec_type::fmadd(a31, b1, r31);
                r41 = vec_type::fmadd(a41, b1, r41);
            }

            vec_type::storeu(c + i + vec_size * 0 + j * M, r11);
            vec_type::storeu(c + i + vec_size * 1 + j * M, r21);
            vec_type::storeu(c + i + vec_size * 2 + j * M, r31);
            vec_type::storeu(c + i + vec_size * 3 + j * M, r41);
        }
    }

    for (; i + 2 * vec_size - 1 < M; i += 2 * vec_size) {
        size_t j = 0;

        for (; (j + 2UL) <= N; j += 2UL) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a11 = vec_type::loadu(a + (i + vec_size * 0) + k * M);
                auto a21 = vec_type::loadu(a + (i + vec_size * 1) + k * M);

                auto b1 = vec_type::set(b[k + (j + 0) * K]);
                auto b2 = vec_type::set(b[k + (j + 1) * K]);

                r11 = vec_type::fmadd(a11, b1, r11);
                r12 = vec_type::fmadd(a11, b2, r12);

                r21 = vec_type::fmadd(a21, b1, r21);
                r22 = vec_type::fmadd(a21, b2, r22);
            }

            vec_type::storeu(c + (i + vec_size * 0) + (j + 0) * M, r11);
            vec_type::storeu(c + (i + vec_size * 0) + (j + 1) * M, r12);

            vec_type::storeu(c + (i + vec_size * 1) + (j + 0) * M, r21);
            vec_type::storeu(c + (i + vec_size * 1) + (j + 1) * M, r22);
        }

        if (j < N) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a11 = vec_type::loadu(a + i + vec_size * 0 + k * M);
                auto a21 = vec_type::loadu(a + i + vec_size * 1 + k * M);

                auto b1 = vec_type::set(b[k + j * K]);

                r11 = vec_type::fmadd(a11, b1, r11);
                r21 = vec_type::fmadd(a21, b1, r21);
            }

            vec_type::storeu(c + i + vec_size * 0 + j * M, r11);
            vec_type::storeu(c + i + vec_size * 1 + j * M, r21);
        }
    }

    for (; i + vec_size - 1 < M; i += vec_size) {
        size_t j = 0;

        for (; (j + 2UL) <= N; j += 2UL) {
            auto r1 = vec_type::template zero<T>();
            auto r2 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M);

                auto b1 = vec_type::set(b[k + (j + 0) * K]);
                auto b2 = vec_type::set(b[k + (j + 1) * K]);

                r1 = vec_type::fmadd(a1, b1, r1);
                r2 = vec_type::fmadd(a1, b2, r2);
            }

            vec_type::storeu(c + i + (j + 0) * M, r1);
            vec_type::storeu(c + i + (j + 1) * M, r2);
        }

        if (j < N) {
            auto r1 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M);

                auto b1 = vec_type::set(b[k + j * K]);

                r1 = vec_type::fmadd(a1, b1, r1);
            }

            vec_type::storeu(c + i + j * M, r1);
        }
    }

    for (; i < M; ++i) {
        size_t j = 0;

        for (; j + 1 < N; j += 2) {
            T value1(0);
            T value2(0);

            for (size_t k = 0; k < K; ++k) {
                value1 += a[i + k * M] * b[k + (j + 0) * K];
                value2 += a[i + k * M] * b[k + (j + 1) * K];
            }

            c[i + (j + 0) * M] = value1;
            c[i + (j + 1) * M] = value2;
        }

        if (j < N) {
            T value(0);

            for (size_t k = 0; k < K; ++k) {
                value += a[i + k * M] * b[k + j * K];
            }

            c[i + j * M] = value;
        }
    }
}

/*!
 * \brief Optimized version of large GEMM for column major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_large_kernel_cc(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t m_block_size = 128;
    const size_t n_block_size = 64;
    const size_t k_block_size = 128;

    for (size_t block_i = 0; block_i < M; block_i += m_block_size) {
        const size_t i_end = std::min(block_i + m_block_size, M);

        for (size_t block_j = 0; block_j < N; block_j += n_block_size) {
            const size_t j_end = std::min(block_j + n_block_size, N);

            for (size_t block_k = 0; block_k < K; block_k += k_block_size) {
                const size_t k_end = std::min(block_k + k_block_size, K);

                size_t i = block_i;

                // 4x unrolled vectorized inner loop
                for (; i + 4 * vec_size - 1 < i_end; i += 4 * vec_size) {

                    size_t j = block_j;

                    for (; j + 1 < j_end; j += 2) {
                        auto r11 = vec_type::loadu(c + i + 0 * vec_size + (j + 0) * M);
                        auto r12 = vec_type::loadu(c + i + 1 * vec_size + (j + 0) * M);
                        auto r13 = vec_type::loadu(c + i + 2 * vec_size + (j + 0) * M);
                        auto r14 = vec_type::loadu(c + i + 3 * vec_size + (j + 0) * M);

                        auto r21 = vec_type::loadu(c + i + 0 * vec_size + (j + 1) * M);
                        auto r22 = vec_type::loadu(c + i + 1 * vec_size + (j + 1) * M);
                        auto r23 = vec_type::loadu(c + i + 2 * vec_size + (j + 1) * M);
                        auto r24 = vec_type::loadu(c + i + 3 * vec_size + (j + 1) * M);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + 0 * vec_size + k * M);
                            auto a2 = vec_type::loadu(a + i + 1 * vec_size + k * M);
                            auto a3 = vec_type::loadu(a + i + 2 * vec_size + k * M);
                            auto a4 = vec_type::loadu(a + i + 3 * vec_size + k * M);

                            auto b1 = vec_type::set(b[k + (j + 0) * K]);
                            auto b2 = vec_type::set(b[k + (j + 1) * K]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);
                            r13 = vec_type::fmadd(a3, b1, r13);
                            r14 = vec_type::fmadd(a4, b1, r14);

                            r21 = vec_type::fmadd(a1, b2, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                            r23 = vec_type::fmadd(a3, b2, r23);
                            r24 = vec_type::fmadd(a4, b2, r24);
                        }

                        vec_type::storeu(c + i + 0 * vec_size + (j + 0) * M, r11);
                        vec_type::storeu(c + i + 1 * vec_size + (j + 0) * M, r12);
                        vec_type::storeu(c + i + 2 * vec_size + (j + 0) * M, r13);
                        vec_type::storeu(c + i + 3 * vec_size + (j + 0) * M, r14);

                        vec_type::storeu(c + i + 0 * vec_size + (j + 1) * M, r21);
                        vec_type::storeu(c + i + 1 * vec_size + (j + 1) * M, r22);
                        vec_type::storeu(c + i + 2 * vec_size + (j + 1) * M, r23);
                        vec_type::storeu(c + i + 3 * vec_size + (j + 1) * M, r24);
                    }

                    for (; j < j_end; ++j) {
                        auto r1 = vec_type::loadu(c + i + 0 * vec_size + j * M);
                        auto r2 = vec_type::loadu(c + i + 1 * vec_size + j * M);
                        auto r3 = vec_type::loadu(c + i + 2 * vec_size + j * M);
                        auto r4 = vec_type::loadu(c + i + 3 * vec_size + j * M);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + 0 * vec_size + k * M);
                            auto a2 = vec_type::loadu(a + i + 1 * vec_size + k * M);
                            auto a3 = vec_type::loadu(a + i + 2 * vec_size + k * M);
                            auto a4 = vec_type::loadu(a + i + 3 * vec_size + k * M);

                            auto b1 = vec_type::set(b[k + j * K]);

                            r1 = vec_type::fmadd(a1, b1, r1);
                            r2 = vec_type::fmadd(a2, b1, r2);
                            r3 = vec_type::fmadd(a3, b1, r3);
                            r4 = vec_type::fmadd(a4, b1, r4);
                        }

                        vec_type::storeu(c + i + 0 * vec_size + j * M, r1);
                        vec_type::storeu(c + i + 1 * vec_size + j * M, r2);
                        vec_type::storeu(c + i + 2 * vec_size + j * M, r3);
                        vec_type::storeu(c + i + 3 * vec_size + j * M, r4);
                    }
                }

                // 2x unrolled vectorized inner loop
                for (; i + 2 * vec_size - 1 < i_end; i += 2 * vec_size) {

                    size_t j = block_j;

                    for (; j + 3 < j_end; j += 4) {
                        auto r11 = vec_type::loadu(c + i * vec_size + 0 + (j + 0) * M);
                        auto r12 = vec_type::loadu(c + i * vec_size + 1 + (j + 0) * M);

                        auto r21 = vec_type::loadu(c + i * vec_size + 0 + (j + 1) * M);
                        auto r22 = vec_type::loadu(c + i * vec_size + 1 + (j + 1) * M);

                        auto r31 = vec_type::loadu(c + i * vec_size + 0 + (j + 2) * M);
                        auto r32 = vec_type::loadu(c + i * vec_size + 1 + (j + 3) * M);

                        auto r41 = vec_type::loadu(c + i * vec_size + 0 + (j + 4) * M);
                        auto r42 = vec_type::loadu(c + i * vec_size + 1 + (j + 4) * M);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + 0 * vec_size + k * M);
                            auto a2 = vec_type::loadu(a + i + 1 * vec_size + k * M);

                            auto b1 = vec_type::set(b[k + (j + 0) * K]);
                            auto b2 = vec_type::set(b[k + (j + 1) * K]);
                            auto b3 = vec_type::set(b[k + (j + 2) * K]);
                            auto b4 = vec_type::set(b[k + (j + 3) * K]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);

                            r21 = vec_type::fmadd(a1, b2, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);

                            r31 = vec_type::fmadd(a1, b3, r31);
                            r32 = vec_type::fmadd(a2, b3, r32);

                            r41 = vec_type::fmadd(a1, b4, r41);
                            r42 = vec_type::fmadd(a2, b4, r42);
                        }

                        vec_type::storeu(c + i + 0 * vec_size + (j + 0) * M, r11);
                        vec_type::storeu(c + i + 1 * vec_size + (j + 0) * M, r12);

                        vec_type::storeu(c + i + 0 * vec_size + (j + 1) * M, r21);
                        vec_type::storeu(c + i + 1 * vec_size + (j + 1) * M, r22);

                        vec_type::storeu(c + i + 0 * vec_size + (j + 2) * M, r31);
                        vec_type::storeu(c + i + 1 * vec_size + (j + 2) * M, r32);

                        vec_type::storeu(c + i + 0 * vec_size + (j + 3) * M, r41);
                        vec_type::storeu(c + i + 1 * vec_size + (j + 3) * M, r42);
                    }

                    for (; j + 1 < j_end; j += 2) {
                        auto r11 = vec_type::loadu(c + i * vec_size + 0 + (j + 0) * M);
                        auto r12 = vec_type::loadu(c + i * vec_size + 1 + (j + 0) * M);

                        auto r21 = vec_type::loadu(c + i * vec_size + 0 + (j + 1) * M);
                        auto r22 = vec_type::loadu(c + i * vec_size + 1 + (j + 1) * M);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + 0 * vec_size + k * M);
                            auto a2 = vec_type::loadu(a + i + 1 * vec_size + k * M);

                            auto b1 = vec_type::set(b[k + (j + 0) * K]);
                            auto b2 = vec_type::set(b[k + (j + 1) * K]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);

                            r21 = vec_type::fmadd(a1, b2, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                        }

                        vec_type::storeu(c + i + 0 * vec_size + (j + 0) * M, r11);
                        vec_type::storeu(c + i + 1 * vec_size + (j + 0) * M, r12);

                        vec_type::storeu(c + i + 0 * vec_size + (j + 1) * M, r21);
                        vec_type::storeu(c + i + 1 * vec_size + (j + 1) * M, r22);
                    }

                    for (; j < j_end; ++j) {
                        auto r1 = vec_type::loadu(c + i + 0 * vec_size + j * M);
                        auto r2 = vec_type::loadu(c + i + 1 * vec_size + j * M);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + 0 * vec_size + k * M);
                            auto a2 = vec_type::loadu(a + i + 1 * vec_size + k * M);

                            auto b1 = vec_type::set(b[k + j * K]);

                            r1 = vec_type::fmadd(a1, b1, r1);
                            r2 = vec_type::fmadd(a2, b1, r2);
                        }

                        vec_type::storeu(c + i + 0 * vec_size + j * M, r1);
                        vec_type::storeu(c + i + 1 * vec_size + j * M, r2);
                    }
                }

                // Vectorized inner loop
                for (; i + vec_size - 1 < i_end; i += vec_size) {
                    for (size_t j = block_j; j < j_end; ++j) {
                        auto r1 = vec_type::loadu(c + i + j * M);

                        for (size_t k = block_k; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + k * M);
                            auto b1 = vec_type::set(b[k + j * K]);

                            r1 = vec_type::fmadd(a1, b1, r1);
                        }

                        vec_type::storeu(c + i + j * M, r1);
                    }
                }

                // Remainder inner loop
                for(; i < i_end; ++i){
                    for(size_t j = block_j; j < j_end; ++j){
                        auto x = c[i + j * M];

                        for(size_t k = block_k; k < k_end; ++k){
                            x += a[i + k * M] * b[k + j * K];
                        }

                        c[i + j * M] = x;
                    }
                }
            }
        }
    }
}

/*!
 * \brief Unoptimized version of GEMM for column major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename A, typename B, typename C, cpp_disable_if((all_row_major<A, B, C>::value))>
void gemm(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    const size_t M = etl::rows(a);
    const size_t N = etl::columns(b);
    const size_t K = etl::columns(a);

    if(etl::size(c) <= gemm_cc_small_threshold){
        gemm_small_kernel_cc<default_vec>(a.memory_start(), b.memory_start(), c.memory_start(), M, N, K);
    } else {
        c = 0;
        gemm_large_kernel_cc<default_vec>(a.memory_start(), b.memory_start(), c.memory_start(), M, N, K);
    }

    c.invalidate_gpu();
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
