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

template<typename T>
struct gemm_config;

template<>
struct gemm_config <float> {
    static constexpr size_t MC = 768;
    static constexpr size_t KC = 384;
    static constexpr size_t NC = 4096;

    static constexpr size_t MR = 8;
    static constexpr size_t NR = 4;
};

template<>
struct gemm_config <double> {
    static constexpr size_t MC = 384;
    static constexpr size_t KC = 384;
    static constexpr size_t NC = 4096;

    static constexpr size_t MR = 4;
    static constexpr size_t NR = 4;
};

template<typename T>
void pack_A(size_t mc, size_t kc, const T *A, size_t incRowA, size_t incColA, T* _A){
    static constexpr const size_t MR = gemm_config<T>::MR;

    const size_t mp  = mc / MR;
    const size_t _mr = mc % MR;

    for (size_t k = 0; k < mp; ++k) {
        for (size_t j = 0; j < kc; ++j) {
            for (size_t i = 0; i < MR; ++i) {
                _A[k * kc * MR + j * MR + i] = A[k * MR * incRowA + j * incColA + i * incRowA];
            }
        }
    }

    if (_mr > 0) {
        const size_t k = mp;

        for (size_t j = 0; j < kc; ++j) {
            for (size_t i = 0; i < _mr; ++i) {
                _A[k * kc * MR + j * MR + i] = A[k * MR * incRowA + j * incColA + i * incRowA];
            }

            for (size_t i = _mr; i < MR; ++i) {
                _A[k * kc * MR + j * MR + i] = 0.0;
            }
        }
    }
}

template <typename T>
void pack_B(size_t kc, size_t nc, const T* B, size_t incRowB, size_t incColB, T* _B) {
    static constexpr const size_t NR = gemm_config<T>::NR;

    const size_t np  = nc / NR;
    const size_t _nr = nc % NR;

    for (size_t k = 0; k < np; ++k) {
        for (size_t i = 0; i < kc; ++i) {
            for (size_t j = 0; j < NR; ++j) {
                _B[k * kc * NR + i * NR + j] = B[k * NR * incColB + i * incRowB + j * incColB];
            }
        }
    }

    if (_nr > 0) {
        const size_t k = np;

        for (size_t i = 0; i < kc; ++i) {
            for (size_t j = 0; j < _nr; ++j) {
                _B[k * kc * NR + i * NR + j] = B[k * NR * incColB + i * incRowB + j * incColB];
            }

            for (size_t j = _nr; j < NR; ++j) {
                _B[k * kc * NR + i * NR + j] = 0.0;
            }
        }
    }
}

template<typename T>
void dgeaxpy(size_t m, size_t n, T alpha, const T* X, size_t incRowX, size_t incColX, T* Y, size_t incRowY, size_t incColY) {
    if (alpha != 1.0) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                Y[i * incRowY + j * incColY] += alpha * X[i * incRowX + j * incColX];
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                Y[i * incRowY + j * incColY] += X[i * incRowX + j * incColX];
            }
        }
    }
}

template <typename T>
void dgescal(size_t m, size_t n, T alpha, T* X, size_t incRowX, size_t incColX) {
    if (alpha != 0.0) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                X[i * incRowX + j * incColX] *= alpha;
            }
        }
    } else {
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                X[i * incRowX + j * incColX] = 0.0;
            }
        }
    }
}

// General version
template <typename V, typename T, cpp_disable_if(std::is_same<float, T>::value && vector_mode == vector_mode_t::AVX)>
void gemm_pico_kernel(size_t kc, const T* A, const T* B, T* AB) {
    static constexpr const size_t MR = gemm_config<T>::MR;
    static constexpr const size_t NR = gemm_config<T>::NR;

    for (size_t l = 0; l < MR * NR; ++l) {
        AB[l] = 0;
    }

    for (size_t l = 0; l < kc; ++l) {
        for (size_t j = 0; j < NR; ++j) {
            for (size_t i = 0; i < MR; ++i) {
                AB[j * MR + i] += A[l * MR + i] * B[l * NR + j];
            }
        }
    }
}

// AVX/float version
template <typename V, typename T, cpp_enable_if(std::is_same<float, T>::value && vector_mode == vector_mode_t::AVX)>
void gemm_pico_kernel(size_t kc, const T* ETL_RESTRICT A, const T* ETL_RESTRICT B, T* ETL_RESTRICT AB) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    static constexpr const size_t MR = gemm_config<T>::MR;
    static constexpr const size_t NR = gemm_config<T>::NR;

    static_assert(NR == 4, "Invalid algorithm selection");
    static_assert(vec_size == MR, "Invalid algorith selection");

    auto AB1 = vec_type::template zero<T>();
    auto AB2 = vec_type::template zero<T>();
    auto AB3 = vec_type::template zero<T>();
    auto AB4 = vec_type::template zero<T>();

    size_t l = 0;

    for (; l + 3 < kc; l += 4) {
        auto A1 = vec_type::loadu(A + (l + 0) * MR);

        auto B11 = vec_type::set(B[(l + 0) * NR + 0]);
        auto B12 = vec_type::set(B[(l + 0) * NR + 1]);
        auto B13 = vec_type::set(B[(l + 0) * NR + 2]);
        auto B14 = vec_type::set(B[(l + 0) * NR + 3]);

        AB1 = vec_type::fmadd(A1, B11, AB1);
        AB2 = vec_type::fmadd(A1, B12, AB2);
        AB3 = vec_type::fmadd(A1, B13, AB3);
        AB4 = vec_type::fmadd(A1, B14, AB4);

        auto A2 = vec_type::loadu(A + (l + 1) * MR);

        auto B21 = vec_type::set(B[(l + 1) * NR + 0]);
        auto B22 = vec_type::set(B[(l + 1) * NR + 1]);
        auto B23 = vec_type::set(B[(l + 1) * NR + 2]);
        auto B24 = vec_type::set(B[(l + 1) * NR + 3]);

        AB1 = vec_type::fmadd(A2, B21, AB1);
        AB2 = vec_type::fmadd(A2, B22, AB2);
        AB3 = vec_type::fmadd(A2, B23, AB3);
        AB4 = vec_type::fmadd(A2, B24, AB4);

        auto A3 = vec_type::loadu(A + (l + 2) * MR);

        auto B31 = vec_type::set(B[(l + 2) * NR + 0]);
        auto B32 = vec_type::set(B[(l + 2) * NR + 1]);
        auto B33 = vec_type::set(B[(l + 2) * NR + 2]);
        auto B34 = vec_type::set(B[(l + 2) * NR + 3]);

        AB1 = vec_type::fmadd(A3, B31, AB1);
        AB2 = vec_type::fmadd(A3, B32, AB2);
        AB3 = vec_type::fmadd(A3, B33, AB3);
        AB4 = vec_type::fmadd(A3, B34, AB4);

        auto A4 = vec_type::loadu(A + (l + 3) * MR);

        auto B41 = vec_type::set(B[(l + 3) * NR + 0]);
        auto B42 = vec_type::set(B[(l + 3) * NR + 1]);
        auto B43 = vec_type::set(B[(l + 3) * NR + 2]);
        auto B44 = vec_type::set(B[(l + 3) * NR + 3]);

        AB1 = vec_type::fmadd(A4, B41, AB1);
        AB2 = vec_type::fmadd(A4, B42, AB2);
        AB3 = vec_type::fmadd(A4, B43, AB3);
        AB4 = vec_type::fmadd(A4, B44, AB4);
    }

    for (; l + 1 < kc; l += 2) {
        auto A1 = vec_type::loadu(A + (l + 0) * MR);

        auto B11 = vec_type::set(B[(l + 0) * NR + 0]);
        auto B12 = vec_type::set(B[(l + 0) * NR + 1]);
        auto B13 = vec_type::set(B[(l + 0) * NR + 2]);
        auto B14 = vec_type::set(B[(l + 0) * NR + 3]);

        AB1 = vec_type::fmadd(A1, B11, AB1);
        AB2 = vec_type::fmadd(A1, B12, AB2);
        AB3 = vec_type::fmadd(A1, B13, AB3);
        AB4 = vec_type::fmadd(A1, B14, AB4);

        auto A2 = vec_type::loadu(A + (l + 1) * MR);

        auto B21 = vec_type::set(B[(l + 1) * NR + 0]);
        auto B22 = vec_type::set(B[(l + 1) * NR + 1]);
        auto B23 = vec_type::set(B[(l + 1) * NR + 2]);
        auto B24 = vec_type::set(B[(l + 1) * NR + 3]);

        AB1 = vec_type::fmadd(A2, B21, AB1);
        AB2 = vec_type::fmadd(A2, B22, AB2);
        AB3 = vec_type::fmadd(A2, B23, AB3);
        AB4 = vec_type::fmadd(A2, B24, AB4);
    }

    for (; l < kc; ++l) {
        auto A1 = vec_type::loadu(A + l * MR);

        auto B1 = vec_type::set(B[l * NR + 0]);
        auto B2 = vec_type::set(B[l * NR + 1]);
        auto B3 = vec_type::set(B[l * NR + 2]);
        auto B4 = vec_type::set(B[l * NR + 3]);

        AB1 = vec_type::fmadd(A1, B1, AB1);
        AB2 = vec_type::fmadd(A1, B2, AB2);
        AB3 = vec_type::fmadd(A1, B3, AB3);
        AB4 = vec_type::fmadd(A1, B4, AB4);
    }

    vec_type::storeu(AB + 0 * MR, AB1);
    vec_type::storeu(AB + 1 * MR, AB2);
    vec_type::storeu(AB + 2 * MR, AB3);
    vec_type::storeu(AB + 3 * MR, AB4);
}

template <typename V, typename T>
void gemm_micro_kernel(size_t kc, T alpha, const T* A, const T* B, T beta, T* C, size_t incRowC, size_t incColC) {
    static constexpr const size_t MR = gemm_config<T>::MR;
    static constexpr const size_t NR = gemm_config<T>::NR;

    T AB[MR * NR];

    // Bottleneck kernel

    gemm_pico_kernel<V>(kc, A, B, AB);

    if (alpha == T(1.0)) {
        if (beta == T(0.0)) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = AB[i + j * MR];
                }
            }
        } else if (beta != T(1.0)) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = beta * C[i * incRowC + j * incColC] + AB[i + j * MR];
                }
            }
        }
    } else {
        if (beta == T(0.0)) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = alpha * AB[i + j * MR];
                }
            }
        } else if (beta != T(1.0)) {
            for (size_t j = 0; j < NR; ++j) {
                for (size_t i = 0; i < MR; ++i) {
                    C[i * incRowC + j * incColC] = beta * C[i * incRowC + j * incColC]+ alpha * AB[i + j * MR];
                }
            }
        }
    }
}


template <typename V, typename T>
void dgemm_macro_kernel(size_t mc, size_t nc, size_t kc, T alpha, T beta, T* C, size_t incRowC, size_t incColC, etl::dyn_matrix<T, 2>& _A, etl::dyn_matrix<T, 2>& _B, etl::dyn_matrix<T, 2>& _C) {
    static constexpr const size_t MR = gemm_config<T>::MR;
    static constexpr const size_t NR = gemm_config<T>::NR;

    const size_t mp = (mc + MR - 1) / MR;
    const size_t np = (nc + NR - 1) / NR;

    const size_t _mr = mc % MR;
    const size_t _nr = nc % NR;

    for (size_t j = 0; j < np; ++j) {
        size_t nr = (j != np - 1 || _nr == 0) ? NR : _nr;

        for (size_t i = 0; i < mp; ++i) {
            size_t mr = (i != mp - 1 || _mr == 0) ? MR : _mr;

            if (mr == MR && nr == NR) {
                gemm_micro_kernel<V>(kc, alpha, &_A[i * kc * MR], &_B[j * kc * NR], beta, &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
            } else {
                gemm_micro_kernel<V>(kc, alpha, &_A[i * kc * MR], &_B[j * kc * NR], T(0.0), _C.memory_start(), 1, MR);
                dgescal(mr, nr, beta, &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
                dgeaxpy(mr, nr, T(1.0), _C.memory_start(), 1, MR, &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
            }
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
template <typename V, typename T, cpp_enable_if(all_floating_t<T>::value)>
void gemm_large_kernel_workspace_rr(const T* A, const T* B, T* C, size_t m, size_t n, size_t k, T beta) {
    static constexpr const size_t MC = gemm_config<T>::MC;
    static constexpr const size_t KC = gemm_config<T>::KC;
    static constexpr const size_t NC = gemm_config<T>::NC;

    static constexpr const size_t MR = gemm_config<T>::MR;
    static constexpr const size_t NR = gemm_config<T>::NR;

    etl::dyn_matrix<T, 2> _A(MC, KC);
    etl::dyn_matrix<T, 2> _B(KC, NC);
    etl::dyn_matrix<T, 2> _C(MR, NR);

    const size_t mb = (m + MC - 1) / MC;
    const size_t nb = (n + NC - 1) / NC;
    const size_t kb = (k + KC - 1) / KC;

    const size_t _mc = m % MC;
    const size_t _nc = n % NC;
    const size_t _kc = k % KC;

    const size_t incRowA = k;
    const size_t incColA = 1;

    const size_t incRowB = n;
    const size_t incColB = 1;

    const size_t incRowC = n;
    const size_t incColC = 1;

    const T alpha = 1.0;

    for (size_t j = 0; j < nb; ++j) {
        const size_t nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

        for (size_t l = 0; l < kb; ++l) {
            const size_t kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
            T _beta         = (l == 0) ? beta : 1.0;

            pack_B(kc, nc, &B[l * KC * incRowB + j * NC * incColB], incRowB, incColB, _B.memory_start());

            for (size_t i = 0; i < mb; ++i) {
                const size_t mc = (i != mb - 1 || _mc == 0) ? MC : _mc;

                pack_A(mc, kc, &A[i * MC * incRowA + l * KC * incColA], incRowA, incColA, _A.memory_start());

                dgemm_macro_kernel<V>(mc, nc, kc, alpha, _beta,
                                   &C[i * MC * incRowC + j * NC * incColC],
                                   incRowC, incColC, _A, _B, _C);
            }
        }
    }
}

template <typename V, typename T, cpp_disable_if(all_floating_t<T>::value)>
void gemm_large_kernel_workspace_rr(const T* , const T* , T* , size_t , size_t , size_t , T ) {

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
        //gemm_large_kernel_workspace_rr<default_vec>(a.memory_start(), b.memory_start(), c.memory_start(), M, N, K, T(0));
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

            // Clear the block
            for (size_t j = block_j; j < j_end; ++j) {
                for (size_t i = block_i; i < i_end; ++i) {
                    c[i + j * M] = 0;
                }
            }

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
        gemm_large_kernel_cc<default_vec>(a.memory_start(), b.memory_start(), c.memory_start(), M, N, K);
    }

    c.invalidate_gpu();
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
