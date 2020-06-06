//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Kernels for column-major matrix - column-major matrix multiplication and
 * assignment to a column-major matrix
 */

#pragma once

namespace etl::impl::vec {

/*!
 * \brief Optimized version of small GEMM for column major version
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_small_kernel_cc_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    auto alpha_vec = vec_type::set(alpha);

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

            vec_type::storeu(c + (i + vec_size * 0) + (j + 0) * M, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + vec_size * 0) + (j + 1) * M, vec_type::mul(alpha_vec, r12));

            vec_type::storeu(c + (i + vec_size * 1) + (j + 0) * M, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + (i + vec_size * 1) + (j + 1) * M, vec_type::mul(alpha_vec, r22));

            vec_type::storeu(c + (i + vec_size * 2) + (j + 0) * M, vec_type::mul(alpha_vec, r31));
            vec_type::storeu(c + (i + vec_size * 2) + (j + 1) * M, vec_type::mul(alpha_vec, r32));

            vec_type::storeu(c + (i + vec_size * 3) + (j + 0) * M, vec_type::mul(alpha_vec, r41));
            vec_type::storeu(c + (i + vec_size * 3) + (j + 1) * M, vec_type::mul(alpha_vec, r42));
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

            vec_type::storeu(c + i + vec_size * 0 + j * M, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i + vec_size * 1 + j * M, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + i + vec_size * 2 + j * M, vec_type::mul(alpha_vec, r31));
            vec_type::storeu(c + i + vec_size * 3 + j * M, vec_type::mul(alpha_vec, r41));
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

            vec_type::storeu(c + (i + vec_size * 0) + (j + 0) * M, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + (i + vec_size * 0) + (j + 1) * M, vec_type::mul(alpha_vec, r12));

            vec_type::storeu(c + (i + vec_size * 1) + (j + 0) * M, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + (i + vec_size * 1) + (j + 1) * M, vec_type::mul(alpha_vec, r22));
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

            vec_type::storeu(c + i + vec_size * 0 + j * M, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i + vec_size * 1 + j * M, vec_type::mul(alpha_vec, r21));
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

            vec_type::storeu(c + i + (j + 0) * M, vec_type::mul(alpha_vec, r1));
            vec_type::storeu(c + i + (j + 1) * M, vec_type::mul(alpha_vec, r2));
        }

        if (j < N) {
            auto r1 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M);

                auto b1 = vec_type::set(b[k + j * K]);

                r1 = vec_type::fmadd(a1, b1, r1);
            }

            vec_type::storeu(c + i + j * M, vec_type::mul(alpha_vec, r1));
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

            c[i + (j + 0) * M] = alpha * value1;
            c[i + (j + 1) * M] = alpha * value2;
        }

        if (j < N) {
            T value(0);

            for (size_t k = 0; k < K; ++k) {
                value += a[i + k * M] * b[k + j * K];
            }

            c[i + j * M] = alpha * value;
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
void gemm_large_kernel_cc_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    constexpr size_t m_block_size = 128;
    constexpr size_t n_block_size = 64;
    constexpr size_t k_block_size = 128;

    auto alpha_vec = vec_type::set(alpha);

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

                        vec_type::storeu(c + i + 0 * vec_size + (j + 0) * M, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + i + 1 * vec_size + (j + 0) * M, vec_type::mul(alpha_vec, r12));
                        vec_type::storeu(c + i + 2 * vec_size + (j + 0) * M, vec_type::mul(alpha_vec, r13));
                        vec_type::storeu(c + i + 3 * vec_size + (j + 0) * M, vec_type::mul(alpha_vec, r14));

                        vec_type::storeu(c + i + 0 * vec_size + (j + 1) * M, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + i + 1 * vec_size + (j + 1) * M, vec_type::mul(alpha_vec, r22));
                        vec_type::storeu(c + i + 2 * vec_size + (j + 1) * M, vec_type::mul(alpha_vec, r23));
                        vec_type::storeu(c + i + 3 * vec_size + (j + 1) * M, vec_type::mul(alpha_vec, r24));
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

                        vec_type::storeu(c + i + 0 * vec_size + j * M, vec_type::mul(alpha_vec, r1));
                        vec_type::storeu(c + i + 1 * vec_size + j * M, vec_type::mul(alpha_vec, r2));
                        vec_type::storeu(c + i + 2 * vec_size + j * M, vec_type::mul(alpha_vec, r3));
                        vec_type::storeu(c + i + 3 * vec_size + j * M, vec_type::mul(alpha_vec, r4));
                    }
                }

                // 2x unrolled vectorized inner loop
                for (; i + 2 * vec_size - 1 < i_end; i += 2 * vec_size) {
                    size_t j = block_j;

                    for (; j + 3 < j_end; j += 4) {
                        auto r11 = vec_type::loadu(c + i + 0 * vec_size + (j + 0) * M);
                        auto r12 = vec_type::loadu(c + i + 1 * vec_size + (j + 0) * M);

                        auto r21 = vec_type::loadu(c + i + 0 * vec_size + (j + 1) * M);
                        auto r22 = vec_type::loadu(c + i + 1 * vec_size + (j + 1) * M);

                        auto r31 = vec_type::loadu(c + i + 0 * vec_size + (j + 2) * M);
                        auto r32 = vec_type::loadu(c + i + 1 * vec_size + (j + 2) * M);

                        auto r41 = vec_type::loadu(c + i + 0 * vec_size + (j + 3) * M);
                        auto r42 = vec_type::loadu(c + i + 1 * vec_size + (j + 3) * M);

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

                        vec_type::storeu(c + i + 0 * vec_size + (j + 0) * M, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + i + 1 * vec_size + (j + 0) * M, vec_type::mul(alpha_vec, r12));

                        vec_type::storeu(c + i + 0 * vec_size + (j + 1) * M, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + i + 1 * vec_size + (j + 1) * M, vec_type::mul(alpha_vec, r22));

                        vec_type::storeu(c + i + 0 * vec_size + (j + 2) * M, vec_type::mul(alpha_vec, r31));
                        vec_type::storeu(c + i + 1 * vec_size + (j + 2) * M, vec_type::mul(alpha_vec, r32));

                        vec_type::storeu(c + i + 0 * vec_size + (j + 3) * M, vec_type::mul(alpha_vec, r41));
                        vec_type::storeu(c + i + 1 * vec_size + (j + 3) * M, vec_type::mul(alpha_vec, r42));
                    }

                    for (; j + 1 < j_end; j += 2) {
                        auto r11 = vec_type::loadu(c + i + 0 * vec_size + (j + 0) * M);
                        auto r12 = vec_type::loadu(c + i + 1 * vec_size + (j + 0) * M);

                        auto r21 = vec_type::loadu(c + i + 0 * vec_size + (j + 1) * M);
                        auto r22 = vec_type::loadu(c + i + 1 * vec_size + (j + 1) * M);

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

                        vec_type::storeu(c + i + 0 * vec_size + (j + 0) * M, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + i + 1 * vec_size + (j + 0) * M, vec_type::mul(alpha_vec, r12));

                        vec_type::storeu(c + i + 0 * vec_size + (j + 1) * M, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + i + 1 * vec_size + (j + 1) * M, vec_type::mul(alpha_vec, r22));
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

                        vec_type::storeu(c + i + 0 * vec_size + j * M, vec_type::mul(alpha_vec, r1));
                        vec_type::storeu(c + i + 1 * vec_size + j * M, vec_type::mul(alpha_vec, r2));
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

                        vec_type::storeu(c + i + j * M, vec_type::mul(alpha_vec, r1));
                    }
                }

                // Remainder inner loop
                for (; i < i_end; ++i) {
                    for (size_t j = block_j; j < j_end; ++j) {
                        auto x = c[i + j * M];

                        for (size_t k = block_k; k < k_end; ++k) {
                            x += a[i + k * M] * b[k + j * K];
                        }

                        c[i + j * M] = alpha * x;
                    }
                }
            }
        }
    }
}

/*!
 * \brief Vectorized implementation of column-major matrix - column-major matrix
 * multiplication and assignment into a column-major matrix.
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
void gemm_cc_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    // Dispatch to the best kernel

    if (M * N <= gemm_cc_small_threshold) {
        gemm_small_kernel_cc_to_c<default_vec>(a, b, c, M, N, K, alpha);
    } else {
        gemm_large_kernel_cc_to_c<default_vec>(a, b, c, M, N, K, alpha);
    }
}

} //end of namespace etl::impl::vec
