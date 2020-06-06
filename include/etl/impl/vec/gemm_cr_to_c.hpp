//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Kernels for colum-major matrix - row-major matrix multiplication and
 * assignment to a column-major matrix
 */

#pragma once

namespace etl::impl::vec {

/*!
 * \brief Optimized version of GEMM for assignment of a small
 * Column-Major Matrix - Row Major Matrix to a Column Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_small_kernel_cr_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    auto alpha_vec = vec_type::set(alpha);

    const auto i_end = M & (size_t(-vec_size));

    size_t i = 0;

    for (; i + 3 * vec_size < i_end; i += 4 * vec_size) {
        size_t j = 0;

        for (; j + 1 < N; j += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();
            auto r31 = vec_type::template zero<T>();
            auto r41 = vec_type::template zero<T>();

            auto r12 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();
            auto r32 = vec_type::template zero<T>();
            auto r42 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);
                auto a3 = vec_type::loadu(a + i + k * M + 2 * vec_size);
                auto a4 = vec_type::loadu(a + i + k * M + 3 * vec_size);

                auto b1 = vec_type::set(b[k * N + (j + 0)]);
                auto b2 = vec_type::set(b[k * N + (j + 1)]);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);
                r31 = vec_type::fmadd(a3, b1, r31);
                r41 = vec_type::fmadd(a4, b1, r41);

                r12 = vec_type::fmadd(a1, b2, r12);
                r22 = vec_type::fmadd(a2, b2, r22);
                r32 = vec_type::fmadd(a3, b2, r32);
                r42 = vec_type::fmadd(a4, b2, r42);
            }

            vec_type::storeu(c + i + (j + 0) * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i + (j + 0) * M + 1 * vec_size, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + i + (j + 0) * M + 2 * vec_size, vec_type::mul(alpha_vec, r31));
            vec_type::storeu(c + i + (j + 0) * M + 3 * vec_size, vec_type::mul(alpha_vec, r41));

            vec_type::storeu(c + i + (j + 1) * M + 0 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + i + (j + 1) * M + 1 * vec_size, vec_type::mul(alpha_vec, r22));
            vec_type::storeu(c + i + (j + 1) * M + 2 * vec_size, vec_type::mul(alpha_vec, r32));
            vec_type::storeu(c + i + (j + 1) * M + 3 * vec_size, vec_type::mul(alpha_vec, r42));
        }

        if (j < N) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();
            auto r31 = vec_type::template zero<T>();
            auto r41 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);
                auto a3 = vec_type::loadu(a + i + k * M + 2 * vec_size);
                auto a4 = vec_type::loadu(a + i + k * M + 3 * vec_size);

                auto b1 = vec_type::set(b[k * N + j]);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);
                r31 = vec_type::fmadd(a3, b1, r31);
                r41 = vec_type::fmadd(a4, b1, r41);
            }

            vec_type::storeu(c + i + j * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i + j * M + 1 * vec_size, vec_type::mul(alpha_vec, r21));
            vec_type::storeu(c + i + j * M + 2 * vec_size, vec_type::mul(alpha_vec, r31));
            vec_type::storeu(c + i + j * M + 3 * vec_size, vec_type::mul(alpha_vec, r41));
        }
    }

    for (; i + 1 * vec_size < i_end; i += 2 * vec_size) {
        size_t j = 0;

        for (; j + 1 < N; j += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            auto r12 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);

                auto b1 = vec_type::set(b[k * N + (j + 0)]);
                auto b2 = vec_type::set(b[k * N + (j + 1)]);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);

                r12 = vec_type::fmadd(a1, b2, r12);
                r22 = vec_type::fmadd(a2, b2, r22);
            }

            vec_type::storeu(c + i + (j + 0) * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i + (j + 0) * M + 1 * vec_size, vec_type::mul(alpha_vec, r21));

            vec_type::storeu(c + i + (j + 1) * M + 0 * vec_size, vec_type::mul(alpha_vec, r12));
            vec_type::storeu(c + i + (j + 1) * M + 1 * vec_size, vec_type::mul(alpha_vec, r22));
        }

        if (j < N) {
            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);

                auto b1 = vec_type::set(b[k * N + j]);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);
            }

            vec_type::storeu(c + i + j * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i + j * M + 1 * vec_size, vec_type::mul(alpha_vec, r21));
        }
    }

    for (; i < i_end; i += vec_size) {
        size_t j = 0;

        for (; j + 1 < N; j += 2) {
            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);

                auto b1 = vec_type::set(b[k * N + (j + 0)]);
                auto b2 = vec_type::set(b[k * N + (j + 1)]);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a1, b2, r12);
            }

            vec_type::storeu(c + i + (j + 0) * M, vec_type::mul(alpha_vec, r11));
            vec_type::storeu(c + i + (j + 1) * M, vec_type::mul(alpha_vec, r12));
        }

        if (j < N) {
            auto r11 = vec_type::template zero<T>();

            for (size_t k = 0; k < K; ++k) {
                auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);

                auto b1 = vec_type::set(b[k * N + j]);

                r11 = vec_type::fmadd(a1, b1, r11);
            }

            vec_type::storeu(c + i + j * M, vec_type::mul(alpha_vec, r11));
        }
    }

    for (; i < M; ++i) {
        size_t j = 0;

        for (; j + 1 < N; j += 2) {
            auto r11 = T();
            auto r12 = T();

            for (size_t k = 0; k < K; ++k) {
                r11 += a[i + k * M] * b[k * N + (j + 0)];
                r12 += a[i + k * M] * b[k * N + (j + 1)];
            }

            c[i + (j + 0) * M] = alpha * r11;
            c[i + (j + 1) * M] = alpha * r12;
        }

        if (j < N) {
            auto r11 = T();

            for (size_t k = 0; k < K; ++k) {
                r11 += a[i + k * M] * b[k * N + j];
            }

            c[i + j * M] = alpha * r11;
        }
    }
}

/*!
 * \brief Optimized version of GEMM for assignment of a large
 * Column-Major Matrix - Row Major Matrix to a Column Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_large_kernel_cr_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    constexpr size_t n_block_size = 128UL;
    constexpr size_t m_block_size = 64UL;
    constexpr size_t k_block_size = 128UL;

    auto alpha_vec = vec_type::set(alpha);

    for (size_t ii = 0; ii < M; ii += m_block_size) {
        const size_t i_end = std::min(ii + m_block_size, M);
        const size_t i_pos = i_end & size_t(-vec_size);

        for (size_t jj = 0; jj < N; jj += n_block_size) {
            const size_t j_end = std::min(jj + n_block_size, N);

            for (size_t kk = 0; kk < K; kk += k_block_size) {
                const size_t k_end = std::min(kk + k_block_size, K);

                size_t i = ii;

#ifdef __clang__
                for (; i + 3 * vec_size < i_pos; i += 4 * vec_size) {
                    size_t j = jj;

                    for (; j + 1 < j_end; j += 2) {
                        auto r11 = vec_type::loadu(c + i + (j + 0) * M + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + i + (j + 0) * M + 1 * vec_size);
                        auto r13 = vec_type::loadu(c + i + (j + 0) * M + 2 * vec_size);
                        auto r14 = vec_type::loadu(c + i + (j + 0) * M + 3 * vec_size);

                        auto r21 = vec_type::loadu(c + i + (j + 1) * M + 0 * vec_size);
                        auto r22 = vec_type::loadu(c + i + (j + 1) * M + 1 * vec_size);
                        auto r23 = vec_type::loadu(c + i + (j + 1) * M + 2 * vec_size);
                        auto r24 = vec_type::loadu(c + i + (j + 1) * M + 3 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);
                            auto a3 = vec_type::loadu(a + i + k * M + 2 * vec_size);
                            auto a4 = vec_type::loadu(a + i + k * M + 3 * vec_size);

                            auto b1 = vec_type::set(b[k * N + (j + 0)]);
                            auto b2 = vec_type::set(b[k * N + (j + 1)]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);
                            r13 = vec_type::fmadd(a3, b1, r13);
                            r14 = vec_type::fmadd(a4, b1, r14);

                            r21 = vec_type::fmadd(a1, b2, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                            r23 = vec_type::fmadd(a3, b2, r23);
                            r24 = vec_type::fmadd(a4, b2, r24);
                        }

                        vec_type::storeu(c + i + (j + 0) * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + i + (j + 0) * M + 1 * vec_size, vec_type::mul(alpha_vec, r12));
                        vec_type::storeu(c + i + (j + 0) * M + 2 * vec_size, vec_type::mul(alpha_vec, r13));
                        vec_type::storeu(c + i + (j + 0) * M + 3 * vec_size, vec_type::mul(alpha_vec, r14));

                        vec_type::storeu(c + i + (j + 1) * M + 0 * vec_size, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + i + (j + 1) * M + 1 * vec_size, vec_type::mul(alpha_vec, r22));
                        vec_type::storeu(c + i + (j + 1) * M + 2 * vec_size, vec_type::mul(alpha_vec, r23));
                        vec_type::storeu(c + i + (j + 1) * M + 3 * vec_size, vec_type::mul(alpha_vec, r24));
                    }

                    if (j < j_end) {
                        auto r11 = vec_type::loadu(c + i + j * M + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + i + j * M + 1 * vec_size);
                        auto r13 = vec_type::loadu(c + i + j * M + 2 * vec_size);
                        auto r14 = vec_type::loadu(c + i + j * M + 3 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);
                            auto a3 = vec_type::loadu(a + i + k * M + 2 * vec_size);
                            auto a4 = vec_type::loadu(a + i + k * M + 3 * vec_size);

                            auto b1 = vec_type::set(b[k * N + j]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);
                            r13 = vec_type::fmadd(a3, b1, r13);
                            r14 = vec_type::fmadd(a4, b1, r14);
                        }

                        vec_type::storeu(c + i + j * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + i + j * M + 1 * vec_size, vec_type::mul(alpha_vec, r12));
                        vec_type::storeu(c + i + j * M + 2 * vec_size, vec_type::mul(alpha_vec, r13));
                        vec_type::storeu(c + i + j * M + 3 * vec_size, vec_type::mul(alpha_vec, r14));
                    }
                }
#endif

                for (; i + 1 * vec_size < i_pos; i += 2 * vec_size) {
                    size_t j = jj;

                    for (; j + 3 < j_end; j += 4) {
                        auto r11 = vec_type::loadu(c + i + (j + 0) * M + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + i + (j + 0) * M + 1 * vec_size);

                        auto r21 = vec_type::loadu(c + i + (j + 1) * M + 0 * vec_size);
                        auto r22 = vec_type::loadu(c + i + (j + 1) * M + 1 * vec_size);

                        auto r31 = vec_type::loadu(c + i + (j + 2) * M + 0 * vec_size);
                        auto r32 = vec_type::loadu(c + i + (j + 2) * M + 1 * vec_size);

                        auto r41 = vec_type::loadu(c + i + (j + 3) * M + 0 * vec_size);
                        auto r42 = vec_type::loadu(c + i + (j + 3) * M + 1 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);

                            auto b1 = vec_type::set(b[k * N + (j + 0)]);
                            auto b2 = vec_type::set(b[k * N + (j + 1)]);
                            auto b3 = vec_type::set(b[k * N + (j + 2)]);
                            auto b4 = vec_type::set(b[k * N + (j + 3)]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);

                            r21 = vec_type::fmadd(a1, b2, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);

                            r31 = vec_type::fmadd(a1, b3, r31);
                            r32 = vec_type::fmadd(a2, b3, r32);

                            r41 = vec_type::fmadd(a1, b4, r41);
                            r42 = vec_type::fmadd(a2, b4, r42);
                        }

                        vec_type::storeu(c + i + (j + 0) * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + i + (j + 0) * M + 1 * vec_size, vec_type::mul(alpha_vec, r12));

                        vec_type::storeu(c + i + (j + 1) * M + 0 * vec_size, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + i + (j + 1) * M + 1 * vec_size, vec_type::mul(alpha_vec, r22));

                        vec_type::storeu(c + i + (j + 2) * M + 0 * vec_size, vec_type::mul(alpha_vec, r31));
                        vec_type::storeu(c + i + (j + 2) * M + 1 * vec_size, vec_type::mul(alpha_vec, r32));

                        vec_type::storeu(c + i + (j + 3) * M + 0 * vec_size, vec_type::mul(alpha_vec, r41));
                        vec_type::storeu(c + i + (j + 3) * M + 1 * vec_size, vec_type::mul(alpha_vec, r42));
                    }

                    for (; j + 1 < j_end; j += 2) {
                        auto r11 = vec_type::loadu(c + i + (j + 0) * M + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + i + (j + 0) * M + 1 * vec_size);

                        auto r21 = vec_type::loadu(c + i + (j + 1) * M + 0 * vec_size);
                        auto r22 = vec_type::loadu(c + i + (j + 1) * M + 1 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);

                            auto b1 = vec_type::set(b[k * N + (j + 0)]);
                            auto b2 = vec_type::set(b[k * N + (j + 1)]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);

                            r21 = vec_type::fmadd(a1, b2, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                        }

                        vec_type::storeu(c + i + (j + 0) * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + i + (j + 0) * M + 1 * vec_size, vec_type::mul(alpha_vec, r12));

                        vec_type::storeu(c + i + (j + 1) * M + 0 * vec_size, vec_type::mul(alpha_vec, r21));
                        vec_type::storeu(c + i + (j + 1) * M + 1 * vec_size, vec_type::mul(alpha_vec, r22));
                    }

                    if (j < j_end) {
                        auto r11 = vec_type::loadu(c + i + j * M + 0 * vec_size);
                        auto r12 = vec_type::loadu(c + i + j * M + 1 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + i + k * M + 1 * vec_size);

                            auto b1 = vec_type::set(b[k * N + j]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);
                        }

                        vec_type::storeu(c + i + j * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                        vec_type::storeu(c + i + j * M + 1 * vec_size, vec_type::mul(alpha_vec, r12));
                    }
                }

                for (; i < i_pos; i += vec_size) {
                    for (size_t j = jj; j < j_end; ++j) {
                        auto r11 = vec_type::loadu(c + i + j * M + 0 * vec_size);

                        for (size_t k = kk; k < k_end; ++k) {
                            auto a1 = vec_type::loadu(a + i + k * M + 0 * vec_size);

                            auto b1 = vec_type::set(b[k * N + j]);

                            r11 = vec_type::fmadd(a1, b1, r11);
                        }

                        vec_type::storeu(c + i + j * M + 0 * vec_size, vec_type::mul(alpha_vec, r11));
                    }
                }

                for (; i < i_end; ++i) {
                    for (size_t j = jj; j < j_end; ++j) {
                        auto r11 = c[i + j * M];

                        for (size_t k = kk; k < k_end; ++k) {
                            r11 += a[i + k * M] * b[k * N + j];
                        }

                        c[i + j * M] = alpha * r11;
                    }
                }
            }
        }
    }
}

/*!
 * \brief Vectorized implementation of column-major matrix - row-major matrix
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
void gemm_cr_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    if (M * N <= gemm_rr_small_threshold) {
        gemm_small_kernel_cr_to_c<default_vec>(a, b, c, M, N, K, alpha);
    } else {
        direct_fill_n(c, M * N, T(0));
        gemm_large_kernel_cr_to_c<default_vec>(a, b, c, M, N, K, alpha);
    }
}

} //end of namespace etl::impl::vec
