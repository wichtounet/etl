//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Kernels for row-major matrix - column-major matrix multiplication and
 * assignment to a column-major matrix
 */

#pragma once

namespace etl::impl::vec {

/*!
 * \brief Optimized version of GEMM for assignment of a small
 * Row-Major Matrix - Column Major Matrix to a Column Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_small_kernel_rc_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t k_pos = K & size_t(-vec_size);

    size_t i = 0;

    for (; i + 3 < M; i += 4) {
        size_t j = 0;

        for (; j + 1 < N; j += 2) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();
            auto r13 = vec_type::template zero<T>();
            auto r14 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();
            auto r23 = vec_type::template zero<T>();
            auto r24 = vec_type::template zero<T>();

            for (; k < k_pos; k += vec_size) {
                auto a1 = vec_type::loadu(a + (i + 0) * K + k + 0 * vec_size);
                auto a2 = vec_type::loadu(a + (i + 1) * K + k + 0 * vec_size);
                auto a3 = vec_type::loadu(a + (i + 2) * K + k + 0 * vec_size);
                auto a4 = vec_type::loadu(a + (i + 3) * K + k + 0 * vec_size);

                auto b1 = vec_type::loadu(b + k + (j + 0) * K + 0 * vec_size);
                auto b2 = vec_type::loadu(b + k + (j + 1) * K + 0 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);
                r13 = vec_type::fmadd(a3, b1, r13);
                r14 = vec_type::fmadd(a4, b1, r14);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);
                r23 = vec_type::fmadd(a3, b2, r23);
                r24 = vec_type::fmadd(a4, b2, r24);
            }

            auto v11 = vec_type::hadd(r11);
            auto v12 = vec_type::hadd(r12);
            auto v13 = vec_type::hadd(r13);
            auto v14 = vec_type::hadd(r14);

            auto v21 = vec_type::hadd(r21);
            auto v22 = vec_type::hadd(r22);
            auto v23 = vec_type::hadd(r23);
            auto v24 = vec_type::hadd(r24);

            for (; k < K; ++k) {
                v11 += a[(i + 0) * K + k] * b[k + (j + 0) * K];
                v12 += a[(i + 1) * K + k] * b[k + (j + 0) * K];
                v13 += a[(i + 2) * K + k] * b[k + (j + 0) * K];
                v14 += a[(i + 3) * K + k] * b[k + (j + 0) * K];

                v21 += a[(i + 0) * K + k] * b[k + (j + 1) * K];
                v22 += a[(i + 1) * K + k] * b[k + (j + 1) * K];
                v23 += a[(i + 2) * K + k] * b[k + (j + 1) * K];
                v24 += a[(i + 3) * K + k] * b[k + (j + 1) * K];
            }

            c[(i + 0) + (j + 0) * M] = alpha * v11;
            c[(i + 1) + (j + 0) * M] = alpha * v12;
            c[(i + 2) + (j + 0) * M] = alpha * v13;
            c[(i + 3) + (j + 0) * M] = alpha * v14;

            c[(i + 0) + (j + 1) * M] = alpha * v21;
            c[(i + 1) + (j + 1) * M] = alpha * v22;
            c[(i + 2) + (j + 1) * M] = alpha * v23;
            c[(i + 3) + (j + 1) * M] = alpha * v24;
        }

        for (; j < N; ++j) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();
            auto r13 = vec_type::template zero<T>();
            auto r14 = vec_type::template zero<T>();

            for (; k < k_pos; k += vec_size) {
                auto a1 = vec_type::loadu(a + (i + 0) * K + k + 0 * vec_size);
                auto a2 = vec_type::loadu(a + (i + 1) * K + k + 0 * vec_size);
                auto a3 = vec_type::loadu(a + (i + 2) * K + k + 0 * vec_size);
                auto a4 = vec_type::loadu(a + (i + 3) * K + k + 0 * vec_size);

                auto b1 = vec_type::loadu(b + k + j * K + 0 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);
                r13 = vec_type::fmadd(a3, b1, r13);
                r14 = vec_type::fmadd(a4, b1, r14);
            }

            auto v11 = vec_type::hadd(r11);
            auto v12 = vec_type::hadd(r12);
            auto v13 = vec_type::hadd(r13);
            auto v14 = vec_type::hadd(r14);

            for (; k < K; ++k) {
                v11 += a[(i + 0) * K + k] * b[k + j * K];
                v12 += a[(i + 1) * K + k] * b[k + j * K];
                v13 += a[(i + 2) * K + k] * b[k + j * K];
                v14 += a[(i + 3) * K + k] * b[k + j * K];
            }

            c[(i + 0) + j * M] = alpha * v11;
            c[(i + 1) + j * M] = alpha * v12;
            c[(i + 2) + j * M] = alpha * v13;
            c[(i + 3) + j * M] = alpha * v14;
        }
    }

    for (; i + 1 < M; i += 2) {
        size_t j = 0;

        for (; j + 1 < N; j += 2) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            auto r21 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            for (; k < k_pos; k += vec_size) {
                auto a1 = vec_type::loadu(a + (i + 0) * K + k + 0 * vec_size);
                auto a2 = vec_type::loadu(a + (i + 1) * K + k + 0 * vec_size);

                auto b1 = vec_type::loadu(b + k + (j + 0) * K + 0 * vec_size);
                auto b2 = vec_type::loadu(b + k + (j + 1) * K + 0 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);

                r21 = vec_type::fmadd(a1, b2, r21);
                r22 = vec_type::fmadd(a2, b2, r22);
            }

            auto v11 = vec_type::hadd(r11);
            auto v12 = vec_type::hadd(r12);

            auto v21 = vec_type::hadd(r21);
            auto v22 = vec_type::hadd(r22);

            for (; k < K; ++k) {
                v11 += a[(i + 0) * K + k] * b[k + (j + 0) * K];
                v12 += a[(i + 1) * K + k] * b[k + (j + 0) * K];

                v21 += a[(i + 0) * K + k] * b[k + (j + 1) * K];
                v22 += a[(i + 1) * K + k] * b[k + (j + 1) * K];
            }

            c[(i + 0) + (j + 0) * M] = alpha * v11;
            c[(i + 1) + (j + 0) * M] = alpha * v12;

            c[(i + 0) + (j + 1) * M] = alpha * v21;
            c[(i + 1) + (j + 1) * M] = alpha * v22;
        }

        for (; j < N; ++j) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            for (; k < k_pos; k += vec_size) {
                auto a1 = vec_type::loadu(a + (i + 0) * K + k + 0 * vec_size);
                auto a2 = vec_type::loadu(a + (i + 1) * K + k + 0 * vec_size);

                auto b1 = vec_type::loadu(b + k + j * K + 0 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a2, b1, r12);
            }

            auto v11 = vec_type::hadd(r11);
            auto v12 = vec_type::hadd(r12);

            for (; k < K; ++k) {
                v11 += a[(i + 0) * K + k] * b[k + j * K];
                v12 += a[(i + 1) * K + k] * b[k + j * K];
            }

            c[(i + 0) + j * M] = alpha * v11;
            c[(i + 1) + j * M] = alpha * v12;
        }
    }

    for (; i < M; ++i) {
        size_t j = 0;

        for (; j + 1 < N; j += 2) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            for (; k < k_pos; k += vec_size) {
                auto a1 = vec_type::loadu(a + i * K + k + 0 * vec_size);

                auto b1 = vec_type::loadu(b + k + (j + 0) * K + 0 * vec_size);
                auto b2 = vec_type::loadu(b + k + (j + 1) * K + 0 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a1, b2, r21);
            }

            auto v11 = vec_type::hadd(r11);
            auto v21 = vec_type::hadd(r21);

            for (; k < K; ++k) {
                v11 += a[i * K + k] * b[k + (j + 0) * K];
                v21 += a[i * K + k] * b[k + (j + 1) * K];
            }

            c[i + (j + 0) * M] = alpha * v11;
            c[i + (j + 1) * M] = alpha * v21;
        }

        for (; j < N; ++j) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();

            for (; k < k_pos; k += vec_size) {
                auto a1 = vec_type::loadu(a + i * K + k + 0 * vec_size);

                auto b1 = vec_type::loadu(b + k + j * K + 0 * vec_size);

                r11 = vec_type::fmadd(a1, b1, r11);
            }

            auto v11 = vec_type::hadd(r11);

            for (; k < K; ++k) {
                v11 += a[i * K + k] * b[k + j * K];
            }

            c[i + j * M] = alpha * v11;
        }
    }
}

/*!
 * \brief Optimized version of GEMM for assignment of a large
 * Row-Major Matrix - Column Major Matrix to a Column Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_large_kernel_rc_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    constexpr size_t n_block_size = 128UL;
    constexpr size_t m_block_size = 64UL;
    constexpr size_t k_block_size = 128UL;

    for (size_t ii = 0; ii < M; ii += m_block_size) {
        const size_t i_end = std::min(ii + m_block_size, M);

        for (size_t jj = 0; jj < N; jj += n_block_size) {
            const size_t j_end = std::min(jj + n_block_size, N);

            for (size_t kk = 0; kk < K; kk += k_block_size) {
                const size_t k_end = std::min(kk + k_block_size, K);
                const size_t k_pos = k_end & size_t(-vec_size);

                size_t i = ii;

                for (; i + 3 < i_end; i += 4) {
                    size_t j = jj;

                    for (; j + 1 < j_end; j += 2) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r12 = vec_type::template zero<T>();
                        auto r13 = vec_type::template zero<T>();
                        auto r14 = vec_type::template zero<T>();

                        auto r21 = vec_type::template zero<T>();
                        auto r22 = vec_type::template zero<T>();
                        auto r23 = vec_type::template zero<T>();
                        auto r24 = vec_type::template zero<T>();

                        for (; k < k_pos; k += vec_size) {
                            auto a1 = vec_type::loadu(a + (i + 0) * K + k + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + (i + 1) * K + k + 0 * vec_size);
                            auto a3 = vec_type::loadu(a + (i + 2) * K + k + 0 * vec_size);
                            auto a4 = vec_type::loadu(a + (i + 3) * K + k + 0 * vec_size);

                            auto b1 = vec_type::loadu(b + k + (j + 0) * K + 0 * vec_size);
                            auto b2 = vec_type::loadu(b + k + (j + 1) * K + 0 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);
                            r13 = vec_type::fmadd(a3, b1, r13);
                            r14 = vec_type::fmadd(a4, b1, r14);

                            r21 = vec_type::fmadd(a1, b2, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                            r23 = vec_type::fmadd(a3, b2, r23);
                            r24 = vec_type::fmadd(a4, b2, r24);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v12 = vec_type::hadd(r12);
                        auto v13 = vec_type::hadd(r13);
                        auto v14 = vec_type::hadd(r14);

                        auto v21 = vec_type::hadd(r21);
                        auto v22 = vec_type::hadd(r22);
                        auto v23 = vec_type::hadd(r23);
                        auto v24 = vec_type::hadd(r24);

                        for (; k < k_end; ++k) {
                            v11 += a[(i + 0) * K + k] * b[k + (j + 0) * K];
                            v12 += a[(i + 1) * K + k] * b[k + (j + 0) * K];
                            v13 += a[(i + 2) * K + k] * b[k + (j + 0) * K];
                            v14 += a[(i + 3) * K + k] * b[k + (j + 0) * K];

                            v21 += a[(i + 0) * K + k] * b[k + (j + 1) * K];
                            v22 += a[(i + 1) * K + k] * b[k + (j + 1) * K];
                            v23 += a[(i + 2) * K + k] * b[k + (j + 1) * K];
                            v24 += a[(i + 3) * K + k] * b[k + (j + 1) * K];
                        }

                        c[(i + 0) + (j + 0) * M] += alpha * v11;
                        c[(i + 1) + (j + 0) * M] += alpha * v12;
                        c[(i + 2) + (j + 0) * M] += alpha * v13;
                        c[(i + 3) + (j + 0) * M] += alpha * v14;

                        c[(i + 0) + (j + 1) * M] += alpha * v21;
                        c[(i + 1) + (j + 1) * M] += alpha * v22;
                        c[(i + 2) + (j + 1) * M] += alpha * v23;
                        c[(i + 3) + (j + 1) * M] += alpha * v24;
                    }

                    for (; j < j_end; ++j) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r12 = vec_type::template zero<T>();
                        auto r13 = vec_type::template zero<T>();
                        auto r14 = vec_type::template zero<T>();

                        for (; k < k_pos; k += vec_size) {
                            auto a1 = vec_type::loadu(a + (i + 0) * K + k + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + (i + 1) * K + k + 0 * vec_size);
                            auto a3 = vec_type::loadu(a + (i + 2) * K + k + 0 * vec_size);
                            auto a4 = vec_type::loadu(a + (i + 3) * K + k + 0 * vec_size);

                            auto b1 = vec_type::loadu(b + k + j * K + 0 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);
                            r13 = vec_type::fmadd(a3, b1, r13);
                            r14 = vec_type::fmadd(a4, b1, r14);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v12 = vec_type::hadd(r12);
                        auto v13 = vec_type::hadd(r13);
                        auto v14 = vec_type::hadd(r14);

                        for (; k < k_end; ++k) {
                            v11 += a[(i + 0) * K + k] * b[k + j * K];
                            v12 += a[(i + 1) * K + k] * b[k + j * K];
                            v13 += a[(i + 2) * K + k] * b[k + j * K];
                            v14 += a[(i + 3) * K + k] * b[k + j * K];
                        }

                        c[(i + 0) + j * M] += alpha * v11;
                        c[(i + 1) + j * M] += alpha * v12;
                        c[(i + 2) + j * M] += alpha * v13;
                        c[(i + 3) + j * M] += alpha * v14;
                    }
                }

                for (; i + 1 < i_end; i += 2) {
                    size_t j = jj;

                    for (; j + 1 < j_end; j += 2) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r12 = vec_type::template zero<T>();

                        auto r21 = vec_type::template zero<T>();
                        auto r22 = vec_type::template zero<T>();

                        for (; k < k_pos; k += vec_size) {
                            auto a1 = vec_type::loadu(a + (i + 0) * K + k + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + (i + 1) * K + k + 0 * vec_size);

                            auto b1 = vec_type::loadu(b + k + (j + 0) * K + 0 * vec_size);
                            auto b2 = vec_type::loadu(b + k + (j + 1) * K + 0 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);

                            r21 = vec_type::fmadd(a1, b2, r21);
                            r22 = vec_type::fmadd(a2, b2, r22);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v12 = vec_type::hadd(r12);

                        auto v21 = vec_type::hadd(r21);
                        auto v22 = vec_type::hadd(r22);

                        for (; k < k_end; ++k) {
                            v11 += a[(i + 0) * K + k] * b[k + (j + 0) * K];
                            v12 += a[(i + 1) * K + k] * b[k + (j + 0) * K];

                            v21 += a[(i + 0) * K + k] * b[k + (j + 1) * K];
                            v22 += a[(i + 1) * K + k] * b[k + (j + 1) * K];
                        }

                        c[(i + 0) + (j + 0) * M] += alpha * v11;
                        c[(i + 1) + (j + 0) * M] += alpha * v12;

                        c[(i + 0) + (j + 1) * M] += alpha * v21;
                        c[(i + 1) + (j + 1) * M] += alpha * v22;
                    }

                    for (; j < j_end; ++j) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r12 = vec_type::template zero<T>();

                        for (; k < k_pos; k += vec_size) {
                            auto a1 = vec_type::loadu(a + (i + 0) * K + k + 0 * vec_size);
                            auto a2 = vec_type::loadu(a + (i + 1) * K + k + 0 * vec_size);

                            auto b1 = vec_type::loadu(b + k + j * K + 0 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a2, b1, r12);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v12 = vec_type::hadd(r12);

                        for (; k < k_end; ++k) {
                            v11 += a[(i + 0) * K + k] * b[k + j * K];
                            v12 += a[(i + 1) * K + k] * b[k + j * K];
                        }

                        c[(i + 0) + j * M] += alpha * v11;
                        c[(i + 1) + j * M] += alpha * v12;
                    }
                }

                for (; i < i_end; ++i) {
                    size_t j = jj;

                    for (; j + 1 < j_end; j += 2) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r21 = vec_type::template zero<T>();

                        for (; k < k_pos; k += vec_size) {
                            auto a1 = vec_type::loadu(a + i * K + k + 0 * vec_size);

                            auto b1 = vec_type::loadu(b + k + (j + 0) * K + 0 * vec_size);
                            auto b2 = vec_type::loadu(b + k + (j + 1) * K + 0 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r21 = vec_type::fmadd(a1, b2, r21);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v21 = vec_type::hadd(r21);

                        for (; k < k_end; ++k) {
                            v11 += a[i * K + k] * b[k + (j + 0) * K];
                            v21 += a[i * K + k] * b[k + (j + 1) * K];
                        }

                        c[i + (j + 0) * M] += alpha * v11;
                        c[i + (j + 1) * M] += alpha * v21;
                    }

                    for (; j < j_end; ++j) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();

                        for (; k < k_pos; k += vec_size) {
                            auto a1 = vec_type::loadu(a + i * K + k + 0 * vec_size);

                            auto b1 = vec_type::loadu(b + k + j * K + 0 * vec_size);

                            r11 = vec_type::fmadd(a1, b1, r11);
                        }

                        auto v11 = vec_type::hadd(r11);

                        for (; k < k_end; ++k) {
                            v11 += a[i * K + k] * b[k + j * K];
                        }

                        c[i + j * M] += alpha * v11;
                    }
                }
            }
        }
    }
}

/*!
 * \brief Vectorized implementation of row-major matrix - column-major matrix
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
void gemm_rc_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K, T alpha) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    if (M * N <= 10) {
        gemm_small_kernel_rc_to_c<default_vec>(a, b, c, M, N, K, alpha);
    } else {
        // TODO Use the large kernel once it's been made faster
        gemm_small_kernel_rc_to_c<default_vec>(a, b, c, M, N, K, alpha);
    }
}

} //end of namespace etl::impl::vec
