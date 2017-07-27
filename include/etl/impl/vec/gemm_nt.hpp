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
 * \brief Optimized version of GEMM for assignment of a small
 * Row-Major Matrix - Column Major Matrix to a Row Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_nt_small_kernel_rr(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    size_t i = 0;

    const auto k_end = K & (size_t(-vec_size));

    for (; i + 1 < M; i += 2) {
        size_t j = 0;

        for (; j + 3 < N; j += 4) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            auto r12 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            auto r13 = vec_type::template zero<T>();
            auto r23 = vec_type::template zero<T>();

            auto r14 = vec_type::template zero<T>();
            auto r24 = vec_type::template zero<T>();

            for (; k < k_end; k += vec_size) {
                auto a1 = vec_type::loadu(a + (i + 0) * K + k + vec_size * 0);
                auto a2 = vec_type::loadu(a + (i + 1) * K + k + vec_size * 0);

                auto b1 = vec_type::loadu(b + (j + 0) * K + k + vec_size * 0);
                auto b2 = vec_type::loadu(b + (j + 1) * K + k + vec_size * 0);
                auto b3 = vec_type::loadu(b + (j + 2) * K + k + vec_size * 0);
                auto b4 = vec_type::loadu(b + (j + 3) * K + k + vec_size * 0);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);

                r12 = vec_type::fmadd(a1, b2, r12);
                r22 = vec_type::fmadd(a2, b2, r22);

                r13 = vec_type::fmadd(a1, b3, r13);
                r23 = vec_type::fmadd(a2, b3, r23);

                r14 = vec_type::fmadd(a1, b4, r14);
                r24 = vec_type::fmadd(a2, b4, r24);
            }

            auto v11 = vec_type::hadd(r11);
            auto v21 = vec_type::hadd(r21);

            auto v12 = vec_type::hadd(r12);
            auto v22 = vec_type::hadd(r22);

            auto v13 = vec_type::hadd(r13);
            auto v23 = vec_type::hadd(r23);

            auto v14 = vec_type::hadd(r14);
            auto v24 = vec_type::hadd(r24);

            for (; k < K; ++k) {
                v11 += a[(i + 0) * K + k] * b[k + (j + 0) * K];
                v21 += a[(i + 1) * K + k] * b[k + (j + 0) * K];

                v12 += a[(i + 0) * K + k] * b[k + (j + 1) * K];
                v22 += a[(i + 1) * K + k] * b[k + (j + 1) * K];

                v13 += a[(i + 0) * K + k] * b[k + (j + 2) * K];
                v23 += a[(i + 1) * K + k] * b[k + (j + 2) * K];

                v14 += a[(i + 0) * K + k] * b[k + (j + 3) * K];
                v24 += a[(i + 1) * K + k] * b[k + (j + 3) * K];
            }

            c[(i + 0) * N + (j + 0)] = v11;
            c[(i + 1) * N + (j + 0)] = v21;

            c[(i + 0) * N + (j + 1)] = v12;
            c[(i + 1) * N + (j + 1)] = v22;

            c[(i + 0) * N + (j + 2)] = v13;
            c[(i + 1) * N + (j + 2)] = v23;

            c[(i + 0) * N + (j + 3)] = v14;
            c[(i + 1) * N + (j + 3)] = v24;
        }

        for (; j + 1 < N; j += 2) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            auto r12 = vec_type::template zero<T>();
            auto r22 = vec_type::template zero<T>();

            for (; k < k_end; k += vec_size) {
                auto a1 = vec_type::loadu(a + (i + 0) * K + k + vec_size * 0);
                auto a2 = vec_type::loadu(a + (i + 1) * K + k + vec_size * 0);

                auto b1 = vec_type::loadu(b + (j + 0) * K + k + vec_size * 0);
                auto b2 = vec_type::loadu(b + (j + 1) * K + k + vec_size * 0);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);

                r12 = vec_type::fmadd(a1, b2, r12);
                r22 = vec_type::fmadd(a2, b2, r22);
            }

            auto v11 = vec_type::hadd(r11);
            auto v21 = vec_type::hadd(r21);

            auto v12 = vec_type::hadd(r12);
            auto v22 = vec_type::hadd(r22);

            for (; k < K; ++k) {
                v11 += a[(i + 0) * K + k] * b[k + (j + 0) * K];
                v21 += a[(i + 1) * K + k] * b[k + (j + 0) * K];

                v12 += a[(i + 0) * K + k] * b[k + (j + 1) * K];
                v22 += a[(i + 1) * K + k] * b[k + (j + 1) * K];
            }

            c[(i + 0) * N + (j + 0)] = v11;
            c[(i + 1) * N + (j + 0)] = v21;

            c[(i + 0) * N + (j + 1)] = v12;
            c[(i + 1) * N + (j + 1)] = v22;
        }

        for (; j < N; ++j) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r21 = vec_type::template zero<T>();

            for (; k < k_end; k += vec_size) {
                auto a1 = vec_type::loadu(a + (i + 0) * K + k + vec_size * 0);
                auto a2 = vec_type::loadu(a + (i + 1) * K + k + vec_size * 0);

                auto b1 = vec_type::loadu(b + j * K + k + vec_size * 0);

                r11 = vec_type::fmadd(a1, b1, r11);
                r21 = vec_type::fmadd(a2, b1, r21);
            }

            auto v11 = vec_type::hadd(r11);
            auto v21 = vec_type::hadd(r21);

            for (; k < K; ++k) {
                v11 += a[(i + 0) * K + k] * b[k + j * K];
                v21 += a[(i + 1) * K + k] * b[k + j * K];
            }

            c[(i + 0) * N + j] = v11;
            c[(i + 1) * N + j] = v21;
        }
    }

    for (; i < M; ++i) {
        size_t j = 0;

#ifdef __clang__
        for (; j + 3 < N; j += 4) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();
            auto r13 = vec_type::template zero<T>();
            auto r14 = vec_type::template zero<T>();

            for (; k < k_end; k += vec_size) {
                auto a1 = vec_type::loadu(a + i * K + k + vec_size * 0);

                auto b1 = vec_type::loadu(b + (j + 0) * K + k + vec_size * 0);
                auto b2 = vec_type::loadu(b + (j + 1) * K + k + vec_size * 0);
                auto b3 = vec_type::loadu(b + (j + 2) * K + k + vec_size * 0);
                auto b4 = vec_type::loadu(b + (j + 3) * K + k + vec_size * 0);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a1, b2, r12);
                r13 = vec_type::fmadd(a1, b3, r13);
                r14 = vec_type::fmadd(a1, b4, r14);
            }

            auto v11 = vec_type::hadd(r11);
            auto v12 = vec_type::hadd(r12);
            auto v13 = vec_type::hadd(r13);
            auto v14 = vec_type::hadd(r14);

            for (; k < K; ++k) {
                v11 += a[i * K + k] * b[k + (j + 0) * K];
                v12 += a[i * K + k] * b[k + (j + 1) * K];
                v13 += a[i * K + k] * b[k + (j + 2) * K];
                v14 += a[i * K + k] * b[k + (j + 3) * K];
            }

            c[i * N + (j + 0)] = v11;
            c[i * N + (j + 1)] = v12;
            c[i * N + (j + 2)] = v13;
            c[i * N + (j + 3)] = v14;
        }
#endif

        for (; j + 1 < N; j += 2) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();
            auto r12 = vec_type::template zero<T>();

            for (; k < k_end; k += vec_size) {
                auto a1 = vec_type::loadu(a + i * K + k + vec_size * 0);

                auto b1 = vec_type::loadu(b + (j + 0) * K + k + vec_size * 0);
                auto b2 = vec_type::loadu(b + (j + 1) * K + k + vec_size * 0);

                r11 = vec_type::fmadd(a1, b1, r11);
                r12 = vec_type::fmadd(a1, b2, r12);
            }

            auto v11 = vec_type::hadd(r11);
            auto v12 = vec_type::hadd(r12);

            for (; k < K; ++k) {
                v11 += a[i * K + k] * b[k + (j + 0) * K];
                v12 += a[i * K + k] * b[k + (j + 1) * K];
            }

            c[i * N + (j + 0)] = v11;
            c[i * N + (j + 1)] = v12;
        }

        for (; j < N; ++j) {
            size_t k = 0;

            auto r11 = vec_type::template zero<T>();

            for (; k < k_end; k += vec_size) {
                auto a1 = vec_type::loadu(a + i * K + k + vec_size * 0);

                auto b1 = vec_type::loadu(b + j * K + k + vec_size * 0);

                r11 = vec_type::fmadd(a1, b1, r11);
            }

            auto v11 = vec_type::hadd(r11);

            for (; k < K; ++k) {
                v11 += a[i * K + k] * b[k + j * K];
            }

            c[i * N + j] = v11;
        }
    }
}

/*!
 * \brief Optimized version of GEMM for assignment of a large
 * Row-Major Matrix - Column Major Matrix to a Row Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_nt_large_kernel_rr(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    constexpr size_t n_block_size = 256UL;
    constexpr size_t m_block_size = 128UL;
    constexpr size_t k_block_size = 256UL;

    for (size_t ii = 0; ii < M; ii += m_block_size) {
        const size_t i_end = std::min(ii + m_block_size, M);

        for (size_t jj = 0; jj < N; jj += n_block_size) {
            const size_t j_end = std::min(jj + n_block_size, N);

            for (size_t kk = 0; kk < K; kk += k_block_size) {
                const size_t k_end_a = std::min(kk + k_block_size, K);
                const size_t k_end   = k_end_a & size_t(-vec_size);

                size_t i = ii;

                for (; i + 1 < i_end; i += 2) {
                    size_t j = jj;

                    for (; j + 3 < j_end; j += 4) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r21 = vec_type::template zero<T>();

                        auto r12 = vec_type::template zero<T>();
                        auto r22 = vec_type::template zero<T>();

                        auto r13 = vec_type::template zero<T>();
                        auto r23 = vec_type::template zero<T>();

                        auto r14 = vec_type::template zero<T>();
                        auto r24 = vec_type::template zero<T>();

                        for (; k < k_end; k += vec_size) {
                            auto a1 = vec_type::loadu(a + (i + 0) * K + k + vec_size * 0);
                            auto a2 = vec_type::loadu(a + (i + 1) * K + k + vec_size * 0);

                            auto b1 = vec_type::loadu(b + (j + 0) * K + k + vec_size * 0);
                            auto b2 = vec_type::loadu(b + (j + 1) * K + k + vec_size * 0);
                            auto b3 = vec_type::loadu(b + (j + 2) * K + k + vec_size * 0);
                            auto b4 = vec_type::loadu(b + (j + 3) * K + k + vec_size * 0);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r21 = vec_type::fmadd(a2, b1, r21);

                            r12 = vec_type::fmadd(a1, b2, r12);
                            r22 = vec_type::fmadd(a2, b2, r22);

                            r13 = vec_type::fmadd(a1, b3, r13);
                            r23 = vec_type::fmadd(a2, b3, r23);

                            r14 = vec_type::fmadd(a1, b4, r14);
                            r24 = vec_type::fmadd(a2, b4, r24);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v21 = vec_type::hadd(r21);

                        auto v12 = vec_type::hadd(r12);
                        auto v22 = vec_type::hadd(r22);

                        auto v13 = vec_type::hadd(r13);
                        auto v23 = vec_type::hadd(r23);

                        auto v14 = vec_type::hadd(r14);
                        auto v24 = vec_type::hadd(r24);

                        for (; k < k_end_a; ++k) {
                            v11 += a[(i + 0) * K + k] * b[k + (j + 0) * K];
                            v21 += a[(i + 1) * K + k] * b[k + (j + 0) * K];

                            v12 += a[(i + 0) * K + k] * b[k + (j + 1) * K];
                            v22 += a[(i + 1) * K + k] * b[k + (j + 1) * K];

                            v13 += a[(i + 0) * K + k] * b[k + (j + 2) * K];
                            v23 += a[(i + 1) * K + k] * b[k + (j + 2) * K];

                            v14 += a[(i + 0) * K + k] * b[k + (j + 3) * K];
                            v24 += a[(i + 1) * K + k] * b[k + (j + 3) * K];
                        }

                        c[(i + 0) * N + (j + 0)] += v11;
                        c[(i + 1) * N + (j + 0)] += v21;

                        c[(i + 0) * N + (j + 1)] += v12;
                        c[(i + 1) * N + (j + 1)] += v22;

                        c[(i + 0) * N + (j + 2)] += v13;
                        c[(i + 1) * N + (j + 2)] += v23;

                        c[(i + 0) * N + (j + 3)] += v14;
                        c[(i + 1) * N + (j + 3)] += v24;
                    }

                    for (; j + 1 < j_end; j += 2) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r21 = vec_type::template zero<T>();

                        auto r12 = vec_type::template zero<T>();
                        auto r22 = vec_type::template zero<T>();

                        for (; k < k_end; k += vec_size) {
                            auto a1 = vec_type::loadu(a + (i + 0) * K + k + vec_size * 0);
                            auto a2 = vec_type::loadu(a + (i + 1) * K + k + vec_size * 0);

                            auto b1 = vec_type::loadu(b + (j + 0) * K + k + vec_size * 0);
                            auto b2 = vec_type::loadu(b + (j + 1) * K + k + vec_size * 0);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r21 = vec_type::fmadd(a2, b1, r21);

                            r12 = vec_type::fmadd(a1, b2, r12);
                            r22 = vec_type::fmadd(a2, b2, r22);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v21 = vec_type::hadd(r21);

                        auto v12 = vec_type::hadd(r12);
                        auto v22 = vec_type::hadd(r22);

                        for (; k < k_end_a; ++k) {
                            v11 += a[(i + 0) * K + k] * b[k + (j + 0) * K];
                            v21 += a[(i + 1) * K + k] * b[k + (j + 0) * K];

                            v12 += a[(i + 0) * K + k] * b[k + (j + 1) * K];
                            v22 += a[(i + 1) * K + k] * b[k + (j + 1) * K];
                        }

                        c[(i + 0) * N + (j + 0)] += v11;
                        c[(i + 1) * N + (j + 0)] += v21;

                        c[(i + 0) * N + (j + 1)] += v12;
                        c[(i + 1) * N + (j + 1)] += v22;
                    }

                    for (; j < j_end; ++j) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r21 = vec_type::template zero<T>();

                        for (; k < k_end; k += vec_size) {
                            auto a1 = vec_type::loadu(a + (i + 0) * K + k + vec_size * 0);
                            auto a2 = vec_type::loadu(a + (i + 1) * K + k + vec_size * 0);

                            auto b1 = vec_type::loadu(b + j * K + k + vec_size * 0);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r21 = vec_type::fmadd(a2, b1, r21);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v21 = vec_type::hadd(r21);

                        for (; k < k_end_a; ++k) {
                            v11 += a[(i + 0) * K + k] * b[k + j * K];
                            v21 += a[(i + 1) * K + k] * b[k + j * K];
                        }

                        c[(i + 0) * N + j] += v11;
                        c[(i + 1) * N + j] += v21;
                    }
                }

                for (; i < i_end; ++i) {
                    size_t j = jj;

                    for (; j + 1 < j_end; j += 2) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();
                        auto r12 = vec_type::template zero<T>();

                        for (; k < k_end; k += vec_size) {
                            auto a1 = vec_type::loadu(a + i * K + k + vec_size * 0);

                            auto b1 = vec_type::loadu(b + (j + 0) * K + k + vec_size * 0);
                            auto b2 = vec_type::loadu(b + (j + 1) * K + k + vec_size * 0);

                            r11 = vec_type::fmadd(a1, b1, r11);
                            r12 = vec_type::fmadd(a1, b2, r12);
                        }

                        auto v11 = vec_type::hadd(r11);
                        auto v12 = vec_type::hadd(r12);

                        for (; k < k_end_a; ++k) {
                            v11 += a[i * K + k] * b[k + (j + 0) * K];
                            v12 += a[i * K + k] * b[k + (j + 1) * K];
                        }

                        c[i * N + (j + 0)] += v11;
                        c[i * N + (j + 1)] += v12;
                    }

                    for (; j < j_end; ++j) {
                        size_t k = kk;

                        auto r11 = vec_type::template zero<T>();

                        for (; k < k_end; k += vec_size) {
                            auto a1 = vec_type::loadu(a + i * K + k + vec_size * 0);

                            auto b1 = vec_type::loadu(b + j * K + k + vec_size * 0);

                            r11 = vec_type::fmadd(a1, b1, r11);
                        }

                        auto v11 = vec_type::hadd(r11);

                        for (; k < k_end_a; ++k) {
                            v11 += a[i * K + k] * b[k + j * K];
                        }

                        c[i * N + j] += v11;
                    }
                }
            }
        }
    }
}

/*!
 * \brief Optimized version of GEMM for assignment of a
 * Column-Major Matrix - Row Major Matrix to a Row Major Matrix.
 *
 * \param a The lhs matrix (row major)
 * \param b The rhs matrix (transposed row major)
 * \param c The result matrix (row major)
 */
template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemm_nt(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    const size_t M = etl::rows(a);
    const size_t N = etl::rows(b);
    const size_t K = etl::columns(a);

    if (etl::size(c) <= gemm_nt_rr_small_threshold) {
        gemm_nt_small_kernel_rr<default_vec>(a.memory_start(), b.memory_start(), c.memory_start(), M, N, K);
    } else {
        c = 0;
        gemm_nt_large_kernel_rr<default_vec>(a.memory_start(), b.memory_start(), c.memory_start(), M, N, K);
    }

    c.invalidate_gpu();
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
