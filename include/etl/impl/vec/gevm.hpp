//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// The idea of the GEMM kernels is largely inspired by the kernels in Blaze by
// Klaus Igleberg

namespace etl::impl::vec {

/*!
 * \brief Optimized version of small GEVM for row major version
 * \param aa The lhs vector
 * \param bb The rhs matrix
 * \param c The result vector
 */
template <typename V, typename T, typename C>
void gevm_small_kernel_rr(const T* aa, size_t m, size_t n, const T* bb, C&& c) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t j_end = n & size_t(-vec_size);

    size_t j = 0;

    for (; j + vec_size * 7 < j_end; j += vec_size * 8) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();
        auto r5 = vec_type::template zero<T>();
        auto r6 = vec_type::template zero<T>();
        auto r7 = vec_type::template zero<T>();
        auto r8 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(aa[k]);

            auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);
            auto b2 = vec_type::loadu(bb + k * n + j + 1 * vec_size);
            auto b3 = vec_type::loadu(bb + k * n + j + 2 * vec_size);
            auto b4 = vec_type::loadu(bb + k * n + j + 3 * vec_size);
            auto b5 = vec_type::loadu(bb + k * n + j + 4 * vec_size);
            auto b6 = vec_type::loadu(bb + k * n + j + 5 * vec_size);
            auto b7 = vec_type::loadu(bb + k * n + j + 6 * vec_size);
            auto b8 = vec_type::loadu(bb + k * n + j + 7 * vec_size);

            r1 = vec_type::fmadd(a1, b1, r1);
            r2 = vec_type::fmadd(a1, b2, r2);
            r3 = vec_type::fmadd(a1, b3, r3);
            r4 = vec_type::fmadd(a1, b4, r4);
            r5 = vec_type::fmadd(a1, b5, r5);
            r6 = vec_type::fmadd(a1, b6, r6);
            r7 = vec_type::fmadd(a1, b7, r7);
            r8 = vec_type::fmadd(a1, b8, r8);
        }

        c.template store<vec_type>(r1, j + 0 * vec_size);
        c.template store<vec_type>(r2, j + 1 * vec_size);
        c.template store<vec_type>(r3, j + 2 * vec_size);
        c.template store<vec_type>(r4, j + 3 * vec_size);
        c.template store<vec_type>(r5, j + 4 * vec_size);
        c.template store<vec_type>(r6, j + 5 * vec_size);
        c.template store<vec_type>(r7, j + 6 * vec_size);
        c.template store<vec_type>(r8, j + 7 * vec_size);
    }

    for (; j + vec_size * 3 < j_end; j += vec_size * 4) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(aa[k]);

            auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);
            auto b2 = vec_type::loadu(bb + k * n + j + 1 * vec_size);
            auto b3 = vec_type::loadu(bb + k * n + j + 2 * vec_size);
            auto b4 = vec_type::loadu(bb + k * n + j + 3 * vec_size);

            r1 = vec_type::fmadd(a1, b1, r1);
            r2 = vec_type::fmadd(a1, b2, r2);
            r3 = vec_type::fmadd(a1, b3, r3);
            r4 = vec_type::fmadd(a1, b4, r4);
        }

        c.template store<vec_type>(r1, j + 0 * vec_size);
        c.template store<vec_type>(r2, j + 1 * vec_size);
        c.template store<vec_type>(r3, j + 2 * vec_size);
        c.template store<vec_type>(r4, j + 3 * vec_size);
    }

    for (; j + vec_size < j_end; j += vec_size * 2) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(aa[k]);

            auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);
            auto b2 = vec_type::loadu(bb + k * n + j + 1 * vec_size);

            r1 = vec_type::fmadd(a1, b1, r1);
            r2 = vec_type::fmadd(a1, b2, r2);
        }

        c.template store<vec_type>(r1, j + 0 * vec_size);
        c.template store<vec_type>(r2, j + 1 * vec_size);
    }

    for (; j < j_end; j += vec_size) {
        auto r1 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(aa[k]);

            auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);

            r1 = vec_type::fmadd(a1, b1, r1);
        }

        c.template store<vec_type>(r1, j + 0 * vec_size);
    }

    for (; j + 1 < n; j += 2) {
        auto v1 = T();
        auto v2 = T();

        for (size_t k = 0; k < m; k++) {
            v1 += aa[k] * bb[k * n + j + 0];
            v2 += aa[k] * bb[k * n + j + 1];
        }

        c[j + 0] = v1;
        c[j + 1] = v2;
    }

    for (; j < n; j++) {
        auto value = T();

        for (size_t k = 0; k < m; k++) {
            value += aa[k] * bb[k * n + j];
        }

        c[j] = value;
    }
}

/*!
 * \brief Optimized version of large GEVM for row major version
 * \param aa The lhs vector
 * \param bb The rhs matrix
 * \param cc The result vector
 */
template <typename V, typename T, typename C>
void gevm_large_kernel_rr(const T* aa, size_t m, size_t n, const T* bb, C&& cc) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t n_block = (32 * 1024) / sizeof(T);
    const size_t m_block = n < n_block ? 8 : 4;

    cc = 0;

    for (size_t block_j = 0; block_j < n; block_j += n_block) {
        const size_t n_end = std::min(block_j + n_block, n) & size_t(-vec_size);

        for (size_t block_k = 0; block_k < m; block_k += m_block) {
            const size_t m_end = std::min(block_k + m_block, m);

            size_t j = block_j;

            // 8-Unrolled Vectorized loop
            for (; j + vec_size * 7 < n_end; j += vec_size * 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(aa[k]);

                    auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);
                    auto b2 = vec_type::loadu(bb + k * n + j + 1 * vec_size);
                    auto b3 = vec_type::loadu(bb + k * n + j + 2 * vec_size);
                    auto b4 = vec_type::loadu(bb + k * n + j + 3 * vec_size);
                    auto b5 = vec_type::loadu(bb + k * n + j + 4 * vec_size);
                    auto b6 = vec_type::loadu(bb + k * n + j + 5 * vec_size);
                    auto b7 = vec_type::loadu(bb + k * n + j + 6 * vec_size);
                    auto b8 = vec_type::loadu(bb + k * n + j + 7 * vec_size);

                    r1 = vec_type::fmadd(a1, b1, r1);
                    r2 = vec_type::fmadd(a1, b2, r2);
                    r3 = vec_type::fmadd(a1, b3, r3);
                    r4 = vec_type::fmadd(a1, b4, r4);
                    r5 = vec_type::fmadd(a1, b5, r5);
                    r6 = vec_type::fmadd(a1, b6, r6);
                    r7 = vec_type::fmadd(a1, b7, r7);
                    r8 = vec_type::fmadd(a1, b8, r8);
                }

                cc.template store<V>(vec_type::add(r1, cc.template load<V>(j + 0 * vec_size)), j + 0 * vec_size);
                cc.template store<V>(vec_type::add(r2, cc.template load<V>(j + 1 * vec_size)), j + 1 * vec_size);
                cc.template store<V>(vec_type::add(r3, cc.template load<V>(j + 2 * vec_size)), j + 2 * vec_size);
                cc.template store<V>(vec_type::add(r4, cc.template load<V>(j + 3 * vec_size)), j + 3 * vec_size);
                cc.template store<V>(vec_type::add(r5, cc.template load<V>(j + 4 * vec_size)), j + 4 * vec_size);
                cc.template store<V>(vec_type::add(r6, cc.template load<V>(j + 5 * vec_size)), j + 5 * vec_size);
                cc.template store<V>(vec_type::add(r7, cc.template load<V>(j + 6 * vec_size)), j + 6 * vec_size);
                cc.template store<V>(vec_type::add(r8, cc.template load<V>(j + 7 * vec_size)), j + 7 * vec_size);
            }

            // 4-Unrolled vectorized loop
            for (; j + vec_size * 3 < n_end; j += vec_size * 4) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(aa[k]);

                    auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);
                    auto b2 = vec_type::loadu(bb + k * n + j + 1 * vec_size);
                    auto b3 = vec_type::loadu(bb + k * n + j + 2 * vec_size);
                    auto b4 = vec_type::loadu(bb + k * n + j + 3 * vec_size);

                    r1 = vec_type::fmadd(a1, b1, r1);
                    r2 = vec_type::fmadd(a1, b2, r2);
                    r3 = vec_type::fmadd(a1, b3, r3);
                    r4 = vec_type::fmadd(a1, b4, r4);
                }

                cc.template store<V>(vec_type::add(r1, cc.template load<V>(j + 0 * vec_size)), j + 0 * vec_size);
                cc.template store<V>(vec_type::add(r2, cc.template load<V>(j + 1 * vec_size)), j + 1 * vec_size);
                cc.template store<V>(vec_type::add(r3, cc.template load<V>(j + 2 * vec_size)), j + 2 * vec_size);
                cc.template store<V>(vec_type::add(r4, cc.template load<V>(j + 3 * vec_size)), j + 3 * vec_size);
            }

            // 2-Unrolled vectorized loop
            for (; j + vec_size < n_end; j += vec_size * 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(aa[k]);

                    auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);
                    auto b2 = vec_type::loadu(bb + k * n + j + 1 * vec_size);

                    r1 = vec_type::fmadd(a1, b1, r1);
                    r2 = vec_type::fmadd(a1, b2, r2);
                }

                cc.template store<V>(vec_type::add(r1, cc.template load<V>(j + 0 * vec_size)), j + 0 * vec_size);
                cc.template store<V>(vec_type::add(r2, cc.template load<V>(j + 1 * vec_size)), j + 1 * vec_size);
            }

            // Base vectorized loop
            for (; j < n_end; j += vec_size) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(aa[k]);
                    auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);
                    r1      = vec_type::fmadd(a1, b1, r1);
                }

                cc.template store<V>(vec_type::add(r1, cc.template load<V>(j + 0 * vec_size)), j + 0 * vec_size);
            }

            // Remainder non-vectorized loop
            for (; j < n_end; ++j) {
                auto r1 = T();

                for (size_t k = block_k; k < m_end; ++k) {
                    r1 += aa[k] * bb[k * n + j];
                }

                cc[j] += r1;
            }
        }
    }
}

/*!
 * \brief Optimized version of small GEVM for row major version
 * \param aa The lhs vector
 * \param bb The rhs matrix
 * \param cc The result vector
 */
template <typename V, typename T>
void gevm_small_kernel_cc(const T* aa, size_t m, size_t n, const T* bb, T* cc) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    size_t j = 0;

    for (; j + 7 < n; j += 8) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();
        auto r5 = vec_type::template zero<T>();
        auto r6 = vec_type::template zero<T>();
        auto r7 = vec_type::template zero<T>();
        auto r8 = vec_type::template zero<T>();

        size_t i = 0;

        for (; i + vec_size - 1 < m; i += vec_size) {
            auto x1 = vec_type::loadu(aa + i);

            r1 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 0) * m), r1);
            r2 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 1) * m), r2);
            r3 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 2) * m), r3);
            r4 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 3) * m), r4);
            r5 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 4) * m), r5);
            r6 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 5) * m), r6);
            r7 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 6) * m), r7);
            r8 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 7) * m), r8);
        }

        cc[j + 0] = vec_type::hadd(r1);
        cc[j + 1] = vec_type::hadd(r2);
        cc[j + 2] = vec_type::hadd(r3);
        cc[j + 3] = vec_type::hadd(r4);
        cc[j + 4] = vec_type::hadd(r5);
        cc[j + 5] = vec_type::hadd(r6);
        cc[j + 6] = vec_type::hadd(r7);
        cc[j + 7] = vec_type::hadd(r8);

        for (; i < m; ++i) {
            cc[j + 0] += aa[i] * bb[i + (j + 0) * m];
            cc[j + 1] += aa[i] * bb[i + (j + 1) * m];
            cc[j + 2] += aa[i] * bb[i + (j + 2) * m];
            cc[j + 3] += aa[i] * bb[i + (j + 3) * m];
            cc[j + 4] += aa[i] * bb[i + (j + 4) * m];
            cc[j + 5] += aa[i] * bb[i + (j + 5) * m];
            cc[j + 6] += aa[i] * bb[i + (j + 6) * m];
            cc[j + 7] += aa[i] * bb[i + (j + 7) * m];
        }
    }

    for (; j + 3 < n; j += 4) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();

        size_t i = 0;

        for (; i + vec_size - 1 < m; i += vec_size) {
            auto x1 = vec_type::loadu(aa + i);

            r1 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 0) * m), r1);
            r2 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 1) * m), r2);
            r3 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 2) * m), r3);
            r4 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 3) * m), r4);
        }

        cc[j + 0] = vec_type::hadd(r1);
        cc[j + 1] = vec_type::hadd(r2);
        cc[j + 2] = vec_type::hadd(r3);
        cc[j + 3] = vec_type::hadd(r4);

        for (; i < m; ++i) {
            cc[j + 0] += aa[i] * bb[i + (j + 0) * m];
            cc[j + 1] += aa[i] * bb[i + (j + 1) * m];
            cc[j + 2] += aa[i] * bb[i + (j + 2) * m];
            cc[j + 3] += aa[i] * bb[i + (j + 3) * m];
        }
    }

    for (; j + 1 < n; j += 2) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();

        size_t i = 0;

        for (; i + vec_size - 1 < m; i += vec_size) {
            auto x1 = vec_type::loadu(aa + i);

            r1 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 0) * m), r1);
            r2 = vec_type::fmadd(x1, vec_type::loadu(bb + i + (j + 1) * m), r2);
        }

        cc[j + 0] = vec_type::hadd(r1);
        cc[j + 1] = vec_type::hadd(r2);

        for (; i < m; ++i) {
            cc[j + 0] += aa[i] * bb[i + (j + 0) * m];
            cc[j + 1] += aa[i] * bb[i + (j + 1) * m];
        }
    }

    for (; j < n; ++j) {
        auto r1 = vec_type::template zero<T>();

        size_t i = 0;

        for (; i + vec_size - 1 < m; i += vec_size) {
            auto x1 = vec_type::loadu(aa + i);

            r1 = vec_type::fmadd(x1, vec_type::loadu(bb + i + j * m), r1);
        }

        cc[j] = vec_type::hadd(r1);

        for (; i < m; ++i) {
            cc[j] += aa[i] * bb[i + j * m];
        }
    }
}

/*!
 * \brief Optimized version of small GEVM for row major version
 * \param aa The lhs vector
 * \param bb The rhs matrix
 * \param cc The result vector
 */
template <typename V, typename T>
void gevm_large_kernel_cc(const T* aa, size_t m, size_t n, const T* bb, T* cc) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    // TODO The inner kernels should probably rewritten to exploit FMA

    size_t j = 0;

    for (; j + 3 < n; j += 4) {
        size_t i = 0;

        for (; i + vec_size * 4 - 1 < m; i += vec_size * 4) {
            auto x1 = vec_type::loadu(aa + i + vec_size * 0);
            auto x2 = vec_type::loadu(aa + i + vec_size * 1);
            auto x3 = vec_type::loadu(aa + i + vec_size * 2);
            auto x4 = vec_type::loadu(aa + i + vec_size * 3);

            auto r11 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 0) * m));
            auto r12 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 0) * m));
            auto r13 = vec_type::mul(x3, vec_type::loadu(bb + i + vec_size * 2 + (j + 0) * m));
            auto r14 = vec_type::mul(x4, vec_type::loadu(bb + i + vec_size * 3 + (j + 0) * m));

            auto r21 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 1) * m));
            auto r22 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 1) * m));
            auto r23 = vec_type::mul(x3, vec_type::loadu(bb + i + vec_size * 2 + (j + 1) * m));
            auto r24 = vec_type::mul(x4, vec_type::loadu(bb + i + vec_size * 3 + (j + 1) * m));

            auto r31 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 2) * m));
            auto r32 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 2) * m));
            auto r33 = vec_type::mul(x3, vec_type::loadu(bb + i + vec_size * 2 + (j + 2) * m));
            auto r34 = vec_type::mul(x4, vec_type::loadu(bb + i + vec_size * 3 + (j + 2) * m));

            auto r41 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 3) * m));
            auto r42 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 3) * m));
            auto r43 = vec_type::mul(x3, vec_type::loadu(bb + i + vec_size * 2 + (j + 3) * m));
            auto r44 = vec_type::mul(x4, vec_type::loadu(bb + i + vec_size * 3 + (j + 3) * m));

            auto t11 = vec_type::add(r11, r12);
            auto t12 = vec_type::add(r13, r14);

            auto t21 = vec_type::add(r21, r22);
            auto t22 = vec_type::add(r23, r24);

            auto t31 = vec_type::add(r31, r32);
            auto t32 = vec_type::add(r33, r34);

            auto t41 = vec_type::add(r41, r42);
            auto t42 = vec_type::add(r43, r44);

            auto r1 = vec_type::add(t11, t12);
            auto r2 = vec_type::add(t21, t22);
            auto r3 = vec_type::add(t31, t32);
            auto r4 = vec_type::add(t41, t42);

            cc[j + 0] += vec_type::hadd(r1);
            cc[j + 1] += vec_type::hadd(r2);
            cc[j + 2] += vec_type::hadd(r3);
            cc[j + 3] += vec_type::hadd(r4);
        }

        for (; i + vec_size * 2 - 1 < m; i += vec_size * 2) {
            auto x1 = vec_type::loadu(aa + i + vec_size * 0);
            auto x2 = vec_type::loadu(aa + i + vec_size * 1);

            auto r11 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 0) * m));
            auto r12 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 0) * m));

            auto r21 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 1) * m));
            auto r22 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 1) * m));

            auto r31 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 2) * m));
            auto r32 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 2) * m));

            auto r41 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 3) * m));
            auto r42 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 3) * m));

            auto t1 = vec_type::add(r11, r12);
            auto t2 = vec_type::add(r21, r22);
            auto t3 = vec_type::add(r31, r32);
            auto t4 = vec_type::add(r41, r42);

            cc[j + 0] += vec_type::hadd(t1);
            cc[j + 1] += vec_type::hadd(t2);
            cc[j + 2] += vec_type::hadd(t3);
            cc[j + 3] += vec_type::hadd(t4);
        }

        for (; i + vec_size - 1 < m; i += vec_size) {
            auto x1 = vec_type::loadu(aa + i + vec_size * 0);

            auto r11 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 0) * m));
            auto r21 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 1) * m));
            auto r31 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 2) * m));
            auto r41 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 3) * m));

            cc[j + 0] += vec_type::hadd(r11);
            cc[j + 1] += vec_type::hadd(r21);
            cc[j + 2] += vec_type::hadd(r31);
            cc[j + 3] += vec_type::hadd(r41);
        }

        for (; i < m; ++i) {
            cc[j + 0] += aa[i] * bb[i + (j + 0) * m];
            cc[j + 1] += aa[i] * bb[i + (j + 1) * m];
            cc[j + 2] += aa[i] * bb[i + (j + 2) * m];
            cc[j + 3] += aa[i] * bb[i + (j + 3) * m];
        }
    }

    for (; j + 1 < n; j += 2) {
        size_t i = 0;

        for (; i + vec_size * 4 - 1 < m; i += vec_size * 4) {
            auto x1 = vec_type::loadu(aa + i + vec_size * 0);
            auto x2 = vec_type::loadu(aa + i + vec_size * 1);
            auto x3 = vec_type::loadu(aa + i + vec_size * 2);
            auto x4 = vec_type::loadu(aa + i + vec_size * 3);

            auto r11 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 0) * m));
            auto r12 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 0) * m));
            auto r13 = vec_type::mul(x3, vec_type::loadu(bb + i + vec_size * 2 + (j + 0) * m));
            auto r14 = vec_type::mul(x4, vec_type::loadu(bb + i + vec_size * 3 + (j + 0) * m));

            auto r21 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 1) * m));
            auto r22 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 1) * m));
            auto r23 = vec_type::mul(x3, vec_type::loadu(bb + i + vec_size * 2 + (j + 1) * m));
            auto r24 = vec_type::mul(x4, vec_type::loadu(bb + i + vec_size * 3 + (j + 1) * m));

            auto t11 = vec_type::add(r11, r12);
            auto t12 = vec_type::add(r13, r14);

            auto t21 = vec_type::add(r21, r22);
            auto t22 = vec_type::add(r23, r24);

            auto r1 = vec_type::add(t11, t12);
            auto r2 = vec_type::add(t21, t22);

            cc[j + 0] += vec_type::hadd(r1);
            cc[j + 1] += vec_type::hadd(r2);
        }

        for (; i + vec_size * 2 - 1 < m; i += vec_size * 2) {
            auto x1 = vec_type::loadu(aa + i + vec_size * 0);
            auto x2 = vec_type::loadu(aa + i + vec_size * 1);

            auto r11 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 0) * m));
            auto r12 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 0) * m));

            auto r21 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 1) * m));
            auto r22 = vec_type::mul(x2, vec_type::loadu(bb + i + vec_size * 1 + (j + 1) * m));

            auto t1 = vec_type::add(r11, r12);
            auto t2 = vec_type::add(r21, r22);

            cc[j + 0] += vec_type::hadd(t1);
            cc[j + 1] += vec_type::hadd(t2);
        }

        for (; i + vec_size - 1 < m; i += vec_size) {
            auto x1 = vec_type::loadu(aa + i + vec_size * 0);

            auto r11 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 0) * m));

            auto r21 = vec_type::mul(x1, vec_type::loadu(bb + i + vec_size * 0 + (j + 1) * m));

            cc[j + 0] += vec_type::hadd(r11);
            cc[j + 1] += vec_type::hadd(r21);
        }

        for (; i < m; ++i) {
            cc[j + 0] += aa[i] * bb[i + (j + 0) * m];
            cc[j + 1] += aa[i] * bb[i + (j + 1) * m];
        }
    }

    for (; j < n; ++j) {
        size_t i = 0;

        for (; i + vec_size * 4 - 1 < m; i += vec_size * 4) {
            auto r1 = vec_type::mul(vec_type::loadu(aa + i + vec_size * 0), vec_type::loadu(bb + i + vec_size * 0 + j * m));
            auto r2 = vec_type::mul(vec_type::loadu(aa + i + vec_size * 1), vec_type::loadu(bb + i + vec_size * 1 + j * m));
            auto r3 = vec_type::mul(vec_type::loadu(aa + i + vec_size * 2), vec_type::loadu(bb + i + vec_size * 2 + j * m));
            auto r4 = vec_type::mul(vec_type::loadu(aa + i + vec_size * 3), vec_type::loadu(bb + i + vec_size * 3 + j * m));

            r1 = vec_type::add(r1, r2);
            r3 = vec_type::add(r3, r4);

            r1 = vec_type::add(r1, r3);

            cc[j] += vec_type::hadd(r1);
        }

        for (; i + vec_size * 2 - 1 < m; i += vec_size * 2) {
            auto r1 = vec_type::mul(vec_type::loadu(aa + i + vec_size * 0), vec_type::loadu(bb + i + vec_size * 0 + j * m));
            auto r2 = vec_type::mul(vec_type::loadu(aa + i + vec_size * 1), vec_type::loadu(bb + i + vec_size * 1 + j * m));

            r1 = vec_type::add(r1, r2);

            cc[j] += vec_type::hadd(r1);
        }

        for (; i + vec_size - 1 < m; i += vec_size) {
            auto r1 = vec_type::mul(vec_type::loadu(aa + i + vec_size * 0), vec_type::loadu(bb + i + vec_size * 0 + j * m));
            cc[j] += vec_type::hadd(r1);
        }

        for (; i < m; ++i) {
            cc[j] += aa[i] * bb[i + j * m];
        }
    }
}

/*!
 * \brief Optimized version of GEVM for row major version
 * \param a The lhs vector
 * \param b The rhs matrix
 * \param c The result vector
 */
template <typename A, typename B, typename C>
void gevm(A&& a, B&& b, C&& c) {
    if constexpr (vec_enabled && vectorize_impl && all_homogeneous<A, B, C> && all_vectorizable<vector_mode, A, B, C>) {
        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();

        const auto m = rows(b);
        const auto n = columns(b);

        if constexpr (is_row_major<B>) {
            if (etl::size(b) < gevm_rm_small_threshold) {
                gevm_small_kernel_rr<default_vec>(a.memory_start(), m, n, b.memory_start(), c);
            } else {
                gevm_large_kernel_rr<default_vec>(a.memory_start(), m, n, b.memory_start(), c);
            }
        } else {
            if (etl::size(b) < gevm_cm_small_threshold) {
                gevm_small_kernel_cc<default_vec>(a.memory_start(), m, n, b.memory_start(), c.memory_start());
            } else {
                c = 0;

                gevm_large_kernel_cc<default_vec>(a.memory_start(), m, n, b.memory_start(), c.memory_start());
            }
        }

        c.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid operation called vec::gevm");
    }
}

// Versions with transposition

/*!
 * \brief Optimized version of GEVM for row major version
 * \param a The lhs vector
 * \param b The rhs matrix
 * \param c The result vector
 */
template <typename A, typename B, typename C>
void gevm_t(A&& a, B&& b, C&& c) {
    if constexpr (vec_enabled && vectorize_impl && all_homogeneous<A, B, C> && all_vectorizable<vector_mode, A, B, C>) {
        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();

        const auto m = rows(b);
        const auto n = columns(b);

        if constexpr (is_row_major<B>) {
            if (etl::size(b) < gevm_rm_small_threshold) {
                gevm_small_kernel_cc<default_vec>(a.memory_start(), n, m, b.memory_start(), c.memory_start());
            } else {
                c = 0;

                gevm_large_kernel_cc<default_vec>(a.memory_start(), n, m, b.memory_start(), c.memory_start());
            }
        } else {
            if (etl::size(b) < gevm_cm_small_threshold) {
                gevm_small_kernel_rr<default_vec>(a.memory_start(), n, m, b.memory_start(), c);
            } else {
                c = 0;

                gevm_large_kernel_rr<default_vec>(a.memory_start(), n, m, b.memory_start(), c);
            }
        }

        c.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid operation called vec::gevm");
    }
}

} //end of namespace etl::impl::vec
