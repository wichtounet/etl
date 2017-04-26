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
 * \brief Optimized version of small GEVM for row major version
 * \param aa The lhs vector
 * \param bb The rhs matrix
 * \param cc The result vector
 */
template <typename V, typename T>
void gevm_small_kernel_rr(const T* aa, size_t m, size_t n, const T* bb, T* cc) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    size_t j = 0;

    for (; j + vec_size * 8 - 1 < n; j += vec_size * 8) {
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

        vec_type::storeu(cc + j + 0 * vec_size, r1);
        vec_type::storeu(cc + j + 1 * vec_size, r2);
        vec_type::storeu(cc + j + 2 * vec_size, r3);
        vec_type::storeu(cc + j + 3 * vec_size, r4);
        vec_type::storeu(cc + j + 4 * vec_size, r5);
        vec_type::storeu(cc + j + 5 * vec_size, r6);
        vec_type::storeu(cc + j + 6 * vec_size, r7);
        vec_type::storeu(cc + j + 7 * vec_size, r8);
    }

    for (; j + vec_size * 4 - 1 < n; j += vec_size * 4) {
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

        vec_type::storeu(cc + j + 0 * vec_size, r1);
        vec_type::storeu(cc + j + 1 * vec_size, r2);
        vec_type::storeu(cc + j + 2 * vec_size, r3);
        vec_type::storeu(cc + j + 3 * vec_size, r4);
    }

    for (; j + vec_size - 1 < n; j += vec_size) {
        auto r1 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(aa[k]);

            auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);

            r1 = vec_type::fmadd(a1, b1, r1);
        }

        vec_type::storeu(cc + j + 0 * vec_size, r1);
    }

    for (; j < n; j++) {
        auto value = T();

        for (size_t k = 0; k < m; k++) {
            value += aa[k] * bb[k * n + j];
        }

        cc[j] = value;
    }
}

/*!
 * \brief Optimized version of large GEVM for row major version
 * \param aa The lhs vector
 * \param bb The rhs matrix
 * \param cc The result vector
 */
template <typename V, typename T>
void gevm_large_kernel_rr(const T* aa, size_t m, size_t n, const T* bb, T* cc) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t n_block = (32 * 1024) / sizeof(T);
    const size_t m_block = n < n_block ? 8 : 4;

    for (size_t block_j = 0; block_j < n; block_j += n_block) {
        for (size_t block_k = 0; block_k < m; block_k += m_block) {
            const size_t m_end = std::min(block_k + m_block, m);
            const size_t n_end = std::min(block_j + n_block, n) & size_t(-vec_size);

            size_t j = block_j;

            // 8-Unrolled Vectorized loop
            for (; j + vec_size * 8 - 1 < n_end; j += vec_size * 8) {
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

                vec_type::storeu(cc + j + 0 * vec_size, vec_type::add(r1, vec_type::loadu(cc + j + 0 * vec_size)));
                vec_type::storeu(cc + j + 1 * vec_size, vec_type::add(r2, vec_type::loadu(cc + j + 1 * vec_size)));
                vec_type::storeu(cc + j + 2 * vec_size, vec_type::add(r3, vec_type::loadu(cc + j + 2 * vec_size)));
                vec_type::storeu(cc + j + 3 * vec_size, vec_type::add(r4, vec_type::loadu(cc + j + 3 * vec_size)));
                vec_type::storeu(cc + j + 4 * vec_size, vec_type::add(r5, vec_type::loadu(cc + j + 4 * vec_size)));
                vec_type::storeu(cc + j + 5 * vec_size, vec_type::add(r6, vec_type::loadu(cc + j + 5 * vec_size)));
                vec_type::storeu(cc + j + 6 * vec_size, vec_type::add(r7, vec_type::loadu(cc + j + 6 * vec_size)));
                vec_type::storeu(cc + j + 7 * vec_size, vec_type::add(r8, vec_type::loadu(cc + j + 7 * vec_size)));
            }

            // 4-Unrolled vectorized loop
            for (; j + vec_size * 4 - 1 < n_end; j += vec_size * 4) {
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

                vec_type::storeu(cc + j + 0 * vec_size, vec_type::add(r1, vec_type::loadu(cc + j + 0 * vec_size)));
                vec_type::storeu(cc + j + 1 * vec_size, vec_type::add(r2, vec_type::loadu(cc + j + 1 * vec_size)));
                vec_type::storeu(cc + j + 2 * vec_size, vec_type::add(r3, vec_type::loadu(cc + j + 2 * vec_size)));
                vec_type::storeu(cc + j + 3 * vec_size, vec_type::add(r4, vec_type::loadu(cc + j + 3 * vec_size)));
            }

            // Base vectorized loop
            for (; j + vec_size - 1 < n_end; j += vec_size) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(aa[k]);
                    auto b1 = vec_type::loadu(bb + k * n + j + 0 * vec_size);
                    r1 = vec_type::fmadd(a1, b1, r1);
                }

                vec_type::storeu(cc + j + 0 * vec_size, vec_type::add(r1, vec_type::loadu(cc + j + 0 * vec_size)));
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
 * \brief Optimized version of GEVM for row major version
 * \param a The lhs vector
 * \param b The rhs matrix
 * \param c The result vector
 */
template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gevm(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    const auto m = rows(b);
    const auto n = columns(b);

    if(etl::size(b) < gevm_small_threshold){
        gevm_small_kernel_rr<default_vec>(a.memory_start(), m, n, b.memory_start(), c.memory_start());
    } else {
        c = 0;

        gevm_large_kernel_rr<default_vec>(a.memory_start(), m, n, b.memory_start(), c.memory_start());
    }

    c.invalidate_gpu();
}

/*!
 * \brief Unoptimized version of GEVM for column major version
 * \param a The lhs vector
 * \param b The rhs matrix
 * \param c The result vector
 */
template <typename A, typename B, typename C, cpp_disable_if((all_row_major<A, B, C>::value))>
void gevm(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    c = 0;

    for (std::size_t k = 0; k < etl::dim<0>(a); k++) {
        for (std::size_t j = 0; j < columns(b); j++) {
            c(j) += a(k) * b(k, j);
        }
    }
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
