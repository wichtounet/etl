//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//======================================================================= #pragma once

namespace etl {

namespace impl {

namespace vec {

template <typename V, typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemv(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const auto m = rows(a);
    const auto n = columns(a);

    auto fun_i = [&](const size_t first, const size_t last){
        for (size_t i = first; i < last; ++i) {
            //TODO Add FMA (fmadd) support to vectorization

            size_t k = 0;

            auto r1 = vec_type::template zero<T>();
            auto r2 = vec_type::template zero<T>();
            auto r3 = vec_type::template zero<T>();
            auto r4 = vec_type::template zero<T>();

            for (; k + (vec_size * 4) - 1 < n; k += 4 * vec_size) {
                auto a1 = a.template loadu<vec_type>(i * n + k + 0 * vec_size);
                auto a2 = a.template loadu<vec_type>(i * n + k + 1 * vec_size);
                auto a3 = a.template loadu<vec_type>(i * n + k + 2 * vec_size);
                auto a4 = a.template loadu<vec_type>(i * n + k + 3 * vec_size);

                auto b1 = b.template loadu<vec_type>(k + 0 * vec_size);
                auto b2 = b.template loadu<vec_type>(k + 1 * vec_size);
                auto b3 = b.template loadu<vec_type>(k + 2 * vec_size);
                auto b4 = b.template loadu<vec_type>(k + 3 * vec_size);

                auto t1 = vec_type::template mul<Cx>(a1, b1);
                auto t2 = vec_type::template mul<Cx>(a2, b2);
                auto t3 = vec_type::template mul<Cx>(a3, b3);
                auto t4 = vec_type::template mul<Cx>(a4, b4);

                r1 = vec_type::add(r1, t1);
                r2 = vec_type::add(r2, t2);
                r3 = vec_type::add(r3, t3);
                r4 = vec_type::add(r4, t4);
            }

            for (; k + vec_size - 1 < n; k += vec_size) {
                auto a1 = a.template loadu<vec_type>(i * n + k);
                auto b1 = b.template loadu<vec_type>(k);

                auto t1 = vec_type::template mul<Cx>(a1, b1);
                r1 = vec_type::add(r1, t1);
            }

            auto result =
                  vec_type::template hadd<T>(r1)
                + vec_type::template hadd<T>(r2)
                + vec_type::template hadd<T>(r3)
                + vec_type::template hadd<T>(r4);

            for (; k < n; ++k) {
                result += a(i, k) * b(k);
            }

            c[i] = result;
        }
    };

    dispatch_1d_any(select_parallel(m, 300), fun_i, 0, m);
}

template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemv(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    gemv<default_vec>(a, b, c);
}

// Default, unoptimized should not be called unless in tests
template <typename A, typename B, typename C, cpp_disable_if((all_row_major<A, B, C>::value))>
void gemv(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    const auto m = rows(a);
    const auto n = columns(a);

    c = 0;

    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < n; k++) {
            c(i) += a(i, k) * b(k);
        }
    }
}

template <typename V, typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gevm_small_kernel(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const auto m = rows(b);
    const auto n = columns(b);

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
            auto a1 = vec_type::set(a[k]);

            auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
            auto b2 = b.template loadu<vec_type>(k * n + j + 1 * vec_size);
            auto b3 = b.template loadu<vec_type>(k * n + j + 2 * vec_size);
            auto b4 = b.template loadu<vec_type>(k * n + j + 3 * vec_size);
            auto b5 = b.template loadu<vec_type>(k * n + j + 4 * vec_size);
            auto b6 = b.template loadu<vec_type>(k * n + j + 5 * vec_size);
            auto b7 = b.template loadu<vec_type>(k * n + j + 6 * vec_size);
            auto b8 = b.template loadu<vec_type>(k * n + j + 7 * vec_size);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
            r5 = vec_type::add(r5, vec_type::template mul<Cx>(a1, b5));
            r6 = vec_type::add(r6, vec_type::template mul<Cx>(a1, b6));
            r7 = vec_type::add(r7, vec_type::template mul<Cx>(a1, b7));
            r8 = vec_type::add(r8, vec_type::template mul<Cx>(a1, b8));
        }

        c.template storeu<vec_type>(r1, j + 0 * vec_size);
        c.template storeu<vec_type>(r2, j + 1 * vec_size);
        c.template storeu<vec_type>(r3, j + 2 * vec_size);
        c.template storeu<vec_type>(r4, j + 3 * vec_size);
        c.template storeu<vec_type>(r5, j + 4 * vec_size);
        c.template storeu<vec_type>(r6, j + 5 * vec_size);
        c.template storeu<vec_type>(r7, j + 6 * vec_size);
        c.template storeu<vec_type>(r8, j + 7 * vec_size);
    }

    for (; j + vec_size * 4 - 1 < n; j += vec_size * 4) {
        auto r1 = vec_type::template zero<T>();
        auto r2 = vec_type::template zero<T>();
        auto r3 = vec_type::template zero<T>();
        auto r4 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(a[k]);

            auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
            auto b2 = b.template loadu<vec_type>(k * n + j + 1 * vec_size);
            auto b3 = b.template loadu<vec_type>(k * n + j + 2 * vec_size);
            auto b4 = b.template loadu<vec_type>(k * n + j + 3 * vec_size);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
            r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
            r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
            r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
        }

        c.template storeu<vec_type>(r1, j + 0 * vec_size);
        c.template storeu<vec_type>(r2, j + 1 * vec_size);
        c.template storeu<vec_type>(r3, j + 2 * vec_size);
        c.template storeu<vec_type>(r4, j + 3 * vec_size);
    }

    for (; j + vec_size - 1 < n; j += vec_size) {
        auto r1 = vec_type::template zero<T>();

        for (size_t k = 0; k < m; k++) {
            auto a1 = vec_type::set(a[k]);

            auto b1 = b.template loadu<vec_type>(k * n + j);

            r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
        }

        c.template storeu<vec_type>(r1, j);
    }

    for (; j < n; j++) {
        auto value = T();

        for (size_t k = 0; k < m; k++) {
            value += a(k) * b(k, j);
        }

        c[j] = 0;
    }
}

template <typename V, typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gevm_large_kernel(const A& a, const B& b, C& c) {
    using vec_type = V;
    using T        = value_t<A>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const auto m = rows(b);
    const auto n = columns(b);

    const size_t n_block = (32 * 1024) / sizeof(T);
    const size_t m_block = n < n_block ? 8UL : 4UL;

    c = 0;

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
                    auto a1 = vec_type::set(a[k]);

                    auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
                    auto b2 = b.template loadu<vec_type>(k * n + j + 1 * vec_size);
                    auto b3 = b.template loadu<vec_type>(k * n + j + 2 * vec_size);
                    auto b4 = b.template loadu<vec_type>(k * n + j + 3 * vec_size);
                    auto b5 = b.template loadu<vec_type>(k * n + j + 4 * vec_size);
                    auto b6 = b.template loadu<vec_type>(k * n + j + 5 * vec_size);
                    auto b7 = b.template loadu<vec_type>(k * n + j + 6 * vec_size);
                    auto b8 = b.template loadu<vec_type>(k * n + j + 7 * vec_size);

                    r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                    r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
                    r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
                    r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
                    r5 = vec_type::add(r5, vec_type::template mul<Cx>(a1, b5));
                    r6 = vec_type::add(r6, vec_type::template mul<Cx>(a1, b6));
                    r7 = vec_type::add(r7, vec_type::template mul<Cx>(a1, b7));
                    r8 = vec_type::add(r8, vec_type::template mul<Cx>(a1, b8));
                }

                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 0 * vec_size), r1), j + 0 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 1 * vec_size), r2), j + 1 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 2 * vec_size), r3), j + 2 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 3 * vec_size), r4), j + 3 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 4 * vec_size), r5), j + 4 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 5 * vec_size), r6), j + 5 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 6 * vec_size), r7), j + 6 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 7 * vec_size), r8), j + 7 * vec_size);
            }

            // 4-Unrolled vectorized loop
            for (; j + vec_size * 4 - 1 < n_end; j += vec_size * 4) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(a[k]);

                    auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
                    auto b2 = b.template loadu<vec_type>(k * n + j + 1 * vec_size);
                    auto b3 = b.template loadu<vec_type>(k * n + j + 2 * vec_size);
                    auto b4 = b.template loadu<vec_type>(k * n + j + 3 * vec_size);

                    r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                    r2 = vec_type::add(r2, vec_type::template mul<Cx>(a1, b2));
                    r3 = vec_type::add(r3, vec_type::template mul<Cx>(a1, b3));
                    r4 = vec_type::add(r4, vec_type::template mul<Cx>(a1, b4));
                }

                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 0 * vec_size), r1), j + 0 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 1 * vec_size), r2), j + 1 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 2 * vec_size), r3), j + 2 * vec_size);
                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 3 * vec_size), r4), j + 3 * vec_size);
            }

            // Base vectorized loop
            for (; j + vec_size - 1 < n_end; j += vec_size) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = block_k; k < m_end; ++k) {
                    auto a1 = vec_type::set(a[k]);
                    auto b1 = b.template loadu<vec_type>(k * n + j + 0 * vec_size);
                    r1 = vec_type::add(r1, vec_type::template mul<Cx>(a1, b1));
                }

                c.template storeu<vec_type>(vec_type::add(c.template loadu<vec_type>(j + 0 * vec_size), r1), j + 0 * vec_size);
            }

            // Remainder non-vectorized loop
            for (; j < n_end; ++j) {
                auto r1 = T();

                for (size_t k = block_k; k < m_end; ++k) {
                    r1 += a[k] * b(k, j);
                }

                c[j] += r1;
            }
        }
    }
}

template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gevm(A&& a, B&& b, C&& c) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    if(etl::size(b) < gevm_small_threshold){
        gevm_small_kernel<default_vec>(a, b, c);
    } else {
        gevm_large_kernel<default_vec>(a, b, c);
    }
}

// Default, unoptimized should not be called unless in tests
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
