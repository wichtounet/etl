//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

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
    gemv<default_vec>(a, b, c);
}

// Default, unoptimized should not be called unless in tests
template <typename A, typename B, typename C, cpp_disable_if((all_row_major<A, B, C>::value))>
void gemv(A&& a, B&& b, C&& c) {
    const auto m = rows(a);
    const auto n = columns(a);

    c = 0;

    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < n; k++) {
            c(i) += a(i, k) * b(k);
        }
    }
}

} //end of namespace vec

} //end of namespace impl

} //end of namespace etl
