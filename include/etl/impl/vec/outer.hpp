//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the outer product
 */

#pragma once

namespace etl {

namespace impl {

namespace vec {

/*!
 * \brief Compute the batch outer product of a and b and store the result in c
 * \param lhs The a expression
 * \param rhs The b expression
 * \param result The c expression
 */
template <typename V, typename L, typename R, typename C>
void batch_outer(const L& lhs, const R& rhs, C&& result) {
    using vec_type = V;
    using T        = value_t<L>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const auto B = etl::dim<0>(lhs);
    const auto M = etl::dim<0>(result);
    const auto N = etl::dim<1>(result);

    auto batch_fun_m = [&](const size_t first, const size_t last) {
        for (size_t i = first; i < last; ++i) {
            size_t j = 0;

            for (; j + 7 < N; j += 8) {
                const auto j1 = j + 0;
                const auto j2 = j + 1;
                const auto j3 = j + 2;
                const auto j4 = j + 3;
                const auto j5 = j + 4;
                const auto j6 = j + 5;
                const auto j7 = j + 6;
                const auto j8 = j + 7;

                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                size_t b = 0;

                for (; b + 2 * vec_size - 1 < B; b += 2 * vec_size) {
                    const auto b1 = b + 0;
                    const auto b2 = b + 1;

                    auto a11 = lhs.template loadu<vec_type>(b1 * M + i);
                    auto a21 = lhs.template loadu<vec_type>(b2 * M + i);

                    auto b11 = lhs.template loadu<vec_type>(b1 * N + j1);
                    auto b12 = lhs.template loadu<vec_type>(b1 * N + j2);
                    auto b13 = lhs.template loadu<vec_type>(b1 * N + j3);
                    auto b14 = lhs.template loadu<vec_type>(b1 * N + j4);
                    auto b15 = lhs.template loadu<vec_type>(b1 * N + j5);
                    auto b16 = lhs.template loadu<vec_type>(b1 * N + j6);
                    auto b17 = lhs.template loadu<vec_type>(b1 * N + j7);
                    auto b18 = lhs.template loadu<vec_type>(b1 * N + j8);

                    auto b21 = lhs.template loadu<vec_type>(b2 * N + j1);
                    auto b22 = lhs.template loadu<vec_type>(b2 * N + j2);
                    auto b23 = lhs.template loadu<vec_type>(b2 * N + j3);
                    auto b24 = lhs.template loadu<vec_type>(b2 * N + j4);
                    auto b25 = lhs.template loadu<vec_type>(b2 * N + j5);
                    auto b26 = lhs.template loadu<vec_type>(b2 * N + j6);
                    auto b27 = lhs.template loadu<vec_type>(b2 * N + j7);
                    auto b28 = lhs.template loadu<vec_type>(b2 * N + j8);

                    r1 = vec_type::template fmadd<Cx>(a11, b11, r1);
                    r2 = vec_type::template fmadd<Cx>(a11, b12, r2);
                    r3 = vec_type::template fmadd<Cx>(a11, b13, r3);
                    r4 = vec_type::template fmadd<Cx>(a11, b14, r4);
                    r5 = vec_type::template fmadd<Cx>(a11, b15, r5);
                    r6 = vec_type::template fmadd<Cx>(a11, b16, r6);
                    r7 = vec_type::template fmadd<Cx>(a11, b17, r7);
                    r8 = vec_type::template fmadd<Cx>(a11, b18, r8);

                    r1 = vec_type::template fmadd<Cx>(a21, b21, r1);
                    r2 = vec_type::template fmadd<Cx>(a21, b22, r2);
                    r3 = vec_type::template fmadd<Cx>(a21, b23, r3);
                    r4 = vec_type::template fmadd<Cx>(a21, b24, r4);
                    r5 = vec_type::template fmadd<Cx>(a21, b25, r5);
                    r6 = vec_type::template fmadd<Cx>(a21, b26, r6);
                    r7 = vec_type::template fmadd<Cx>(a21, b27, r7);
                    r8 = vec_type::template fmadd<Cx>(a21, b28, r8);
                }

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto a1 = lhs.template loadu<vec_type>(b * M + i);

                    auto b1 = lhs.template loadu<vec_type>(b * N + j1);
                    auto b2 = lhs.template loadu<vec_type>(b * N + j2);
                    auto b3 = lhs.template loadu<vec_type>(b * N + j3);
                    auto b4 = lhs.template loadu<vec_type>(b * N + j4);
                    auto b5 = lhs.template loadu<vec_type>(b * N + j5);
                    auto b6 = lhs.template loadu<vec_type>(b * N + j6);
                    auto b7 = lhs.template loadu<vec_type>(b * N + j7);
                    auto b8 = lhs.template loadu<vec_type>(b * N + j8);

                    r1 = vec_type::template fmadd<Cx>(a1, b1, r1);
                    r2 = vec_type::template fmadd<Cx>(a1, b2, r2);
                    r3 = vec_type::template fmadd<Cx>(a1, b3, r3);
                    r4 = vec_type::template fmadd<Cx>(a1, b4, r4);
                    r5 = vec_type::template fmadd<Cx>(a1, b5, r5);
                    r6 = vec_type::template fmadd<Cx>(a1, b6, r6);
                    r7 = vec_type::template fmadd<Cx>(a1, b7, r7);
                    r8 = vec_type::template fmadd<Cx>(a1, b8, r8);
                }

                T t1 = T(0);
                T t2 = T(0);
                T t3 = T(0);
                T t4 = T(0);
                T t5 = T(0);
                T t6 = T(0);
                T t7 = T(0);
                T t8 = T(0);

                for (; b < B; ++b) {
                    t1 += lhs(b, i) * rhs(b, j1);
                    t2 += lhs(b, i) * rhs(b, j2);
                    t3 += lhs(b, i) * rhs(b, j3);
                    t4 += lhs(b, i) * rhs(b, j4);
                    t5 += lhs(b, i) * rhs(b, j5);
                    t6 += lhs(b, i) * rhs(b, j6);
                    t7 += lhs(b, i) * rhs(b, j7);
                    t8 += lhs(b, i) * rhs(b, j8);
                }

                result(i, j1) = vec_type::hadd(r1) + t1;
                result(i, j2) = vec_type::hadd(r2) + t2;
                result(i, j3) = vec_type::hadd(r3) + t3;
                result(i, j4) = vec_type::hadd(r4) + t4;
                result(i, j5) = vec_type::hadd(r5) + t5;
                result(i, j6) = vec_type::hadd(r6) + t6;
                result(i, j7) = vec_type::hadd(r7) + t7;
                result(i, j8) = vec_type::hadd(r8) + t8;
            }

            for (; j + 3 < N; j += 4) {
                const auto j1 = j + 0;
                const auto j2 = j + 1;
                const auto j3 = j + 2;
                const auto j4 = j + 3;

                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();

                size_t b = 0;

                for (; b + 2 * vec_size - 1 < B; b += 2 * vec_size) {
                    const auto b1 = b + 0;
                    const auto b2 = b + 1;

                    auto a11 = lhs.template loadu<vec_type>(b1 * M + i);
                    auto a21 = lhs.template loadu<vec_type>(b2 * M + i);

                    auto b11 = lhs.template loadu<vec_type>(b1 * N + j1);
                    auto b12 = lhs.template loadu<vec_type>(b1 * N + j2);
                    auto b13 = lhs.template loadu<vec_type>(b1 * N + j3);
                    auto b14 = lhs.template loadu<vec_type>(b1 * N + j4);

                    auto b21 = lhs.template loadu<vec_type>(b2 * N + j1);
                    auto b22 = lhs.template loadu<vec_type>(b2 * N + j2);
                    auto b23 = lhs.template loadu<vec_type>(b2 * N + j3);
                    auto b24 = lhs.template loadu<vec_type>(b2 * N + j4);

                    r1 = vec_type::template fmadd<Cx>(a11, b11, r1);
                    r2 = vec_type::template fmadd<Cx>(a11, b12, r2);
                    r3 = vec_type::template fmadd<Cx>(a11, b13, r3);
                    r4 = vec_type::template fmadd<Cx>(a11, b14, r4);

                    r1 = vec_type::template fmadd<Cx>(a21, b21, r1);
                    r2 = vec_type::template fmadd<Cx>(a21, b22, r2);
                    r3 = vec_type::template fmadd<Cx>(a21, b23, r3);
                    r4 = vec_type::template fmadd<Cx>(a21, b24, r4);
                }

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto a1 = lhs.template loadu<vec_type>(b * M + i);

                    auto b1 = lhs.template loadu<vec_type>(b * N + j1);
                    auto b2 = lhs.template loadu<vec_type>(b * N + j2);
                    auto b3 = lhs.template loadu<vec_type>(b * N + j3);
                    auto b4 = lhs.template loadu<vec_type>(b * N + j4);

                    r1 = vec_type::template fmadd<Cx>(a1, b1, r1);
                    r2 = vec_type::template fmadd<Cx>(a1, b2, r2);
                    r3 = vec_type::template fmadd<Cx>(a1, b3, r3);
                    r4 = vec_type::template fmadd<Cx>(a1, b4, r4);
                }

                T t1 = T(0);
                T t2 = T(0);
                T t3 = T(0);
                T t4 = T(0);

                for (; b < B; ++b) {
                    t1 += lhs(b, i) * rhs(b, j1);
                    t2 += lhs(b, i) * rhs(b, j2);
                    t3 += lhs(b, i) * rhs(b, j3);
                    t4 += lhs(b, i) * rhs(b, j4);
                }

                result(i, j1) = vec_type::hadd(r1) + t1;
                result(i, j2) = vec_type::hadd(r2) + t2;
                result(i, j3) = vec_type::hadd(r3) + t3;
                result(i, j4) = vec_type::hadd(r4) + t4;
            }

            for (; j + 1 < N; j += 2) {
                const auto j1 = j + 0;
                const auto j2 = j + 1;

                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                size_t b = 0;

                for (; b + 2 * vec_size - 1 < B; b += 2 * vec_size) {
                    const auto b1 = b + 0;
                    const auto b2 = b + 1;

                    auto a11 = lhs.template loadu<vec_type>(b1 * M + i);
                    auto a21 = lhs.template loadu<vec_type>(b2 * M + i);

                    auto b11 = lhs.template loadu<vec_type>(b1 * N + j1);
                    auto b12 = lhs.template loadu<vec_type>(b1 * N + j2);

                    auto b21 = lhs.template loadu<vec_type>(b2 * N + j1);
                    auto b22 = lhs.template loadu<vec_type>(b2 * N + j2);

                    r1 = vec_type::template fmadd<Cx>(a11, b11, r1);
                    r2 = vec_type::template fmadd<Cx>(a11, b12, r2);

                    r1 = vec_type::template fmadd<Cx>(a21, b21, r1);
                    r2 = vec_type::template fmadd<Cx>(a21, b22, r2);
                }

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto a1 = lhs.template loadu<vec_type>(b * M + i);

                    auto b1 = lhs.template loadu<vec_type>(b * N + j1);
                    auto b2 = lhs.template loadu<vec_type>(b * N + j2);

                    r1 = vec_type::template fmadd<Cx>(a1, b1, r1);
                    r2 = vec_type::template fmadd<Cx>(a1, b2, r2);
                }

                T t1 = T(0);
                T t2 = T(0);

                for (; b < B; ++b) {
                    t1 += lhs(b, i) * rhs(b, j1);
                    t2 += lhs(b, i) * rhs(b, j2);
                }

                result(i, j1) = vec_type::hadd(r1) + t1;
                result(i, j2) = vec_type::hadd(r2) + t2;
            }

            if (j < N) {
                auto r1 = vec_type::template zero<T>();

                size_t b = 0;

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto a1 = lhs.template loadu<vec_type>(b * M + i);
                    auto b1 = lhs.template loadu<vec_type>(b * N + j);

                    r1 = vec_type::template fmadd<Cx>(a1, b1, r1);
                }

                T t1 = T(0);

                for (; b < B; ++b) {
                    t1 += lhs(b, i) * rhs(b, j);
                }

                result(i, j) = vec_type::hadd(r1) + t1;
            }
        }
    };

    dispatch_1d_any(select_parallel(M, 2) && N > 25, batch_fun_m, 0, M);
}

template <typename A, typename B, typename C>
void batch_outer(const A& lhs, const B& rhs, C&& c) {
    batch_outer<default_vec>(lhs, rhs, c);
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
