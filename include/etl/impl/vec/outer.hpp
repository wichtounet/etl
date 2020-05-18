//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the outer product
 */

#pragma once

namespace etl::impl::vec {

/*!
 * \brief Compute the batch outer product of a and b and store the result in c
 * \param lhs The a expression
 * \param rhs The b expression
 * \param result The c expression
 */
template <typename V, typename L, typename R, typename C>
void batch_outer_impl(const L& lhs, const R& rhs, C&& result) {
    using vec_type = V;
    using T        = value_t<L>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const auto B = etl::dim<0>(lhs);
    const auto M = etl::dim<0>(result);
    const auto N = etl::dim<1>(result);

    lhs.ensure_cpu_up_to_date();
    rhs.ensure_cpu_up_to_date();

    // TODO Ideally, we would need a kernel for very small matrices
    // TODO If N is a multiple of the vector size, this would only perform
    // aligned loads and then we can use load/store instead of loadu/storeu

    result = 0;

    auto batch_fun_m = [&](const size_t first, const size_t last) {
        for (size_t i = first; i < last; ++i) {
            size_t b = 0;

            for (; b + 7 < B; b += 8) {
                const auto b1 = b + 0;
                const auto b2 = b + 1;
                const auto b3 = b + 2;
                const auto b4 = b + 3;
                const auto b5 = b + 4;
                const auto b6 = b + 5;
                const auto b7 = b + 6;
                const auto b8 = b + 7;

                auto factor1 = lhs(b1, i);
                auto factor2 = lhs(b2, i);
                auto factor3 = lhs(b3, i);
                auto factor4 = lhs(b4, i);
                auto factor5 = lhs(b5, i);
                auto factor6 = lhs(b6, i);
                auto factor7 = lhs(b7, i);
                auto factor8 = lhs(b8, i);

                auto f1 = vec_type::set(factor1);
                auto f2 = vec_type::set(factor2);
                auto f3 = vec_type::set(factor3);
                auto f4 = vec_type::set(factor4);
                auto f5 = vec_type::set(factor5);
                auto f6 = vec_type::set(factor6);
                auto f7 = vec_type::set(factor7);
                auto f8 = vec_type::set(factor8);

                size_t j = 0;

                for (; j + 2 * vec_size - 1 < N; j += 2 * vec_size) {
                    auto r11 = result.template loadu<vec_type>(i * N + j + 0 * vec_size);
                    auto r21 = result.template loadu<vec_type>(i * N + j + 1 * vec_size);

                    auto a11 = rhs.template loadu<vec_type>(b1 * N + j + 0 * vec_size);
                    auto a12 = rhs.template loadu<vec_type>(b2 * N + j + 0 * vec_size);
                    auto a13 = rhs.template loadu<vec_type>(b3 * N + j + 0 * vec_size);
                    auto a14 = rhs.template loadu<vec_type>(b4 * N + j + 0 * vec_size);
                    auto a15 = rhs.template loadu<vec_type>(b5 * N + j + 0 * vec_size);
                    auto a16 = rhs.template loadu<vec_type>(b6 * N + j + 0 * vec_size);
                    auto a17 = rhs.template loadu<vec_type>(b7 * N + j + 0 * vec_size);
                    auto a18 = rhs.template loadu<vec_type>(b8 * N + j + 0 * vec_size);

                    r11 = vec_type::fmadd(f1, a11, r11);
                    r11 = vec_type::fmadd(f2, a12, r11);
                    r11 = vec_type::fmadd(f3, a13, r11);
                    r11 = vec_type::fmadd(f4, a14, r11);
                    r11 = vec_type::fmadd(f5, a15, r11);
                    r11 = vec_type::fmadd(f6, a16, r11);
                    r11 = vec_type::fmadd(f7, a17, r11);
                    r11 = vec_type::fmadd(f8, a18, r11);

                    auto a21 = rhs.template loadu<vec_type>(b1 * N + j + 1 * vec_size);
                    auto a22 = rhs.template loadu<vec_type>(b2 * N + j + 1 * vec_size);
                    auto a23 = rhs.template loadu<vec_type>(b3 * N + j + 1 * vec_size);
                    auto a24 = rhs.template loadu<vec_type>(b4 * N + j + 1 * vec_size);
                    auto a25 = rhs.template loadu<vec_type>(b5 * N + j + 1 * vec_size);
                    auto a26 = rhs.template loadu<vec_type>(b6 * N + j + 1 * vec_size);
                    auto a27 = rhs.template loadu<vec_type>(b7 * N + j + 1 * vec_size);
                    auto a28 = rhs.template loadu<vec_type>(b8 * N + j + 1 * vec_size);

                    r21 = vec_type::fmadd(f1, a21, r21);
                    r21 = vec_type::fmadd(f2, a22, r21);
                    r21 = vec_type::fmadd(f3, a23, r21);
                    r21 = vec_type::fmadd(f4, a24, r21);
                    r21 = vec_type::fmadd(f5, a25, r21);
                    r21 = vec_type::fmadd(f6, a26, r21);
                    r21 = vec_type::fmadd(f7, a27, r21);
                    r21 = vec_type::fmadd(f8, a28, r21);

                    result.template storeu<vec_type>(r11, i * N + j + 0 * vec_size);
                    result.template storeu<vec_type>(r21, i * N + j + 1 * vec_size);
                }

                for (; j + vec_size - 1 < N; j += vec_size) {
                    auto r1 = result.template loadu<vec_type>(i * N + j);

                    auto a1 = rhs.template loadu<vec_type>(b1 * N + j);
                    auto a2 = rhs.template loadu<vec_type>(b2 * N + j);
                    auto a3 = rhs.template loadu<vec_type>(b3 * N + j);
                    auto a4 = rhs.template loadu<vec_type>(b4 * N + j);
                    auto a5 = rhs.template loadu<vec_type>(b5 * N + j);
                    auto a6 = rhs.template loadu<vec_type>(b6 * N + j);
                    auto a7 = rhs.template loadu<vec_type>(b7 * N + j);
                    auto a8 = rhs.template loadu<vec_type>(b8 * N + j);

                    r1 = vec_type::fmadd(f1, a1, r1);
                    r1 = vec_type::fmadd(f2, a2, r1);
                    r1 = vec_type::fmadd(f3, a3, r1);
                    r1 = vec_type::fmadd(f4, a4, r1);
                    r1 = vec_type::fmadd(f5, a5, r1);
                    r1 = vec_type::fmadd(f6, a6, r1);
                    r1 = vec_type::fmadd(f7, a7, r1);
                    r1 = vec_type::fmadd(f8, a8, r1);

                    result.template storeu<vec_type>(r1, i * N + j);
                }

                for (; j + 1 < N; j += 2) {
                    result(i, j + 0) += factor1 * rhs(b1, j + 0);
                    result(i, j + 0) += factor2 * rhs(b2, j + 0);
                    result(i, j + 0) += factor3 * rhs(b3, j + 0);
                    result(i, j + 0) += factor4 * rhs(b4, j + 0);
                    result(i, j + 0) += factor5 * rhs(b5, j + 0);
                    result(i, j + 0) += factor6 * rhs(b6, j + 0);
                    result(i, j + 0) += factor7 * rhs(b7, j + 0);
                    result(i, j + 0) += factor8 * rhs(b8, j + 0);

                    result(i, j + 1) += factor1 * rhs(b1, j + 1);
                    result(i, j + 1) += factor2 * rhs(b2, j + 1);
                    result(i, j + 1) += factor3 * rhs(b3, j + 1);
                    result(i, j + 1) += factor4 * rhs(b4, j + 1);
                    result(i, j + 1) += factor5 * rhs(b5, j + 1);
                    result(i, j + 1) += factor6 * rhs(b6, j + 1);
                    result(i, j + 1) += factor7 * rhs(b7, j + 1);
                    result(i, j + 1) += factor8 * rhs(b8, j + 1);
                }

                if (j < N) {
                    result(i, j) += factor1 * rhs(b1, j);
                    result(i, j) += factor2 * rhs(b2, j);
                    result(i, j) += factor3 * rhs(b3, j);
                    result(i, j) += factor4 * rhs(b4, j);
                    result(i, j) += factor5 * rhs(b5, j);
                    result(i, j) += factor6 * rhs(b6, j);
                    result(i, j) += factor7 * rhs(b7, j);
                    result(i, j) += factor8 * rhs(b8, j);
                }
            }

            for (; b + 3 < B; b += 4) {
                const auto b1 = b + 0;
                const auto b2 = b + 1;
                const auto b3 = b + 2;
                const auto b4 = b + 3;

                auto factor1 = lhs(b1, i);
                auto factor2 = lhs(b2, i);
                auto factor3 = lhs(b3, i);
                auto factor4 = lhs(b4, i);

                auto f1 = vec_type::set(factor1);
                auto f2 = vec_type::set(factor2);
                auto f3 = vec_type::set(factor3);
                auto f4 = vec_type::set(factor4);

                size_t j = 0;

                for (; j + vec_size - 1 < N; j += vec_size) {
                    auto r1 = result.template loadu<vec_type>(i * N + j);

                    auto a1 = rhs.template loadu<vec_type>(b1 * N + j);
                    auto a2 = rhs.template loadu<vec_type>(b2 * N + j);
                    auto a3 = rhs.template loadu<vec_type>(b3 * N + j);
                    auto a4 = rhs.template loadu<vec_type>(b4 * N + j);

                    r1 = vec_type::fmadd(f1, a1, r1);
                    r1 = vec_type::fmadd(f2, a2, r1);
                    r1 = vec_type::fmadd(f3, a3, r1);
                    r1 = vec_type::fmadd(f4, a4, r1);

                    result.template storeu<vec_type>(r1, i * N + j);
                }

                for (; j + 1 < N; j += 2) {
                    result(i, j + 0) += factor1 * rhs(b1, j + 0);
                    result(i, j + 0) += factor2 * rhs(b2, j + 0);
                    result(i, j + 0) += factor3 * rhs(b3, j + 0);
                    result(i, j + 0) += factor4 * rhs(b4, j + 0);

                    result(i, j + 1) += factor1 * rhs(b1, j + 1);
                    result(i, j + 1) += factor2 * rhs(b2, j + 1);
                    result(i, j + 1) += factor3 * rhs(b3, j + 1);
                    result(i, j + 1) += factor4 * rhs(b4, j + 1);
                }

                if (j < N) {
                    result(i, j) += factor1 * rhs(b1, j);
                    result(i, j) += factor2 * rhs(b2, j);
                    result(i, j) += factor3 * rhs(b3, j);
                    result(i, j) += factor4 * rhs(b4, j);
                }
            }

            for (; b < B; ++b) {
                auto factor1 = lhs(b, i);

                auto f1 = vec_type::set(factor1);

                size_t j = 0;

                for (; j + vec_size - 1 < N; j += vec_size) {
                    auto r1 = result.template loadu<vec_type>(i * N + j);

                    auto a1 = rhs.template loadu<vec_type>(b * N + j);

                    r1 = vec_type::fmadd(f1, a1, r1);

                    result.template storeu<vec_type>(r1, i * N + j);
                }

                for (; j + 1 < N; j += 2) {
                    result(i, j + 0) += factor1 * rhs(b, j + 0);
                    result(i, j + 1) += factor1 * rhs(b, j + 1);
                }

                if (j < N) {
                    result(i, j) += factor1 * rhs(b, j);
                }
            }
        }
    };

    engine_dispatch_1d(batch_fun_m, 0, M, engine_select_parallel(M, 2) && N > 25);

    result.invalidate_gpu();
}

/*!
 * \brief Compute the batch outer product of a and b and store the result in c
 * \param lhs The a expression
 * \param rhs The b expression
 * \param c The c expression
 */
template <typename A, typename B, typename C>
void batch_outer(const A& lhs, const B& rhs, C&& c) {
    if constexpr (all_vectorizable<vector_mode, A, B, C>) {
        batch_outer_impl<default_vec>(lhs, rhs, c);
    } else {
        cpp_unreachable("Invalid call to vec::batch_outer");
    }
}

} //end of namespace etl::impl::vec
