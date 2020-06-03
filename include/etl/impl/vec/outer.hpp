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

#pragma GCC push_options
#pragma GCC optimize ("-fno-aggressive-loop-optimizations")

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

    auto L2 = force_temporary_opp(lhs);
    auto R2 = force_temporary_opp(rhs);

    auto batch_fun_m = [&](const size_t first, const size_t last) {
        size_t i = first;

        for (; i + 1 < last; i += 2) {
            size_t j = 0;

            for (; j + 3 < N; j += 4) {
                size_t b = 0;

                auto xmm1 = vec_type::template zero<T>();
                auto xmm2 = vec_type::template zero<T>();
                auto xmm3 = vec_type::template zero<T>();
                auto xmm4 = vec_type::template zero<T>();
                auto xmm5 = vec_type::template zero<T>();
                auto xmm6 = vec_type::template zero<T>();
                auto xmm7 = vec_type::template zero<T>();
                auto xmm8 = vec_type::template zero<T>();

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto l1 = L2.template loadu<vec_type>((i + 0) * B + b);
                    auto l2 = L2.template loadu<vec_type>((i + 1) * B + b);

                    auto r1 = R2.template loadu<vec_type>((j + 0) * B + b);
                    auto r2 = R2.template loadu<vec_type>((j + 1) * B + b);
                    auto r3 = R2.template loadu<vec_type>((j + 2) * B + b);
                    auto r4 = R2.template loadu<vec_type>((j + 3) * B + b);

                    xmm1 = vec_type::fmadd(l1, r1, xmm1);
                    xmm2 = vec_type::fmadd(l1, r2, xmm2);
                    xmm3 = vec_type::fmadd(l1, r3, xmm3);
                    xmm4 = vec_type::fmadd(l1, r4, xmm4);

                    xmm5 = vec_type::fmadd(l2, r1, xmm5);
                    xmm6 = vec_type::fmadd(l2, r2, xmm6);
                    xmm7 = vec_type::fmadd(l2, r3, xmm7);
                    xmm8 = vec_type::fmadd(l2, r4, xmm8);
                }

                T r1 = vec_type::hadd(xmm1);
                T r2 = vec_type::hadd(xmm2);;
                T r3 = vec_type::hadd(xmm3);;
                T r4 = vec_type::hadd(xmm4);;
                T r5 = vec_type::hadd(xmm5);
                T r6 = vec_type::hadd(xmm6);;
                T r7 = vec_type::hadd(xmm7);;
                T r8 = vec_type::hadd(xmm8);;

                for (; b + 1 < B; b += 2) {
                    r1 += L2(b + 0, i + 0) * R2(b + 0, j + 0);
                    r1 += L2(b + 1, i + 0) * R2(b + 1, j + 0);

                    r2 += L2(b + 0, i + 0) * R2(b + 0, j + 1);
                    r2 += L2(b + 1, i + 0) * R2(b + 1, j + 1);

                    r3 += L2(b + 0, i + 0) * R2(b + 0, j + 2);
                    r3 += L2(b + 1, i + 0) * R2(b + 1, j + 2);

                    r4 += L2(b + 0, i + 0) * R2(b + 0, j + 3);
                    r4 += L2(b + 1, i + 0) * R2(b + 1, j + 3);

                    r5 += L2(b + 0, i + 1) * R2(b + 0, j + 0);
                    r5 += L2(b + 1, i + 1) * R2(b + 1, j + 0);

                    r6 += L2(b + 0, i + 1) * R2(b + 0, j + 1);
                    r6 += L2(b + 1, i + 1) * R2(b + 1, j + 1);

                    r7 += L2(b + 0, i + 1) * R2(b + 0, j + 2);
                    r7 += L2(b + 1, i + 1) * R2(b + 1, j + 2);

                    r8 += L2(b + 0, i + 1) * R2(b + 0, j + 3);
                    r8 += L2(b + 1, i + 1) * R2(b + 1, j + 3);
                }

                if (b < B) {
                    r1 += L2(b, i + 0) * R2(b, j + 0);
                    r2 += L2(b, i + 0) * R2(b, j + 1);
                    r3 += L2(b, i + 0) * R2(b, j + 2);
                    r4 += L2(b, i + 0) * R2(b, j + 3);

                    r5 += L2(b, i + 1) * R2(b, j + 0);
                    r6 += L2(b, i + 1) * R2(b, j + 1);
                    r7 += L2(b, i + 1) * R2(b, j + 2);
                    r8 += L2(b, i + 1) * R2(b, j + 3);
                }

                result(i + 0, j + 0) = r1;
                result(i + 0, j + 1) = r2;
                result(i + 0, j + 2) = r3;
                result(i + 0, j + 3) = r4;

                result(i + 1, j + 0) = r5;
                result(i + 1, j + 1) = r6;
                result(i + 1, j + 2) = r7;
                result(i + 1, j + 3) = r8;
            }

            for (; j + 1 < N; j += 2) {
                size_t b = 0;

                auto xmm1 = vec_type::template zero<T>();
                auto xmm2 = vec_type::template zero<T>();
                auto xmm3 = vec_type::template zero<T>();
                auto xmm4 = vec_type::template zero<T>();

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto l1 = L2.template loadu<vec_type>((i + 0) * B + b);
                    auto l2 = L2.template loadu<vec_type>((i + 1) * B + b);

                    auto r1 = R2.template loadu<vec_type>((j + 0) * B + b);
                    auto r2 = R2.template loadu<vec_type>((j + 1) * B + b);

                    xmm1 = vec_type::fmadd(l1, r1, xmm1);
                    xmm2 = vec_type::fmadd(l1, r2, xmm2);

                    xmm3 = vec_type::fmadd(l2, r1, xmm3);
                    xmm4 = vec_type::fmadd(l2, r2, xmm4);
                }

                T r1 = vec_type::hadd(xmm1);
                T r2 = vec_type::hadd(xmm2);;
                T r3 = vec_type::hadd(xmm3);;
                T r4 = vec_type::hadd(xmm4);;

                for (; b + 1 < B; b += 2) {
                    r1 += L2(b + 0, i + 0) * R2(b + 0, j + 0);
                    r1 += L2(b + 1, i + 0) * R2(b + 1, j + 0);

                    r2 += L2(b + 0, i + 0) * R2(b + 0, j + 1);
                    r2 += L2(b + 1, i + 0) * R2(b + 1, j + 1);

                    r3 += L2(b + 0, i + 1) * R2(b + 0, j + 0);
                    r3 += L2(b + 1, i + 1) * R2(b + 1, j + 0);

                    r4 += L2(b + 0, i + 1) * R2(b + 0, j + 1);
                    r4 += L2(b + 1, i + 1) * R2(b + 1, j + 1);
                }

                if (b < B) {
                    r1 += L2(b, i + 0) * R2(b, j + 0);
                    r2 += L2(b, i + 0) * R2(b, j + 1);
                    r3 += L2(b, i + 1) * R2(b, j + 0);
                    r4 += L2(b, i + 1) * R2(b, j + 1);
                }

                result(i + 0, j + 0) = r1;
                result(i + 0, j + 1) = r2;

                result(i + 1, j + 0) = r3;
                result(i + 1, j + 1) = r4;
            }

            if (j < N) {
                size_t b = 0;

                auto xmm1 = vec_type::template zero<T>();
                auto xmm2 = vec_type::template zero<T>();

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto l1 = L2.template loadu<vec_type>((i + 0) * B + b);
                    auto l2 = L2.template loadu<vec_type>((i + 1) * B + b);

                    auto r1 = R2.template loadu<vec_type>(j * B + b);

                    xmm1 = vec_type::fmadd(l1, r1, xmm1);
                    xmm2 = vec_type::fmadd(l2, r1, xmm2);
                }

                T r1 = vec_type::hadd(xmm1);
                T r2 = vec_type::hadd(xmm2);

                for (; b + 1 < B; b += 2) {
                    r1 += L2(b + 0, i + 0) * R2(b + 0, j);
                    r1 += L2(b + 1, i + 0) * R2(b + 1, j);

                    r2 += L2(b + 0, i + 1) * R2(b + 0, j);
                    r2 += L2(b + 1, i + 1) * R2(b + 1, j);
                }

                if (b < B) {
                    r1 += L2(b, i + 0) * R2(b, j);
                    r2 += L2(b, i + 1) * R2(b, j);
                }

                result(i + 0, j) = r1;
                result(i + 1, j) = r2;
            }
        }

        if (i < last) {
            size_t j = 0;

            for (; j + 1 < N; j += 2) {
                size_t b = 0;

                auto xmm1 = vec_type::template zero<T>();
                auto xmm2 = vec_type::template zero<T>();

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto l1 = L2.template loadu<vec_type>(i * B + b);

                    auto r1 = R2.template loadu<vec_type>((j + 0) * B + b);
                    auto r2 = R2.template loadu<vec_type>((j + 1) * B + b);

                    xmm1 = vec_type::fmadd(l1, r1, xmm1);
                    xmm2 = vec_type::fmadd(l1, r2, xmm2);
                }

                T r1 = vec_type::hadd(xmm1);
                T r2 = vec_type::hadd(xmm2);;

                for (; b + 1 < B; b += 2) {
                    r1 += L2(b + 0, i) * R2(b + 0, j + 0);
                    r1 += L2(b + 1, i) * R2(b + 1, j + 0);

                    r2 += L2(b + 0, i) * R2(b + 0, j + 1);
                    r2 += L2(b + 1, i) * R2(b + 1, j + 1);
                }

                if (b < B) {
                    r1 += L2(b, i) * R2(b, j + 0);
                    r2 += L2(b, i) * R2(b, j + 1);
                }

                result(i, j + 0) = r1;
                result(i, j + 1) = r2;
            }

            for (; j < N; ++j) {
                size_t b = 0;

                auto xmm1 = vec_type::template zero<T>();

                for (; b + vec_size - 1 < B; b += vec_size) {
                    auto l1 = L2.template loadu<vec_type>(i * B + b);
                    auto r1 = R2.template loadu<vec_type>(j * B + b);

                    xmm1 = vec_type::fmadd(l1, r1, xmm1);
                }

                T r1 = vec_type::hadd(xmm1);

                for (; b + 1 < B; b += 2) {
                    r1 += L2(b + 0, i) * R2(b + 0, j);
                    r1 += L2(b + 1, i) * R2(b + 1, j);
                }

                if (b < B) {
                    r1 += L2(b, i) * R2(b, j);
                }

                result(i, j) = r1;
            }
        }
    };

    engine_dispatch_1d(batch_fun_m, 0, M, engine_select_parallel(M, 2) && N > 20);

    result.invalidate_gpu();
}

#pragma GCC pop_options

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
