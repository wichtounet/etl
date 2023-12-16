//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

namespace etl {

template <etl_1d A, etl_2d_or_4d B, etl_1d C>
struct batch_k_minus_scale_expr : base_temporary_expr_tern<batch_k_minus_scale_expr<A, B, C>, A, B, C> {
    using value_type  = value_t<A>;                                  ///< The type of value of the expression
    using this_type   = batch_k_minus_scale_expr<A, B, C>;            ///< The type of this expression
    using base_type   = base_temporary_expr_tern<this_type, A, B, C>; ///< The base type
    using left_traits = decay_traits<A>;                             ///< The traits of the sub type

    static constexpr bool D4 = is_4d<B>; ///< If the expression is 4D (instead of 2D)

    static constexpr auto storage_order = left_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable =
               (!D4 && impl::egblas::has_sbatch_k_minus_scale2 && all_row_major<A, B> && all_floating<A, B>)
            || (D4  && impl::egblas::has_dbatch_k_minus_scale4 && all_row_major<A, B> && all_floating<A, B>);

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    batch_k_minus_scale_expr(A a, B b, C c) : base_type(a, b, c) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <same_dimensions<B> L>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const B& b, [[maybe_unused]] const C& c, [[maybe_unused]] L& lhs) {
        if constexpr (D4) {
            if constexpr (all_fast<A, B, C, L>) {
                static_assert(etl::dim<0, B>() == etl::dim<0, L>(), "Invalid dimensions for batch_k_minus_scale");
                static_assert(etl::dim<1, B>() == etl::dim<1, L>(), "Invalid dimensions for batch_k_minus_scale");
                static_assert(etl::dim<2, B>() == etl::dim<2, L>(), "Invalid dimensions for batch_k_minus_scale");
                static_assert(etl::dim<3, B>() == etl::dim<3, L>(), "Invalid dimensions for batch_k_minus_scale");

                static_assert(etl::dim<0, A>() == etl::dim<1, B>(), "Invalid dimensions for batch_k_minus_scale");
                static_assert(etl::dim<0, A>() == etl::dim<0, C>(), "Invalid dimensions for batch_k_minus_scale");
            } else {
                cpp_assert(etl::dim<0>(b) == etl::dim<0>(lhs), "Invalid dimensions for batch_k_minus_scale");
                cpp_assert(etl::dim<1>(b) == etl::dim<1>(lhs), "Invalid dimensions for batch_k_minus_scale");
                cpp_assert(etl::dim<2>(b) == etl::dim<2>(lhs), "Invalid dimensions for batch_k_minus_scale");
                cpp_assert(etl::dim<3>(b) == etl::dim<3>(lhs), "Invalid dimensions for batch_k_minus_scale");

                cpp_assert(etl::dim<0>(a) == etl::dim<1>(b), "Invalid dimensions for batch_k_minus_scale");
                cpp_assert(etl::dim<0>(a) == etl::dim<0>(c), "Invalid dimensions for batch_k_minus_scale");
            }
        } else {
            if constexpr (all_fast<A, B, C, L>) {
                static_assert(etl::dim<0, B>() == etl::dim<0, L>(), "Invalid dimensions for batch_k_minus_scale");
                static_assert(etl::dim<1, B>() == etl::dim<1, L>(), "Invalid dimensions for batch_k_minus_scale");

                static_assert(etl::dim<0, A>() == etl::dim<1, B>(), "Invalid dimensions for batch_k_minus_scale");
                static_assert(etl::dim<0, A>() == etl::dim<0, C>(), "Invalid dimensions for batch_k_minus_scale");
            } else {
                cpp_assert(etl::dim<0>(b) == etl::dim<0>(lhs), "Invalid dimensions for batch_k_minus_scale");
                cpp_assert(etl::dim<1>(b) == etl::dim<1>(lhs), "Invalid dimensions for batch_k_minus_scale");

                cpp_assert(etl::dim<0>(a) == etl::dim<1>(b), "Invalid dimensions for batch_k_minus_scale");
                cpp_assert(etl::dim<0>(a) == etl::dim<0>(c), "Invalid dimensions for batch_k_minus_scale");
            }
        }
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_to(L&& lhs) const {
        inc_counter("temp:assign");

        auto& a = this->a();
        auto& b = this->b();
        auto& c = this->c();

        check(a, b, c, lhs);

        if constexpr (D4) {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);
            const auto M     = etl::dim<2>(lhs);
            const auto N     = etl::dim<3>(lhs);

            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                decltype(auto) t1 = smart_forward_gpu(a);
                decltype(auto) t2 = smart_forward_gpu(b);
                decltype(auto) t3 = smart_forward_gpu(c);

                t1.ensure_gpu_up_to_date();
                t2.ensure_gpu_up_to_date();
                t3.ensure_gpu_up_to_date();

                lhs.ensure_gpu_allocated();

                impl::egblas::batch_k_minus_scale(Batch, K, M, N, t2.gpu_memory(), t1.gpu_memory(), t3.gpu_memory(), lhs.gpu_memory());

                lhs.validate_gpu();
                lhs.invalidate_cpu();
            } else {
                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, L> && all_row_major<A, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            const auto MN = M * N;

                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    T ak = a(k);
                                    T ck = c(k);

                                    auto lhs_sub = lhs(batch)(k);
                                    auto b_sub   = b(batch)(k);

                                    size_t mn = 0;

                                    auto a1 = vec_type::set(ak);
                                    auto c1 = vec_type::set(ck);

                                    for (; mn + 4 * vec_size - 1 < MN; mn += 4 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto b3 = b_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto b4 = b_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto r1 = vec_type::mul(a1, vec_type::sub(b1, c1));
                                        auto r2 = vec_type::mul(a1, vec_type::sub(b2, c1));
                                        auto r3 = vec_type::mul(a1, vec_type::sub(b3, c1));
                                        auto r4 = vec_type::mul(a1, vec_type::sub(b4, c1));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r3, mn + 2 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r4, mn + 3 * vec_size);
                                    }

                                    for (; mn + 2 * vec_size - 1 < MN; mn += 2 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto r1 = vec_type::mul(a1, vec_type::sub(b1, c1));
                                        auto r2 = vec_type::mul(a1, vec_type::sub(b2, c1));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                    }

                                    for (; mn + vec_size - 1 < MN; mn += vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn);

                                        auto r1 = vec_type::mul(a1, vec_type::sub(b1, c1));

                                        lhs_sub.template storeu<vec_type>(r1, mn);
                                    }

                                    for (; mn + 3 < MN; mn += 4) {
                                        lhs_sub[mn + 0] = ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] = ak * (b_sub[mn + 1] - ck);
                                        lhs_sub[mn + 2] = ak * (b_sub[mn + 2] - ck);
                                        lhs_sub[mn + 3] = ak * (b_sub[mn + 3] - ck);
                                    }

                                    for (; mn + 1 < MN; mn += 2) {
                                        lhs_sub[mn + 0] = ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] = ak * (b_sub[mn + 1] - ck);
                                    }

                                    for (; mn < MN; ++mn) {
                                        lhs_sub[mn] = ak * (b_sub[mn] - ck);
                                    }
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    for (size_t m = 0; m < M; ++m) {
                                        for (size_t n = 0; n < N; ++n) {
                                            lhs(batch, k, m, n) = a(k) * (b(batch, k, m, n) - c(k));
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        } else {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);

            if constexpr (impl::egblas::has_sbatch_k_minus_scale2 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                decltype(auto) t1 = smart_forward_gpu(a);
                decltype(auto) t2 = smart_forward_gpu(b);
                decltype(auto) t3 = smart_forward_gpu(c);

                t1.ensure_gpu_up_to_date();
                t2.ensure_gpu_up_to_date();
                t3.ensure_gpu_up_to_date();

                lhs.ensure_gpu_allocated();

                impl::egblas::batch_k_minus_scale(Batch, K, t2.gpu_memory(), t1.gpu_memory(), t3.gpu_memory(), lhs.gpu_memory());

                lhs.validate_gpu();
                lhs.invalidate_cpu();
            } else {
                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            for (size_t batch = first; batch < last; ++batch) {
                                size_t k = 0;

                                const size_t base = batch * K;

                                for (; k + 4 * vec_size - 1 < K; k += 4 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);
                                    auto a3 = a.template load<vec_type>(k + 2 * vec_size);
                                    auto a4 = a.template load<vec_type>(k + 3 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto b3 = b.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto b4 = b.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);
                                    auto c3 = c.template loadu<vec_type>(k + 2 * vec_size);
                                    auto c4 = c.template loadu<vec_type>(k + 3 * vec_size);

                                    auto r1 = vec_type::mul(a1, vec_type::sub(b1, c1));
                                    auto r2 = vec_type::mul(a2, vec_type::sub(b2, c2));
                                    auto r3 = vec_type::mul(a3, vec_type::sub(b3, c3));
                                    auto r4 = vec_type::mul(a4, vec_type::sub(b4, c4));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                    lhs.template storeu<vec_type>(r3, base + k + 2 * vec_size);
                                    lhs.template storeu<vec_type>(r4, base + k + 3 * vec_size);
                                }

                                for (; k + 2 * vec_size - 1 < K; k += 2 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);

                                    auto r1 = vec_type::mul(a1, vec_type::sub(b1, c1));
                                    auto r2 = vec_type::mul(a2, vec_type::sub(b2, c2));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                }

                                for (; k + vec_size - 1 < K; k += vec_size) {
                                    auto a1 = a.template load<vec_type>(k);

                                    auto b1 = b.template loadu<vec_type>(base + k);

                                    auto c1 = c.template loadu<vec_type>(k);

                                    auto r1 = vec_type::mul(a1, vec_type::sub(b1, c1));

                                    lhs.template storeu<vec_type>(r1, base + k);
                                }

                                for (; k + 3 < K; k += 4) {
                                    lhs(batch, k + 0) = a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) = a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                    lhs(batch, k + 2) = a(k + 2) * (b(batch, k + 2) - c(k + 2));
                                    lhs(batch, k + 3) = a(k + 3) * (b(batch, k + 3) - c(k + 3));
                                }

                                for (; k + 1 < K; k += 2) {
                                    lhs(batch, k + 0) = a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) = a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                }

                                if (k < K) {
                                    lhs(batch, k) = a(k) * (b(batch, k) - c(k));
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    lhs(batch, k) = a(k) * (b(batch, k) - c(k));
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        }
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_add_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();
        auto& c = this->c();

        check(a, b, c, lhs);

        if constexpr (D4) {
            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                std_add_evaluate(*this, lhs);
            } else {
                const auto Batch = etl::dim<0>(lhs);
                const auto K     = etl::dim<1>(lhs);
                const auto M     = etl::dim<2>(lhs);
                const auto N     = etl::dim<3>(lhs);

                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();
                lhs.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            const auto MN = M * N;

                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    T ak = a(k);
                                    T ck = c(k);

                                    auto lhs_sub = lhs(batch)(k);
                                    auto b_sub   = b(batch)(k);

                                    size_t mn = 0;

                                    auto a1 = vec_type::set(ak);
                                    auto c1 = vec_type::set(ck);

                                    for (; mn + 4 * vec_size - 1 < MN; mn += 4 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto b3 = b_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto b4 = b_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto l2 = lhs_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto l3 = lhs_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto l4 = lhs_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto r1 = vec_type::add(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                        auto r2 = vec_type::add(l2, vec_type::mul(a1, vec_type::sub(b2, c1)));
                                        auto r3 = vec_type::add(l3, vec_type::mul(a1, vec_type::sub(b3, c1)));
                                        auto r4 = vec_type::add(l4, vec_type::mul(a1, vec_type::sub(b4, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r3, mn + 2 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r4, mn + 3 * vec_size);
                                    }

                                    for (; mn + 2 * vec_size - 1 < MN; mn += 2 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto l2 = lhs_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto r1 = vec_type::add(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                        auto r2 = vec_type::add(l2, vec_type::mul(a1, vec_type::sub(b2, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                    }

                                    for (; mn + vec_size - 1 < MN; mn += vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn);

                                        auto r1 = vec_type::add(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn);
                                    }

                                    for (; mn + 3 < MN; mn += 4) {
                                        lhs_sub[mn + 0] += ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] += ak * (b_sub[mn + 1] - ck);
                                        lhs_sub[mn + 2] += ak * (b_sub[mn + 2] - ck);
                                        lhs_sub[mn + 3] += ak * (b_sub[mn + 3] - ck);
                                    }

                                    for (; mn + 1 < MN; mn += 2) {
                                        lhs_sub[mn + 0] += ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] += ak * (b_sub[mn + 1] - ck);
                                    }

                                    for (; mn < MN; ++mn) {
                                        lhs_sub[mn] += ak * (b_sub[mn] - ck);
                                    }
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    for (size_t m = 0; m < M; ++m) {
                                        for (size_t n = 0; n < N; ++n) {
                                            lhs(batch, k, m, n) += a(k) * (b(batch, k, m, n) - c(k));
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        } else {
            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                std_add_evaluate(*this, lhs);
            } else {
                const auto Batch = etl::dim<0>(lhs);
                const auto K     = etl::dim<1>(lhs);

                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();
                lhs.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            for (size_t batch = first; batch < last; ++batch) {
                                size_t k = 0;

                                const size_t base = batch * K;

                                for (; k + 4 * vec_size - 1 < K; k += 4 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);
                                    auto a3 = a.template load<vec_type>(k + 2 * vec_size);
                                    auto a4 = a.template load<vec_type>(k + 3 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto b3 = b.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto b4 = b.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);
                                    auto c3 = c.template loadu<vec_type>(k + 2 * vec_size);
                                    auto c4 = c.template loadu<vec_type>(k + 3 * vec_size);

                                    auto l1 = lhs.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto l2 = lhs.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto l3 = lhs.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto l4 = lhs.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto r1 = vec_type::add(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                    auto r2 = vec_type::add(l2, vec_type::mul(a2, vec_type::sub(b2, c2)));
                                    auto r3 = vec_type::add(l3, vec_type::mul(a3, vec_type::sub(b3, c3)));
                                    auto r4 = vec_type::add(l4, vec_type::mul(a4, vec_type::sub(b4, c4)));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                    lhs.template storeu<vec_type>(r3, base + k + 2 * vec_size);
                                    lhs.template storeu<vec_type>(r4, base + k + 3 * vec_size);
                                }

                                for (; k + 2 * vec_size - 1 < K; k += 2 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);

                                    auto l1 = lhs.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto l2 = lhs.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto r1 = vec_type::add(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                    auto r2 = vec_type::add(l2, vec_type::mul(a2, vec_type::sub(b2, c2)));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                }

                                for (; k + vec_size - 1 < K; k += vec_size) {
                                    auto a1 = a.template load<vec_type>(k);

                                    auto b1 = b.template loadu<vec_type>(base + k);

                                    auto c1 = c.template loadu<vec_type>(k);

                                    auto l1 = lhs.template loadu<vec_type>(base + k);

                                    auto r1 = vec_type::add(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));

                                    lhs.template storeu<vec_type>(r1, base + k);
                                }

                                for (; k + 3 < K; k += 4) {
                                    lhs(batch, k + 0) += a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) += a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                    lhs(batch, k + 2) += a(k + 2) * (b(batch, k + 2) - c(k + 2));
                                    lhs(batch, k + 3) += a(k + 3) * (b(batch, k + 3) - c(k + 3));
                                }

                                for (; k + 1 < K; k += 2) {
                                    lhs(batch, k + 0) += a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) += a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                }

                                if (k < K) {
                                    lhs(batch, k) += a(k) * (b(batch, k) - c(k));
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    lhs(batch, k) += a(k) * (b(batch, k) - c(k));
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        }
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_sub_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();
        auto& c = this->c();

        check(a, b, c, lhs);

        if constexpr (D4) {
            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                std_sub_evaluate(*this, lhs);
            } else {
                const auto Batch = etl::dim<0>(lhs);
                const auto K     = etl::dim<1>(lhs);
                const auto M     = etl::dim<2>(lhs);
                const auto N     = etl::dim<3>(lhs);

                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();
                lhs.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            const auto MN = M * N;

                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    T ak = a(k);
                                    T ck = c(k);

                                    auto lhs_sub = lhs(batch)(k);
                                    auto b_sub   = b(batch)(k);

                                    size_t mn = 0;

                                    auto a1 = vec_type::set(ak);
                                    auto c1 = vec_type::set(ck);

                                    for (; mn + 4 * vec_size - 1 < MN; mn += 4 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto b3 = b_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto b4 = b_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto l2 = lhs_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto l3 = lhs_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto l4 = lhs_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto r1 = vec_type::sub(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                        auto r2 = vec_type::sub(l2, vec_type::mul(a1, vec_type::sub(b2, c1)));
                                        auto r3 = vec_type::sub(l3, vec_type::mul(a1, vec_type::sub(b3, c1)));
                                        auto r4 = vec_type::sub(l4, vec_type::mul(a1, vec_type::sub(b4, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r3, mn + 2 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r4, mn + 3 * vec_size);
                                    }

                                    for (; mn + 2 * vec_size - 1 < MN; mn += 2 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto l2 = lhs_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto r1 = vec_type::sub(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                        auto r2 = vec_type::sub(l2, vec_type::mul(a1, vec_type::sub(b2, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                    }

                                    for (; mn + vec_size - 1 < MN; mn += vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn);

                                        auto r1 = vec_type::sub(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn);
                                    }

                                    for (; mn + 3 < MN; mn += 4) {
                                        lhs_sub[mn + 0] -= ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] -= ak * (b_sub[mn + 1] - ck);
                                        lhs_sub[mn + 2] -= ak * (b_sub[mn + 2] - ck);
                                        lhs_sub[mn + 3] -= ak * (b_sub[mn + 3] - ck);
                                    }

                                    for (; mn + 1 < MN; mn += 2) {
                                        lhs_sub[mn + 0] -= ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] -= ak * (b_sub[mn + 1] - ck);
                                    }

                                    for (; mn < MN; ++mn) {
                                        lhs_sub[mn] -= ak * (b_sub[mn] - ck);
                                    }
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    for (size_t m = 0; m < M; ++m) {
                                        for (size_t n = 0; n < N; ++n) {
                                            lhs(batch, k, m, n) -= a(k) * (b(batch, k, m, n) - c(k));
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        } else {
            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                std_sub_evaluate(*this, lhs);
            } else {
                const auto Batch = etl::dim<0>(lhs);
                const auto K     = etl::dim<1>(lhs);

                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();
                lhs.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            for (size_t batch = first; batch < last; ++batch) {
                                size_t k = 0;

                                const size_t base = batch * K;

                                for (; k + 4 * vec_size - 1 < K; k += 4 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);
                                    auto a3 = a.template load<vec_type>(k + 2 * vec_size);
                                    auto a4 = a.template load<vec_type>(k + 3 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto b3 = b.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto b4 = b.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);
                                    auto c3 = c.template loadu<vec_type>(k + 2 * vec_size);
                                    auto c4 = c.template loadu<vec_type>(k + 3 * vec_size);

                                    auto l1 = lhs.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto l2 = lhs.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto l3 = lhs.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto l4 = lhs.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto r1 = vec_type::sub(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                    auto r2 = vec_type::sub(l2, vec_type::mul(a2, vec_type::sub(b2, c2)));
                                    auto r3 = vec_type::sub(l3, vec_type::mul(a3, vec_type::sub(b3, c3)));
                                    auto r4 = vec_type::sub(l4, vec_type::mul(a4, vec_type::sub(b4, c4)));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                    lhs.template storeu<vec_type>(r3, base + k + 2 * vec_size);
                                    lhs.template storeu<vec_type>(r4, base + k + 3 * vec_size);
                                }

                                for (; k + 2 * vec_size - 1 < K; k += 2 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);

                                    auto l1 = lhs.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto l2 = lhs.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto r1 = vec_type::sub(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                    auto r2 = vec_type::sub(l2, vec_type::mul(a2, vec_type::sub(b2, c2)));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                }

                                for (; k + vec_size - 1 < K; k += vec_size) {
                                    auto a1 = a.template load<vec_type>(k);

                                    auto b1 = b.template loadu<vec_type>(base + k);

                                    auto c1 = c.template loadu<vec_type>(k);

                                    auto l1 = lhs.template loadu<vec_type>(base + k);

                                    auto r1 = vec_type::sub(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));

                                    lhs.template storeu<vec_type>(r1, base + k);
                                }

                                for (; k + 3 < K; k += 4) {
                                    lhs(batch, k + 0) -= a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) -= a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                    lhs(batch, k + 2) -= a(k + 2) * (b(batch, k + 2) - c(k + 2));
                                    lhs(batch, k + 3) -= a(k + 3) * (b(batch, k + 3) - c(k + 3));
                                }

                                for (; k + 1 < K; k += 2) {
                                    lhs(batch, k + 0) -= a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) -= a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                }

                                if (k < K) {
                                    lhs(batch, k) -= a(k) * (b(batch, k) - c(k));
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    lhs(batch, k) -= a(k) * (b(batch, k) - c(k));
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        }
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mul_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();
        auto& c = this->c();

        check(a, b, c, lhs);

        if constexpr (D4) {
            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                std_mul_evaluate(*this, lhs);
            } else {
                const auto Batch = etl::dim<0>(lhs);
                const auto K     = etl::dim<1>(lhs);
                const auto M     = etl::dim<2>(lhs);
                const auto N     = etl::dim<3>(lhs);

                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();
                lhs.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            const auto MN = M * N;

                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    T ak = a(k);
                                    T ck = c(k);

                                    auto lhs_sub = lhs(batch)(k);
                                    auto b_sub   = b(batch)(k);

                                    size_t mn = 0;

                                    auto a1 = vec_type::set(ak);
                                    auto c1 = vec_type::set(ck);

                                    for (; mn + 4 * vec_size - 1 < MN; mn += 4 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto b3 = b_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto b4 = b_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto l2 = lhs_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto l3 = lhs_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto l4 = lhs_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto r1 = vec_type::mul(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                        auto r2 = vec_type::mul(l2, vec_type::mul(a1, vec_type::sub(b2, c1)));
                                        auto r3 = vec_type::mul(l3, vec_type::mul(a1, vec_type::sub(b3, c1)));
                                        auto r4 = vec_type::mul(l4, vec_type::mul(a1, vec_type::sub(b4, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r3, mn + 2 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r4, mn + 3 * vec_size);
                                    }

                                    for (; mn + 2 * vec_size - 1 < MN; mn += 2 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto l2 = lhs_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto r1 = vec_type::mul(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                        auto r2 = vec_type::mul(l2, vec_type::mul(a1, vec_type::sub(b2, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                    }

                                    for (; mn + vec_size - 1 < MN; mn += vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn);

                                        auto r1 = vec_type::mul(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn);
                                    }

                                    for (; mn + 3 < MN; mn += 4) {
                                        lhs_sub[mn + 0] *= ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] *= ak * (b_sub[mn + 1] - ck);
                                        lhs_sub[mn + 2] *= ak * (b_sub[mn + 2] - ck);
                                        lhs_sub[mn + 3] *= ak * (b_sub[mn + 3] - ck);
                                    }

                                    for (; mn + 1 < MN; mn += 2) {
                                        lhs_sub[mn + 0] *= ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] *= ak * (b_sub[mn + 1] - ck);
                                    }

                                    for (; mn < MN; ++mn) {
                                        lhs_sub[mn] *= ak * (b_sub[mn] - ck);
                                    }
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    for (size_t m = 0; m < M; ++m) {
                                        for (size_t n = 0; n < N; ++n) {
                                            lhs(batch, k, m, n) *= a(k) * (b(batch, k, m, n) - c(k));
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        } else {
            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                std_mul_evaluate(*this, lhs);
            } else {
                const auto Batch = etl::dim<0>(lhs);
                const auto K     = etl::dim<1>(lhs);

                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();
                lhs.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            for (size_t batch = first; batch < last; ++batch) {
                                size_t k = 0;

                                const size_t base = batch * K;

                                for (; k + 4 * vec_size - 1 < K; k += 4 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);
                                    auto a3 = a.template load<vec_type>(k + 2 * vec_size);
                                    auto a4 = a.template load<vec_type>(k + 3 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto b3 = b.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto b4 = b.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);
                                    auto c3 = c.template loadu<vec_type>(k + 2 * vec_size);
                                    auto c4 = c.template loadu<vec_type>(k + 3 * vec_size);

                                    auto l1 = lhs.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto l2 = lhs.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto l3 = lhs.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto l4 = lhs.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto r1 = vec_type::mul(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                    auto r2 = vec_type::mul(l2, vec_type::mul(a2, vec_type::sub(b2, c2)));
                                    auto r3 = vec_type::mul(l3, vec_type::mul(a3, vec_type::sub(b3, c3)));
                                    auto r4 = vec_type::mul(l4, vec_type::mul(a4, vec_type::sub(b4, c4)));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                    lhs.template storeu<vec_type>(r3, base + k + 2 * vec_size);
                                    lhs.template storeu<vec_type>(r4, base + k + 3 * vec_size);
                                }

                                for (; k + 2 * vec_size - 1 < K; k += 2 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);

                                    auto l1 = lhs.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto l2 = lhs.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto r1 = vec_type::mul(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                    auto r2 = vec_type::mul(l2, vec_type::mul(a2, vec_type::sub(b2, c2)));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                }

                                for (; k + vec_size - 1 < K; k += vec_size) {
                                    auto a1 = a.template load<vec_type>(k);

                                    auto b1 = b.template loadu<vec_type>(base + k);

                                    auto c1 = c.template loadu<vec_type>(k);

                                    auto l1 = lhs.template loadu<vec_type>(base + k);

                                    auto r1 = vec_type::mul(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));

                                    lhs.template storeu<vec_type>(r1, base + k);
                                }

                                for (; k + 3 < K; k += 4) {
                                    lhs(batch, k + 0) *= a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) *= a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                    lhs(batch, k + 2) *= a(k + 2) * (b(batch, k + 2) - c(k + 2));
                                    lhs(batch, k + 3) *= a(k + 3) * (b(batch, k + 3) - c(k + 3));
                                }

                                for (; k + 1 < K; k += 2) {
                                    lhs(batch, k + 0) *= a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) *= a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                }

                                if (k < K) {
                                    lhs(batch, k) *= a(k) * (b(batch, k) - c(k));
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    lhs(batch, k) *= a(k) * (b(batch, k) - c(k));
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        }
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_div_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();
        auto& c = this->c();

        check(a, b, c, lhs);

        if constexpr (D4) {
            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                std_div_evaluate(*this, lhs);
            } else {
                const auto Batch = etl::dim<0>(lhs);
                const auto K     = etl::dim<1>(lhs);
                const auto M     = etl::dim<2>(lhs);
                const auto N     = etl::dim<3>(lhs);

                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();
                lhs.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            const auto MN = M * N;

                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    T ak = a(k);
                                    T ck = c(k);

                                    auto lhs_sub = lhs(batch)(k);
                                    auto b_sub   = b(batch)(k);

                                    size_t mn = 0;

                                    auto a1 = vec_type::set(ak);
                                    auto c1 = vec_type::set(ck);

                                    for (; mn + 4 * vec_size - 1 < MN; mn += 4 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto b3 = b_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto b4 = b_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto l2 = lhs_sub.template loadu<vec_type>(mn + 1 * vec_size);
                                        auto l3 = lhs_sub.template loadu<vec_type>(mn + 2 * vec_size);
                                        auto l4 = lhs_sub.template loadu<vec_type>(mn + 3 * vec_size);

                                        auto r1 = vec_type::div(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                        auto r2 = vec_type::div(l2, vec_type::mul(a1, vec_type::sub(b2, c1)));
                                        auto r3 = vec_type::div(l3, vec_type::mul(a1, vec_type::sub(b3, c1)));
                                        auto r4 = vec_type::div(l4, vec_type::mul(a1, vec_type::sub(b4, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r3, mn + 2 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r4, mn + 3 * vec_size);
                                    }

                                    for (; mn + 2 * vec_size - 1 < MN; mn += 2 * vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto b2 = b_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn + 0 * vec_size);
                                        auto l2 = lhs_sub.template loadu<vec_type>(mn + 1 * vec_size);

                                        auto r1 = vec_type::div(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                        auto r2 = vec_type::div(l2, vec_type::mul(a1, vec_type::sub(b2, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn + 0 * vec_size);
                                        lhs_sub.template storeu<vec_type>(r2, mn + 1 * vec_size);
                                    }

                                    for (; mn + vec_size - 1 < MN; mn += vec_size) {
                                        auto b1 = b_sub.template loadu<vec_type>(mn);

                                        auto l1 = lhs_sub.template loadu<vec_type>(mn);

                                        auto r1 = vec_type::div(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));

                                        lhs_sub.template storeu<vec_type>(r1, mn);
                                    }

                                    for (; mn + 3 < MN; mn += 4) {
                                        lhs_sub[mn + 0] /= ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] /= ak * (b_sub[mn + 1] - ck);
                                        lhs_sub[mn + 2] /= ak * (b_sub[mn + 2] - ck);
                                        lhs_sub[mn + 3] /= ak * (b_sub[mn + 3] - ck);
                                    }

                                    for (; mn + 1 < MN; mn += 2) {
                                        lhs_sub[mn + 0] /= ak * (b_sub[mn + 0] - ck);
                                        lhs_sub[mn + 1] /= ak * (b_sub[mn + 1] - ck);
                                    }

                                    for (; mn < MN; ++mn) {
                                        lhs_sub[mn] /= ak * (b_sub[mn] - ck);
                                    }
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    for (size_t m = 0; m < M; ++m) {
                                        for (size_t n = 0; n < N; ++n) {
                                            lhs(batch, k, m, n) /= a(k) * (b(batch, k, m, n) - c(k));
                                        }
                                    }
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        } else {
            if constexpr (impl::egblas::has_sbatch_k_minus_scale4 && all_row_major<A, B, L> && all_floating<A, B, L>) {
                std_div_evaluate(*this, lhs);
            } else {
                const auto Batch = etl::dim<0>(lhs);
                const auto K     = etl::dim<1>(lhs);

                standard_evaluator::pre_assign_rhs(a);
                standard_evaluator::pre_assign_rhs(b);

                a.ensure_cpu_up_to_date();
                b.ensure_cpu_up_to_date();
                c.ensure_cpu_up_to_date();
                lhs.ensure_cpu_up_to_date();

                auto batch_fun_b = [&](const size_t first, const size_t last) {
                    CPU_SECTION {
                        if constexpr (vec_enabled && all_vectorizable<vector_mode, A, B, L> && all_row_major<A, B, L>) {
                            using vec_type = default_vec;
                            using T        = value_t<L>;

                            static constexpr size_t vec_size = vec_type::template traits<T>::size;

                            for (size_t batch = first; batch < last; ++batch) {
                                size_t k = 0;

                                const size_t base = batch * K;

                                for (; k + 4 * vec_size - 1 < K; k += 4 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);
                                    auto a3 = a.template load<vec_type>(k + 2 * vec_size);
                                    auto a4 = a.template load<vec_type>(k + 3 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto b3 = b.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto b4 = b.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);
                                    auto c3 = c.template loadu<vec_type>(k + 2 * vec_size);
                                    auto c4 = c.template loadu<vec_type>(k + 3 * vec_size);

                                    auto l1 = lhs.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto l2 = lhs.template loadu<vec_type>(base + k + 1 * vec_size);
                                    auto l3 = lhs.template loadu<vec_type>(base + k + 2 * vec_size);
                                    auto l4 = lhs.template loadu<vec_type>(base + k + 3 * vec_size);

                                    auto r1 = vec_type::div(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                    auto r2 = vec_type::div(l2, vec_type::mul(a2, vec_type::sub(b2, c2)));
                                    auto r3 = vec_type::div(l3, vec_type::mul(a3, vec_type::sub(b3, c3)));
                                    auto r4 = vec_type::div(l4, vec_type::mul(a4, vec_type::sub(b4, c4)));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                    lhs.template storeu<vec_type>(r3, base + k + 2 * vec_size);
                                    lhs.template storeu<vec_type>(r4, base + k + 3 * vec_size);
                                }

                                for (; k + 2 * vec_size - 1 < K; k += 2 * vec_size) {
                                    auto a1 = a.template load<vec_type>(k + 0 * vec_size);
                                    auto a2 = a.template load<vec_type>(k + 1 * vec_size);

                                    auto b1 = b.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto b2 = b.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto c1 = c.template loadu<vec_type>(k + 0 * vec_size);
                                    auto c2 = c.template loadu<vec_type>(k + 1 * vec_size);

                                    auto l1 = lhs.template loadu<vec_type>(base + k + 0 * vec_size);
                                    auto l2 = lhs.template loadu<vec_type>(base + k + 1 * vec_size);

                                    auto r1 = vec_type::div(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));
                                    auto r2 = vec_type::div(l2, vec_type::mul(a2, vec_type::sub(b2, c2)));

                                    lhs.template storeu<vec_type>(r1, base + k + 0 * vec_size);
                                    lhs.template storeu<vec_type>(r2, base + k + 1 * vec_size);
                                }

                                for (; k + vec_size - 1 < K; k += vec_size) {
                                    auto a1 = a.template load<vec_type>(k);

                                    auto b1 = b.template loadu<vec_type>(base + k);

                                    auto c1 = c.template loadu<vec_type>(k);

                                    auto l1 = lhs.template loadu<vec_type>(base + k);

                                    auto r1 = vec_type::div(l1, vec_type::mul(a1, vec_type::sub(b1, c1)));

                                    lhs.template storeu<vec_type>(r1, base + k);
                                }

                                for (; k + 3 < K; k += 4) {
                                    lhs(batch, k + 0) /= a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) /= a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                    lhs(batch, k + 2) /= a(k + 2) * (b(batch, k + 2) - c(k + 2));
                                    lhs(batch, k + 3) /= a(k + 3) * (b(batch, k + 3) - c(k + 3));
                                }

                                for (; k + 1 < K; k += 2) {
                                    lhs(batch, k + 0) /= a(k + 0) * (b(batch, k + 0) - c(k + 0));
                                    lhs(batch, k + 1) /= a(k + 1) * (b(batch, k + 1) - c(k + 1));
                                }

                                if (k < K) {
                                    lhs(batch, k) /= a(k) * (b(batch, k) - c(k));
                                }
                            }
                        } else {
                            for (size_t batch = first; batch < last; ++batch) {
                                for (size_t k = 0; k < K; ++k) {
                                    lhs(batch, k) /= a(k) * (b(batch, k) - c(k));
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

                lhs.validate_cpu();
                lhs.invalidate_gpu();
            }
        }
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mod_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();
        auto& c = this->c();

        check(a, b, c, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
        c.ensure_cpu_up_to_date();
        lhs.ensure_cpu_up_to_date();

        if constexpr (D4) {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);
            const auto M     = etl::dim<2>(lhs);
            const auto N     = etl::dim<3>(lhs);

            auto batch_fun_b = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t batch = first; batch < last; ++batch) {
                        for (size_t k = 0; k < K; ++k) {
                            for (size_t m = 0; m < M; ++m) {
                                for (size_t n = 0; n < N; ++n) {
                                    lhs(batch, k, m, n) %= a(k) * (b(batch, k, m, n) - c(k));
                                }
                            }
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

            lhs.validate_cpu();
            lhs.invalidate_gpu();
        } else {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);

            auto batch_fun_b = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t batch = first; batch < last; ++batch) {
                        for (size_t k = 0; k < K; ++k) {
                            lhs(batch, k) %= a(k) * (b(batch, k) - c(k));
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);

            lhs.validate_cpu();
            lhs.invalidate_gpu();
        }
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const batch_k_minus_scale_expr& expr) {
        return os << "batch_k_minus_scale(" << expr._a << "," << expr._b << "," << expr._c << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B, typename C>
struct etl_traits<etl::batch_k_minus_scale_expr<A, B, C>> {
    using expr_t     = etl::batch_k_minus_scale_expr<A, B, C>; ///< The expression type
    using sub_expr_t = std::decay_t<B>;                       ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                ///< The sub traits
    using value_type = value_t<A>;                            ///< The value type of the expression

    static constexpr bool  is_etl         = true;                      ///< Indicates if the type is an ETL expression
    static constexpr bool  is_transformer = false;                     ///< Indicates if the type is a transformer
    static constexpr bool  is_view        = false;                     ///< Indicates if the type is a view
    static constexpr bool  is_magic_view  = false;                     ///< Indicates if the type is a magic view
    static constexpr bool  is_fast        = sub_traits::is_fast;       ///< Indicates if the expression is fast
    static constexpr bool  is_linear      = false;                     ///< Indicates if the expression is linear
    static constexpr bool  is_thread_safe = true;                      ///< Indicates if the expression is thread safe
    static constexpr bool  is_value       = false;                     ///< Indicates if the expression is of value type
    static constexpr bool  is_direct      = true;                      ///< Indicates if the expression has direct memory access
    static constexpr bool  is_generator   = false;                     ///< Indicates if the expression is a generator
    static constexpr bool  is_padded      = false;                     ///< Indicates if the expression is padded
    static constexpr bool  is_aligned     = true;                      ///< Indicates if the expression is padded
    static constexpr bool  is_temporary   = true;                      ///< Indicates if the expression needs a evaluator visitor
    static constexpr bool  gpu_computable = true;                      ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order  = sub_traits::storage_order; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = true;

    /*!
     * \brief Returns the DDth dimension of the expression
     * \return the DDth dimension of the expression
     */
    template <size_t DD>
    static constexpr size_t dim() {
        return decay_traits<B>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        return etl::dim(e._b, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::size(e._b);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return decay_traits<B>::size();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return decay_traits<B>::dimensions();
    }

    /*!
     * \brief Estimate the complexity of computation
     * \return An estimation of the complexity of the expression
     */
    static constexpr int complexity() noexcept {
        return -1;
    }
};

// Note: This function should not be called directly
// instead, batch_hint(a >> b) should be used
// But this function is used as helpers from batch_hint

/*!
 * \brief Returns the transpose of the given expression.
 * \param value The expression
 * \return The transpose of the given expression.
 */
template <etl_1d A, etl_2d_or_4d B, etl_1d C>
batch_k_minus_scale_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>> batch_k_minus_scale(const A& a, const B& b, const C& c) {
    return {a, b, c};
}

} //end of namespace etl
