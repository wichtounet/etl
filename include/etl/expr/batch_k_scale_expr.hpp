//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, typename B>
struct batch_k_scale_expr : base_temporary_expr_bin<batch_k_scale_expr<A, B>, A, B> {
    using value_type  = value_t<A>;                               ///< The type of value of the expression
    using this_type   = batch_k_scale_expr<A, B>;                 ///< The type of this expression
    using base_type   = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using left_traits = decay_traits<A>;                          ///< The traits of the sub type

    static constexpr bool D4 = is_4d<B>; ///< If the expression is 4D (instead of 2D)

    static constexpr auto storage_order = left_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    batch_k_scale_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const B& b, [[maybe_unused]] const C& c) {
        if constexpr (D4) {
            static_assert(etl::dimensions<C>() == 4, "The output of batch_k_scale is a 4D matrix");
            static_assert(etl::dimensions<A>() == 1, "The lhs of batch_k_scale is a 1D matrix");
            static_assert(etl::dimensions<B>() == 4, "The rhs of batch_k_scale is a 4D matrix");

            if constexpr (all_fast<A, C>) {
                static_assert(etl::dim<0, B>() == etl::dim<0, C>(), "Invalid dimensions for batch_k_scale");
                static_assert(etl::dim<1, B>() == etl::dim<1, C>(), "Invalid dimensions for batch_k_scale");
                static_assert(etl::dim<2, B>() == etl::dim<2, C>(), "Invalid dimensions for batch_k_scale");
                static_assert(etl::dim<3, B>() == etl::dim<3, C>(), "Invalid dimensions for batch_k_scale");

                static_assert(etl::dim<0, A>() == etl::dim<1, B>(), "Invalid dimensions for batch_k_scale");
            } else {
                cpp_assert(etl::dim<0>(b) == etl::dim<0>(c), "Invalid dimensions for batch_k_scale");
                cpp_assert(etl::dim<1>(b) == etl::dim<1>(c), "Invalid dimensions for batch_k_scale");
                cpp_assert(etl::dim<2>(b) == etl::dim<2>(c), "Invalid dimensions for batch_k_scale");
                cpp_assert(etl::dim<3>(b) == etl::dim<3>(c), "Invalid dimensions for batch_k_scale");

                cpp_assert(etl::dim<0>(a) == etl::dim<1>(b), "Invalid dimensions for batch_k_scale");
            }
        } else {
            static_assert(etl::dimensions<C>() == 2, "The output of batch_k_scale is a 2D matrix");
            static_assert(etl::dimensions<A>() == 1, "The lhs of batch_k_scale is a 1D matrix");
            static_assert(etl::dimensions<B>() == 2, "The rhs of batch_k_scale is a 2D matrix");

            if constexpr (all_fast<A, C>) {
                static_assert(etl::dim<0, B>() == etl::dim<0, C>(), "Invalid dimensions for batch_k_scale");
                static_assert(etl::dim<1, B>() == etl::dim<1, C>(), "Invalid dimensions for batch_k_scale");

                static_assert(etl::dim<0, A>() == etl::dim<1, B>(), "Invalid dimensions for batch_k_scale");
            } else {
                cpp_assert(etl::dim<0>(b) == etl::dim<0>(c), "Invalid dimensions for batch_k_scale");
                cpp_assert(etl::dim<1>(b) == etl::dim<1>(c), "Invalid dimensions for batch_k_scale");

                cpp_assert(etl::dim<0>(a) == etl::dim<1>(b), "Invalid dimensions for batch_k_scale");
            }
        }
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        static_assert(all_etl_expr<A, B, L>, "batch_k_scale only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();

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
                                    lhs(batch, k, m, n) = a(k) * b(batch, k, m, n);
                                }
                            }
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        } else {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);

            auto batch_fun_b = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t batch = first; batch < last; ++batch) {
                        for (size_t k = 0; k < K; ++k) {
                            lhs(batch, k) = a(k) * b(batch, k);
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        }

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        static_assert(all_etl_expr<A, B, L>, "batch_k_scale only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
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
                                    lhs(batch, k, m, n) += a(k) * b(batch, k, m, n);
                                }
                            }
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        } else {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);

            auto batch_fun_b = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t batch = first; batch < last; ++batch) {
                        for (size_t k = 0; k < K; ++k) {
                            lhs(batch, k) += a(k) * b(batch, k);
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        }

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        static_assert(all_etl_expr<A, B, L>, "batch_k_scale only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
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
                                    lhs(batch, k, m, n) -= a(k) * b(batch, k, m, n);
                                }
                            }
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        } else {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);

            auto batch_fun_b = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t batch = first; batch < last; ++batch) {
                        for (size_t k = 0; k < K; ++k) {
                            lhs(batch, k) -= a(k) * b(batch, k);
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        }

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        static_assert(all_etl_expr<A, B, L>, "batch_k_scale only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
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
                                    lhs(batch, k, m, n) *= a(k) * b(batch, k, m, n);
                                }
                            }
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        } else {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);

            auto batch_fun_b = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t batch = first; batch < last; ++batch) {
                        for (size_t k = 0; k < K; ++k) {
                            lhs(batch, k) *= a(k) * b(batch, k);
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        }

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        static_assert(all_etl_expr<A, B, L>, "batch_k_scale only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
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
                                    lhs(batch, k, m, n) /= a(k) * b(batch, k, m, n);
                                }
                            }
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        } else {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);

            auto batch_fun_b = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t batch = first; batch < last; ++batch) {
                        for (size_t k = 0; k < K; ++k) {
                            lhs(batch, k) /= a(k) * b(batch, k);
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        }

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        static_assert(all_etl_expr<A, B, L>, "batch_k_scale only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
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
                                    lhs(batch, k, m, n) %= a(k) * b(batch, k, m, n);
                                }
                            }
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        } else {
            const auto Batch = etl::dim<0>(lhs);
            const auto K     = etl::dim<1>(lhs);

            auto batch_fun_b = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t batch = first; batch < last; ++batch) {
                        for (size_t k = 0; k < K; ++k) {
                            lhs(batch, k) %= a(k) * b(batch, k);
                        }
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_b, 0, Batch, 2UL);
        }

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const batch_k_scale_expr& expr) {
        return os << "batch_k_scale(" << expr._a << "," << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B>
struct etl_traits<etl::batch_k_scale_expr<A, B>> {
    using expr_t     = etl::batch_k_scale_expr<A, B>; ///< The expression type
    using sub_expr_t = std::decay_t<B>;               ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;        ///< The sub traits
    using value_type = value_t<A>;                    ///< The value type of the expression

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
    static constexpr bool  gpu_computable = false;                     ///< Indicates if the expression can be computed on GPU
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
};

// Note: This function should not be called directly
// instead, batch_hint(a >> b) should be used
// But this function is used as helpers from batch_hint

/*!
 * \brief Returns the transpose of the given expression.
 * \param value The expression
 * \return The transpose of the given expression.
 */
template <typename A, typename B>
batch_k_scale_expr<detail::build_type<A>, detail::build_type<B>> batch_k_scale(const A& a, const B& b) {
    static_assert(all_etl_expr<A, B>, "etl::batch_k_scale can only be used on ETL expressions");
    static_assert(is_1d<A>, "etl::batch_k_scale is only defined for 1D LHS");
    static_assert(is_2d<B> || is_4d<B>, "etl::batch_k_scale is only defined for 2D/4D RHS");

    return {a, b};
}

} //end of namespace etl
