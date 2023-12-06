//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

#include "etl/impl/egblas/bias_batch_sum.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <etl_expr A, etl_expr B>
struct bias_batch_var_4d_expr : base_temporary_expr_bin<bias_batch_var_4d_expr<A, B>, A, B> {
    using value_type = value_t<A>;                               ///< The type of value of the expression
    using this_type  = bias_batch_var_4d_expr<A, B>;             ///< The type of this expression
    using base_type  = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using sub_traits = decay_traits<A>;                          ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable =
               (impl::egblas::has_sbias_batch_var4 && all_row_major<A> && all_single_precision<A>)
            || (impl::egblas::has_dbias_batch_var4 && all_row_major<A> && all_double_precision<A>);

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit bias_batch_var_4d_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <etl_expr C>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const B& b, [[maybe_unused]] const C& c) {
        static_assert(etl::dimensions<C>() == 1, "The output of bias_batch_var_4d is a vector");
        static_assert(etl::dimensions<A>() == 4, "The input of bias_batch_var_4d is a 2d matrix");
        static_assert(etl::dimensions<B>() == 1, "The input of bias_batch_var_4d is a vector");

        if constexpr (all_fast<A, B, C>) {
            static_assert(etl::dim<1, A>() == etl::dim<0, C>(), "Invalid dimensions for bias_batch_var_4d");
            static_assert(etl::dim<0, B>() == etl::dim<0, C>(), "Invalid dimensions for bias_batch_var_4d");
        } else {
            cpp_assert(etl::dim<1>(a) == etl::dim<0>(c), "Invalid dimensions for bias_batch_var_4d");
            cpp_assert(etl::dim<0>(b) == etl::dim<0>(c), "Invalid dimensions for bias_batch_var_4d");
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

        check(a, b, lhs);

        using T = value_t<A>;

        const auto N = etl::dim<0>(a);
        const auto K = etl::dim<1>(a);
        const auto F = static_cast<T>(etl::size(a) / etl::size(lhs));

        if constexpr (impl::egblas::has_sbias_batch_var4 && all_row_major<A> && all_floating<A, L>) {
            const auto W = etl::dim<2>(a);
            const auto H = etl::dim<3>(a);

            decltype(auto) t1 = smart_forward_gpu(a);
            decltype(auto) t2 = smart_forward_gpu(b);

            t1.ensure_gpu_up_to_date();
            t2.ensure_gpu_up_to_date();

            lhs.ensure_gpu_allocated();

            impl::egblas::bias_batch_var4(N, K, W, H, t1.gpu_memory(), t2.gpu_memory(), lhs.gpu_memory());

            lhs.validate_gpu();
            lhs.invalidate_cpu();
        } else {
            standard_evaluator::pre_assign_rhs(a);
            standard_evaluator::pre_assign_rhs(b);

            a.ensure_cpu_up_to_date();
            b.ensure_cpu_up_to_date();

            // Note: We use etl::sum directly instead of doing the sum manually
            // That way, we will access the already vectorized sum
            // Now, this means that evaluator decisions will be called several 
            // times. This could be an issue that could be looked at in the future

            auto batch_fun_k = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t k = first; k < last; ++k) {
                        lhs(k) = 0;
                    }

                    for (size_t bb = 0; bb < N; ++bb) {
                        for (size_t k = first; k < last; ++k) {
                            lhs(k) += sum((a(bb)(k) - b(k)) >> (a(bb)(k) - b(k)));
                        }
                    }

                    for (size_t k = first; k < last; ++k) {
                        lhs(k) /= F;
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_k, 0, K, 2UL);

            lhs.validate_cpu();
            lhs.invalidate_gpu();
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

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        using T = value_t<A>;

        const auto N = etl::dim<0>(a);
        const auto K = etl::dim<1>(a);
        const auto F = static_cast<T>(etl::size(a) / etl::size(lhs));

        check(a, b, lhs);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T var = 0;

                    for (size_t bb = 0; bb < N; ++bb) {
                        var += sum((a(bb)(k) - b(k)) >> (a(bb)(k) - b(k)));
                    }

                    lhs(k) += var / F;
                }
            }
        };

        engine_dispatch_1d_serial(batch_fun_k, 0, K, 2UL);

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_sub_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        using T = value_t<A>;

        const auto N = etl::dim<0>(a);
        const auto K = etl::dim<1>(a);
        const auto F = static_cast<T>(etl::size(a) / etl::size(lhs));

        check(a, b, lhs);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T var = 0;

                    for (size_t bb = 0; bb < N; ++bb) {
                        var += sum((a(bb)(k) - b(k)) >> (a(bb)(k) - b(k)));
                    }

                    lhs(k) -= var / F;
                }
            }
        };

        engine_dispatch_1d_serial(batch_fun_k, 0, K, 2UL);

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mul_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        using T = value_t<A>;

        const auto N = etl::dim<0>(a);
        const auto K = etl::dim<1>(a);
        const auto F = static_cast<T>(etl::size(a) / etl::size(lhs));

        check(a, b, lhs);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T var = 0;

                    for (size_t bb = 0; bb < N; ++bb) {
                        var += sum((a(bb)(k) - b(k)) >> (a(bb)(k) - b(k)));
                    }

                    lhs(k) *= var / F;
                }
            }
        };

        engine_dispatch_1d_serial(batch_fun_k, 0, K, 2UL);

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_div_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        using T = value_t<A>;

        const auto N = etl::dim<0>(a);
        const auto K = etl::dim<1>(a);
        const auto F = static_cast<T>(etl::size(a) / etl::size(lhs));

        check(a, b, lhs);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T var = 0;

                    for (size_t bb = 0; bb < N; ++bb) {
                        var += sum((a(bb)(k) - b(k)) >> (a(bb)(k) - b(k)));
                    }

                    lhs(k) /= var / F;
                }
            }
        };

        engine_dispatch_1d_serial(batch_fun_k, 0, K, 2UL);

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mod_to(L&& lhs) const {
        auto& a = this->a();
        auto& b = this->b();

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        using T = value_t<A>;

        const auto N = etl::dim<0>(a);
        const auto K = etl::dim<1>(a);
        const auto F = static_cast<T>(etl::size(a) / etl::size(lhs));

        check(a, b, lhs);

        a.ensure_cpu_up_to_date();
        b.ensure_cpu_up_to_date();
        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T var = 0;

                    for (size_t bb = 0; bb < N; ++bb) {
                        var += sum((a(bb)(k) - b(k)) >> (a(bb)(k) - b(k)));
                    }

                    lhs(k) %= var / F;
                }
            }
        };

        engine_dispatch_1d_serial(batch_fun_k, 0, K, 2UL);

        lhs.validate_cpu();
        lhs.invalidate_gpu();
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const bias_batch_var_4d_expr& expr) {
        return os << "bias_batch_var_4d(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B>
struct etl_traits<etl::bias_batch_var_4d_expr<A, B>> {
    using expr_t     = etl::bias_batch_var_4d_expr<A, B>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                   ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;            ///< The sub traits
    using value_type = value_t<A>;                        ///< The value type of the expression

    static constexpr bool is_etl         = true;                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast;                  ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                                ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;                                 ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                                ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = true;                                 ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                                ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                                ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = true;                                 ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = true;                                 ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order = sub_traits::storage_order;            ///< The expression's storage order
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled; ///< Indicates if the expression can be computed on GPU

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
        static_assert(DD == 0, "Invalid dimensions access");
        return decay_traits<A>::template dim<1>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, [[maybe_unused]] size_t d) {
        cpp_assert(d == 0, "Invalid dimensions access");
        return etl::dim<1>(e._a);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::dim<1>(e._a);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return decay_traits<A>::template dim<1>();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 1;
    }

    /*!
     * \brief Estimate the complexity of computation
     * \return An estimation of the complexity of the expression
     */
    static constexpr int complexity() noexcept {
        return -1;
    }
};

/*!
 * \brief Returns the transpose of the given expression.
 * \param value The expression
 * \return The transpose of the given expression.
 */
template <etl_4d A, etl_1d B>
bias_batch_var_4d_expr<detail::build_type<A>, detail::build_type<B>> bias_batch_var_4d(const A& a, const B& b) {
    return bias_batch_var_4d_expr<detail::build_type<A>, detail::build_type<B>>{a, b};
}

} //end of namespace etl
