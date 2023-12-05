//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

#include "etl/impl/cudnn/bias_batch_mean.hpp"
#include "etl/impl/egblas/bias_batch_sum.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <etl_expr A, bool Mean>
struct bias_batch_mean_2d_expr : base_temporary_expr_un<bias_batch_mean_2d_expr<A, Mean>, A> {
    using value_type = value_t<A>;                           ///< The type of value of the expression
    using this_type  = bias_batch_mean_2d_expr<A, Mean>;     ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>; ///< The base type
    using sub_traits = decay_traits<A>;                      ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = (!Mean && cudnn_enabled && is_floating<A>)
            || (impl::egblas::has_sbias_batch_sum && all_row_major<A> && all_single_precision<A>)
            || (impl::egblas::has_dbias_batch_sum && all_row_major<A> && all_double_precision<A>);

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit bias_batch_mean_2d_expr(A a) : base_type(a) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const C& c) {
        static_assert(etl::dimensions<C>() == 1, "The output of bias_batch_mean_2d is a vector");
        static_assert(etl::dimensions<A>() == 2, "The input of bias_batch_mean_2d is a 2d matrix");

        if constexpr (all_fast<A, C>) {
            static_assert(etl::dim<1, A>() == etl::dim<0, C>(), "Invalid dimensions for bias_batch_mean_2d");
        } else {
            cpp_assert(etl::dim<1>(a) == etl::dim<0>(c), "Invalid dimensions for bias_batch_mean_2d");
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

        using T = value_t<A>;

        check(a, lhs);

        if constexpr (!Mean && impl::egblas::has_sbias_batch_sum && all_row_major<A> && all_floating<A, L>) {
            const auto N = etl::dim<0>(a);
            const auto K = etl::dim<1>(a);

            decltype(auto) t1 = smart_forward_gpu(a);
            t1.ensure_gpu_up_to_date();

            lhs.ensure_gpu_allocated();

            impl::egblas::bias_batch_sum(N, K, t1.gpu_memory(), 1, lhs.gpu_memory(), 1);

            lhs.validate_gpu();
            lhs.invalidate_cpu();
        } else if constexpr (Mean && impl::egblas::has_sbias_batch_mean && all_row_major<A> && all_floating<A, L>) {
            const auto N = etl::dim<0>(a);
            const auto K = etl::dim<1>(a);

            decltype(auto) t1 = smart_forward_gpu(a);
            t1.ensure_gpu_up_to_date();

            lhs.ensure_gpu_allocated();

            impl::egblas::bias_batch_mean(N, K, t1.gpu_memory(), 1, lhs.gpu_memory(), 1);

            lhs.validate_gpu();
            lhs.invalidate_cpu();
        } else if constexpr (!Mean && cudnn_enabled && all_floating<A, L>) {
            impl::cudnn::bias_batch_mean_2d(smart_forward_gpu(a), lhs);
        } else {
            const auto N = etl::dim<0>(a);
            const auto K = etl::dim<1>(a);

            standard_evaluator::pre_assign_rhs(a);

            auto batch_fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < N; ++b) {
                        mean += a(b, k);
                    }

                    if constexpr (Mean) {
                        lhs(k) = mean / static_cast<T>(N);
                    } else {
                        lhs(k) = mean;
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_k, 0, K, 4UL);
        }
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_add_to(L&& lhs) const {
        if constexpr (all_floating<A, L> && ((!Mean && cudnn_enabled) || (all_row_major<A, L> && impl::egblas::has_sbias_batch_sum))) {
            std_add_evaluate(*this, lhs);
        } else {
            auto& a = this->a();

            standard_evaluator::pre_assign_rhs(a);

            const auto N = etl::size(a) / etl::size(lhs);
            const auto K = etl::size(lhs);

            using T = value_t<A>;

            check(a, lhs);

            auto batch_fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < N; ++b) {
                        mean += a(b, k);
                    }

                    if constexpr (Mean) {
                        lhs(k) += mean / static_cast<T>(N);
                    } else {
                        lhs(k) += mean;
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_k, 0, K, 4UL);
        }
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_sub_to(L&& lhs) const {
        if constexpr (all_floating<A, L> && ((!Mean && cudnn_enabled) || (all_row_major<A, L> && impl::egblas::has_sbias_batch_sum))) {
            std_sub_evaluate(*this, lhs);
        } else {
            auto& a = this->a();

            standard_evaluator::pre_assign_rhs(a);

            const auto N = etl::size(a) / etl::size(lhs);
            const auto K = etl::size(lhs);

            using T = value_t<A>;

            check(a, lhs);

            auto batch_fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < N; ++b) {
                        mean += a(b, k);
                    }

                    if constexpr (Mean) {
                        lhs(k) -= mean / static_cast<T>(N);
                    } else {
                        lhs(k) -= mean;
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_k, 0, K, 4UL);
        }
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mul_to(L&& lhs) const {
        if constexpr (all_floating<A, L> && ((!Mean && cudnn_enabled) || (all_row_major<A, L> && impl::egblas::has_sbias_batch_sum))) {
            std_mul_evaluate(*this, lhs);
        } else {
            auto& a = this->a();

            standard_evaluator::pre_assign_rhs(a);

            const auto N = etl::size(a) / etl::size(lhs);
            const auto K = etl::size(lhs);

            using T = value_t<A>;

            check(a, lhs);

            auto batch_fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < N; ++b) {
                        mean += a(b, k);
                    }

                    if constexpr (Mean) {
                        lhs(k) *= mean / static_cast<T>(N);
                    } else {
                        lhs(k) *= mean;
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_k, 0, K, 4UL);
        }
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_div_to(L&& lhs) const {
        if constexpr (all_floating<A, L> && ((!Mean && cudnn_enabled) || (all_row_major<A, L> && impl::egblas::has_sbias_batch_sum))) {
            std_div_evaluate(*this, lhs);
        } else {
            auto& a = this->a();

            standard_evaluator::pre_assign_rhs(a);

            const auto N = etl::size(a) / etl::size(lhs);
            const auto K = etl::size(lhs);

            using T = value_t<A>;

            check(a, lhs);

            auto batch_fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < N; ++b) {
                        mean += a(b, k);
                    }

                    if constexpr (Mean) {
                        lhs(k) /= mean / static_cast<T>(N);
                    } else {
                        lhs(k) /= mean;
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_k, 0, K, 4UL);
        }
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mod_to(L&& lhs) const {
        if constexpr (all_floating<A, L> && ((!Mean && cudnn_enabled) || (all_row_major<A, L> && impl::egblas::has_sbias_batch_sum))) {
            std_mod_evaluate(*this, lhs);
        } else {
            auto& a = this->a();

            standard_evaluator::pre_assign_rhs(a);

            const auto N = etl::size(a) / etl::size(lhs);
            const auto K = etl::size(lhs);

            using T = value_t<A>;

            check(a, lhs);

            auto batch_fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < N; ++b) {
                        mean += a(b, k);
                    }

                    if constexpr (Mean) {
                        lhs(k) %= mean / static_cast<T>(N);
                    } else {
                        lhs(k) %= mean;
                    }
                }
            };

            engine_dispatch_1d_serial(batch_fun_k, 0, K, 4UL);
        }
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const bias_batch_mean_2d_expr& expr) {
        if (Mean) {
            return os << "bias_batch_mean_2d(" << expr._a << ")";
        } else {
            return os << "bias_batch_sum_2d(" << expr._a << ")";
        }
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, bool Mean>
struct etl_traits<etl::bias_batch_mean_2d_expr<A, Mean>> {
    using expr_t     = etl::bias_batch_mean_2d_expr<A, Mean>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                       ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                ///< The sub traits
    using value_type = value_t<A>;                            ///< The value type of the expression

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
template <etl_2d E>
bias_batch_mean_2d_expr<detail::build_type<E>, true> bias_batch_mean_2d(const E& value) {
    return bias_batch_mean_2d_expr<detail::build_type<E>, true>{value};
}

/*!
 * \brief Returns the transpose of the given expression.
 * \param value The expression
 * \return The transpose of the given expression.
 */
template <etl_2d E>
bias_batch_mean_2d_expr<detail::build_type<E>, false> bias_batch_sum_2d(const E& value) {
    return bias_batch_mean_2d_expr<detail::build_type<E>, false>{value};
}

} //end of namespace etl
