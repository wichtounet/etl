//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

#include "etl/impl/cudnn/bias_batch_mean.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, bool Mean>
struct bias_batch_mean_4d_expr : base_temporary_expr_un<bias_batch_mean_4d_expr<A, Mean>, A> {
    using value_type = value_t<A>;                           ///< The type of value of the expression
    using this_type  = bias_batch_mean_4d_expr<A, Mean>;     ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>; ///< The base type
    using sub_traits = decay_traits<A>;                      ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = !Mean && cudnn_enabled && is_floating<A>;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit bias_batch_mean_4d_expr(A a) : base_type(a) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const C& c) {
        static_assert(etl::dimensions<C>() == 1, "The output of bias_batch_mean_4d is a vector");
        static_assert(etl::dimensions<A>() == 4, "The input of bias_batch_mean_4d is a 4D matrix");

        if constexpr (all_fast<A, C>) {
            static_assert(etl::dim<1, A>() == etl::dim<0, C>(), "Invalid dimensions for bias_batch_mean_4d");
        } else {
            cpp_assert(etl::dim<1>(a) == etl::dim<0>(c), "Invalid dimensions for bias_batch_mean_4d");
        }
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_4d only supported for ETL expressions");

        auto& a = this->a();

        using T = value_t<A>;

        check(a, lhs);

        if constexpr (!Mean && cudnn_enabled && all_floating<A, L>) {
            impl::cudnn::bias_batch_mean_4d(smart_forward_gpu(a), lhs);
        } else {
            const auto N = etl::size(a) / etl::size(lhs);
            const auto K = etl::size(lhs);

            standard_evaluator::pre_assign_rhs(a);

            a.ensure_cpu_up_to_date();

            auto batch_fun_k = [&](const size_t first, const size_t last) {
                CPU_SECTION {
                    for (size_t k = first; k < last; ++k) {
                        T mean(0);

                        for (size_t b = 0; b < etl::dim<0>(a); ++b) {
                            mean += sum(a(b)(k));
                        }

                        if constexpr (Mean) {
                            lhs(k) = mean / N;
                        } else {
                            lhs(k) = mean;
                        }
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
    template <typename L>
    void assign_add_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_4d only supported for ETL expressions");

        auto& a = this->a();

        standard_evaluator::pre_assign_rhs(a);

        a.ensure_cpu_up_to_date();
        a.ensure_gpu_up_to_date();

        const auto N = etl::size(a) / etl::size(lhs);
        const auto K = etl::size(lhs);

        using T = value_t<A>;

        check(a, lhs);

        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < etl::dim<0>(a); ++b) {
                        mean += sum(a(b)(k));
                    }

                    if constexpr (Mean) {
                        lhs(k) += mean / N;
                    } else {
                        lhs(k) += mean;
                    }
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
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_4d only supported for ETL expressions");

        auto& a = this->a();

        standard_evaluator::pre_assign_rhs(a);

        a.ensure_cpu_up_to_date();
        a.ensure_gpu_up_to_date();

        [[maybe_unused]] const auto N = etl::size(a) / etl::size(lhs);
        const auto K = etl::size(lhs);

        using T = value_t<A>;

        check(a, lhs);

        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < etl::dim<0>(a); ++b) {
                        mean += sum(a(b)(k));
                    }

                    if constexpr (Mean) {
                        lhs(k) -= mean / N;
                    } else {
                        lhs(k) -= mean;
                    }
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
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_4d only supported for ETL expressions");

        auto& a = this->a();

        standard_evaluator::pre_assign_rhs(a);

        a.ensure_cpu_up_to_date();
        a.ensure_gpu_up_to_date();

        [[maybe_unused]] const auto N = etl::size(a) / etl::size(lhs);
        const auto K = etl::size(lhs);

        using T = value_t<A>;

        check(a, lhs);

        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < etl::dim<0>(a); ++b) {
                        mean += sum(a(b)(k));
                    }

                    if constexpr (Mean) {
                        lhs(k) *= mean / N;
                    } else {
                        lhs(k) *= mean;
                    }
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
    template <typename L>
    void assign_div_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_4d only supported for ETL expressions");

        auto& a = this->a();

        standard_evaluator::pre_assign_rhs(a);

        a.ensure_cpu_up_to_date();
        a.ensure_gpu_up_to_date();

        [[maybe_unused]] const auto N = etl::size(a) / etl::size(lhs);
        const auto K = etl::size(lhs);

        using T = value_t<A>;

        check(a, lhs);

        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < etl::dim<0>(a); ++b) {
                        mean += sum(a(b)(k));
                    }

                    if constexpr (Mean) {
                        lhs(k) /= mean / N;
                    } else {
                        lhs(k) /= mean;
                    }
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
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_4d only supported for ETL expressions");

        auto& a = this->a();

        standard_evaluator::pre_assign_rhs(a);

        a.ensure_cpu_up_to_date();
        a.ensure_gpu_up_to_date();

        [[maybe_unused]] const auto N = etl::size(a) / etl::size(lhs);
        const auto K = etl::size(lhs);

        using T = value_t<A>;

        check(a, lhs);

        lhs.ensure_cpu_up_to_date();

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            CPU_SECTION {
                for (size_t k = first; k < last; ++k) {
                    T mean(0);

                    for (size_t b = 0; b < etl::dim<0>(a); ++b) {
                        mean += sum(a(b)(k));
                    }

                    if constexpr (Mean) {
                        lhs(k) %= mean / N;
                    } else {
                        lhs(k) %= mean;
                    }
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
    friend std::ostream& operator<<(std::ostream& os, const bias_batch_mean_4d_expr& expr) {
        if (Mean) {
            return os << "bias_batch_mean_4d(" << expr._a << ")";
        } else {
            return os << "bias_batch_sum_4d(" << expr._a << ")";
        }
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, bool Mean>
struct etl_traits<etl::bias_batch_mean_4d_expr<A, Mean>> {
    using expr_t     = etl::bias_batch_mean_4d_expr<A, Mean>; ///< The expression type
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
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled; ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = sub_traits::storage_order;            ///< The expression's storage order

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
};

/*!
 * \brief Returns the transpose of the given expression.
 * \param value The expression
 * \return The transpose of the given expression.
 */
template <typename E>
bias_batch_mean_4d_expr<detail::build_type<E>, true> bias_batch_mean_4d(const E& value) {
    static_assert(is_etl_expr<E>, "etl::bias_batch_mean_4d can only be used on ETL expressions");
    static_assert(is_4d<E>, "etl::bias_batch_mean_4d is only defined for 4D input");

    return bias_batch_mean_4d_expr<detail::build_type<E>, true>{value};
}

/*!
 * \brief Returns the transpose of the given expression.
 * \param value The expression
 * \return The transpose of the given expression.
 */
template <typename E>
bias_batch_mean_4d_expr<detail::build_type<E>, false> bias_batch_sum_4d(const E& value) {
    static_assert(is_etl_expr<E>, "etl::bias_batch_sum_4d can only be used on ETL expressions");
    static_assert(is_4d<E>, "etl::bias_batch_sum_4d is only defined for 4D input");

    return bias_batch_mean_4d_expr<detail::build_type<E>, false>{value};
}

} //end of namespace etl
