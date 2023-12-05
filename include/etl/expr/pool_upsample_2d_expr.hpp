//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/std/max_pooling_upsample.hpp"
#include "etl/impl/std/avg_pooling_upsample.hpp"
#include "etl/impl/cudnn/pooling_upsample.hpp"

namespace etl {

/*!
 * \brief A derivative of the 2D max pooling (combine derivative and upsampling for performance)
 * \tparam A The input type
 * \tparam B The output type
 * \tparam C The errors type
 */
template <etl_expr A, etl_expr B, etl_expr C, size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, bool Max>
struct pool_upsample_2d_expr : base_temporary_expr_tern<pool_upsample_2d_expr<A, B, C, C1, C2, S1, S2, P1, P2, Max>, A, B, C> {
    using value_type = value_t<A>;                                                  ///< The type of value of the expression
    using sub_traits = etl::decay_traits<A>;                                        ///< The traits of the first sub type
    using this_type  = pool_upsample_2d_expr<A, B, C, C1, C2, S1, S2, P1, P2, Max>; ///< The type of this expression
    using base_type  = base_temporary_expr_tern<this_type, A, B, C>;                ///< The base type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = cudnn_enabled && all_floating<A, B> && all_homogeneous<A, B>;

    friend struct etl_traits<pool_upsample_2d_expr>;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    pool_upsample_2d_expr(A a, B b, C c) : base_type(a, b, c) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename R>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const B& b, [[maybe_unused]] const C& c, [[maybe_unused]] const R& result) {
        static constexpr size_t D = etl::decay_traits<A>::dimensions();

        static_assert(etl::decay_traits<B>::dimensions() == D, "Invalid dimensions in max_pool_upsample_3d");
        static_assert(etl::decay_traits<C>::dimensions() == D, "Invalid dimensions in max_pool_upsample_3d");
        static_assert(etl::decay_traits<R>::dimensions() == D, "Invalid dimensions in max_pool_upsample_3d");

        if constexpr (all_fast<A, B, C, R>) {
            static_assert(etl::decay_traits<R>::size() == etl::decay_traits<A>::size(), "max_pool_upsample_2d:A and R must have the same size");
            static_assert(etl::decay_traits<B>::size() == etl::decay_traits<C>::size(), "max_pool_upsample_2d:B and C must have the same size");

            static_assert(etl::decay_traits<A>::template dim<D - 2>() == S1 * (etl::decay_traits<B>::template dim<D - 2>() - 1) + C1 - 2 * P1,
                          "Invalid pooling dimensions for max_pool_upsample_2d");
            static_assert(etl::decay_traits<A>::template dim<D - 1>() == S2 * (etl::decay_traits<B>::template dim<D - 1>() - 1) + C2 - 2 * P2,
                          "Invalid pooling dimensions for max_pool_upsample_2d");
        } else {
            cpp_assert(etl::size(result) == etl::size(a), "max_pool_upsample_2d:A and R must have the same size");
            cpp_assert(etl::size(b) == etl::size(c), "max_pool_upsample_2d:B and C must have the same size");

            cpp_assert(etl::dim<D - 2>(a) == S1 * (etl::dim<D - 2>(b) - 1) + C1 - 2 * P1, "Invalid pooling dimensions for max_pool_upsample_2d");
            cpp_assert(etl::dim<D - 1>(a) == S2 * (etl::dim<D - 1>(b) - 1) + C2 - 2 * P2, "Invalid pooling dimensions for max_pool_upsample_2d");
        }
    }

    /*!
     * \brief Select the pool implementation for an expression of type ABC->R
     *
     * This does not consider the local context
     *
     * \return The implementation to use
     */
    template <typename R>
    static constexpr etl::pool_impl select_default_impl(bool no_gpu) {
        if (cudnn_enabled && all_floating<A, B, C, R> && !no_gpu) {
            return etl::pool_impl::CUDNN;
        }

        return etl::pool_impl::STD;
    }

#ifdef ETL_MANUAL_SELECT

    /*!
     * \brief Select the pool implementation for an expression of type ABC->R
     * \return The implementation to use
     */
    template <typename R>
    static etl::pool_impl select_impl() {
        if (local_context().pool_selector.forced) {
            auto forced = local_context().pool_selector.impl;

            switch (forced) {
                // CUDNN cannot always be used
                case pool_impl::CUDNN:
                    if (!cudnn_enabled || !all_floating<A, B, C, R> || local_context().cpu) {                                            //COVERAGE_EXCLUDE_LINE
                        std::cerr << "Forced selection to CUDNN pool implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                        return select_default_impl<R>(local_context().cpu);                                                              //COVERAGE_EXCLUDE_LINE
                    }                                                                                                                    //COVERAGE_EXCLUDE_LINE

                    return forced;

                //In other cases, simply use the forced impl
                default:
                    return forced;
            }
        }

        return select_default_impl<R>(local_context().cpu);
    }

#else

    /*!
     * \brief Select the pool implementation for an expression of type ABC->R
     *
     * \return The implementation to use
     */
    template <typename R>
    static constexpr etl::pool_impl select_impl() {
        return select_default_impl<R>(false);
    }

#endif

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param result The expression to which assign
     */
    template <etl_expr R>
    void assign_to(R&& result) const {
        inc_counter("temp:assign");

        auto& a = this->a();
        auto& b = this->b();
        auto& c = this->c();

        check(a, b, c, result);

        constexpr_select auto impl = select_impl<R>();

        if constexpr (Max) {
            if
                constexpr_select(impl == pool_impl::STD) {
                    inc_counter("impl:std");
                    impl::standard::max_pool_upsample_2d::apply<C1, C2, S1, S2, P1, P2>(smart_forward(a), smart_forward(b), smart_forward(c), result);
                }
            else if
                constexpr_select(impl == pool_impl::CUDNN) {
                    inc_counter("impl:cudnn");
                    impl::cudnn::max_pool_upsample_2d::apply(smart_forward_gpu(a), smart_forward_gpu(b), smart_forward_gpu(c), result, C1, C2, S1, S2, P1, P2);
                }
            else {
                cpp_unreachable("Invalid pool implementation");
            }
        } else {
            if
                constexpr_select(impl == pool_impl::STD) {
                    inc_counter("impl:std");
                    impl::standard::avg_pool_upsample_2d::apply<C1, C2, S1, S2, P1, P2>(smart_forward(a), smart_forward(b), smart_forward(c), result);
                }
            else if
                constexpr_select(impl == pool_impl::CUDNN) {
                    inc_counter("impl:cudnn");
                    impl::cudnn::avg_pool_upsample_2d::apply(smart_forward_gpu(a), smart_forward_gpu(b), smart_forward_gpu(c), result, C1, C2, S1, S2, P1, P2);
                }
            else {
                cpp_unreachable("Invalid pool implementation");
            }
        }
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const pool_upsample_2d_expr& expr) {
        return os << "max_pool_upsample2(" << expr._a << ", " << expr._b << ", " << expr._c << ")";
    }
};

/*!
 * \brief Traits for a pooling usample expression
 * \tparam A The pooling usample sub type
 */
template <typename A, typename B, typename C, size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, bool Max>
struct etl_traits<etl::pool_upsample_2d_expr<A, B, C, C1, C2, S1, S2, P1, P2, Max>> {
    using expr_t     = etl::pool_upsample_2d_expr<A, B, C, C1, C2, S1, S2, P1, P2, Max>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                                          ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                                   ///< The sub traits
    using value_type = value_t<A>;                                               ///< The value type of the expression

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
        return decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        return etl::dim(e.a(), d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::size(e.a());
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return decay_traits<A>::size();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return sub_traits::dimensions();
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
 * \brief Derivative of the 2D Max Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1 = C1, size_t S2 = C2, size_t P1 = 0, size_t P2 = 0, etl_expr A, etl_expr B, etl_expr C>
pool_upsample_2d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, C1, C2, S1, S2, P1, P2, true> max_pool_upsample_2d(A&& input,
                                                                                                                                              B&& output,
                                                                                                                                              C&& errors) {
    return {input, output, errors};
}

/*!
 * \brief Derivative of the 2D Avg Pooling of the given matrix expression and upsampling.
 * \param input The input
 * \param output The output
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1 = C1, size_t S2 = C2, size_t P1 = 0, size_t P2 = 0, etl_expr A, etl_expr B, etl_expr C>
pool_upsample_2d_expr<detail::build_type<A>, detail::build_type<B>, detail::build_type<C>, C1, C2, S1, S2, P1, P2, false> avg_pool_upsample_2d(A&& input,
                                                                                                                                               B&& output,
                                                                                                                                               C&& errors) {
    return {input, output, errors};
}

} //end of namespace etl
