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
 * \brief A batch softmax function expression
 *
 * \tparam A The unary sub type
 */
template <typename A, bool Stable>
struct batch_softmax_expr : base_temporary_expr_un<batch_softmax_expr<A, Stable>, A> {
    using value_type = value_t<A>;                           ///< The type of value of the expression
    using this_type  = batch_softmax_expr<A, Stable>;        ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>; ///< The base type
    using sub_traits = decay_traits<A>;                      ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit batch_softmax_expr(A a) : base_type(a) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the function dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const C& c) {
        static constexpr etl::order order_lhs = decay_traits<C>::storage_order;
        static constexpr etl::order order_rhs = decay_traits<A>::storage_order;

        static_assert(order_lhs == order_rhs, "Cannot change storage order");
        static_assert(decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "Invalid dimensions");

        if constexpr (all_fast<A, C>) {
            static_assert(decay_traits<A>::size() == decay_traits<C>::size(), "Invalid size");
        } else {
            cpp_assert(etl::size(a) == etl::size(c), "Invalid size");
        }
    }

    // Assignment functions

    /*!
     * \brief Select the best possible implementation for the batch softmax operation
     *
     * This routine does not consider the local context
     */
    template <typename C>
    constexpr static batch_softmax_impl select_default_impl(bool no_gpu) {
        if (cudnn_enabled && all_homogeneous<A, C> && all_floating<A, C> && !no_gpu) {
            return batch_softmax_impl::CUDNN;
        }

        return batch_softmax_impl::STD;
    }

#ifdef ETL_MANUAL_SELECT

    /*!
     * \brief Select the best possible implementation for the batch softmax operation
     */
    template <typename C>
    static batch_softmax_impl select_impl() {
        return select_default_impl<C>(local_context().cpu);
    }

#else

    /*!
     * \brief Select the best possible implementation for the batch softmax operation
     */
    template <typename C>
    constexpr static batch_softmax_impl select_impl() {
        return select_default_impl<C>(false);
    }

#endif

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template <etl_expr C>
    void assign_to(C&& c) const {
        if constexpr (decay_traits<C>::storage_order == storage_order) {
            inc_counter("temp:assign");

            auto& a = this->a();

            standard_evaluator::pre_assign_rhs(a);

            check(a, c);

            constexpr_select auto impl = select_impl<C>();

            if constexpr_select (impl == batch_softmax_impl::CUDNN) {
                inc_counter("impl:cudnn");

                decltype(auto) a_gpu = smart_forward_gpu(a);

                if constexpr (Stable) {
                    impl::cudnn::stable_softmax(a_gpu, c);
                } else {
                    impl::cudnn::softmax(a_gpu, c);
                }
            } else if constexpr_select (impl == batch_softmax_impl::STD) {
                inc_counter("impl:std");

                if constexpr (Stable) {
                    for (size_t i = 0; i < etl::dim<0>(c); ++i) {
                        c(i) = exp(a(i)) / sum(exp(a(i)));
                    }
                } else {
                    for (size_t i = 0; i < etl::dim<0>(c); ++i) {
                        auto m = max(a(i));
                        c(i)   = exp(a(i) - m) / sum(exp(a(i) - m));
                    }
                }
            } else {
                cpp_unreachable("Invalid selection for batch_softmax");
            }
        } else {
            std_assign_evaluate(*this, c);
        }
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const batch_softmax_expr& expr) {
        return os << "batch_softmax(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for an unary function expression
 * \tparam A The unary sub type
 */
template <typename A, bool Stable>
struct etl_traits<etl::batch_softmax_expr<A, Stable>> {
    using expr_t     = etl::batch_softmax_expr<A, Stable>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                    ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;             ///< The sub traits
    using value_type = value_t<A>;                         ///< The value type of the expression

    static constexpr bool is_etl         = true;                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast;                  ///< Indicates if the expression is fast
    static constexpr bool is_linear      = true;                                 ///< Indicates if the expression is linear
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
    using vectorizable = std::true_type;

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
        return etl::dim(e._a, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::size(e._a);
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
        return decay_traits<A>::dimensions();
    }

    /*!
     * \brief Estimate the complexity of computation
     * \return An estimation of the complexity of the expression
     */
    static constexpr int complexity() noexcept {
        return -1;
    }
};

} //end of namespace etl
