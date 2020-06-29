//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/pooling.hpp"
#include "etl/impl/prob_pooling.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename Impl>
struct pool_2d_expr : base_temporary_expr_un<pool_2d_expr<A, C1, C2, S1, S2, P1, P2, Impl>, A> {
    using value_type = value_t<A>;                                    ///< The type of value of the expression
    using this_type  = pool_2d_expr<A, C1, C2, S1, S2, P1, P2, Impl>; ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>;          ///< The base type
    using sub_traits = decay_traits<A>;                               ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = Impl::template gpu_computable<A>;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit pool_2d_expr(A a) : base_type(a) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template <typename C>
    void assign_to(C&& c) const {
        static_assert(all_etl_expr<A, C>, "max_pool_2d only supported for ETL expressions");
        static_assert(etl::dimensions<A>() == etl::dimensions<C>(), "max_pool_2d must be applied on matrices of same dimensionality");

        auto& a = this->a();

        Impl::template apply<C1, C2, S1, S2, P1, P2>(a, c);
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
    friend std::ostream& operator<<(std::ostream& os, const pool_2d_expr& expr) {
        return os << "pool2(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename Impl>
struct etl_traits<etl::pool_2d_expr<A, C1, C2, S1, S2, P1, P2, Impl>> {
    using expr_t     = etl::pool_2d_expr<A, C1, C2, S1, S2, P1, P2, Impl>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                                    ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                             ///< The sub traits
    using value_type = value_t<A>;                                         ///< The value type of the expression

    static constexpr size_t D = sub_traits::dimensions(); ///< The number of dimensions of this expressions

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
        return DD == D - 2 ? (decay_traits<A>::template dim<DD>() - C1 + 2 * P1) / S1 + 1
                           : DD == D - 1 ? (decay_traits<A>::template dim<DD>() - C2 + 2 * P2) / S2 + 1 : decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        if (d == D - 2) {
            return (etl::dim(e._a, d) - C1 + 2 * P1) / S1 + 1;
        } else if (d == D - 1) {
            return (etl::dim(e._a, d) - C2 + 2 * P2) / S2 + 1;
        } else {
            return etl::dim(e._a, d);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        size_t acc = 1;
        for (size_t i = 0; i < D; ++i) {
            acc *= dim(e, i);
        }
        return acc;
    }

    /*!
     * \brief Returns the multiplicative sum of the dimensions at the given indices
     * \return the multiplicative sum of the dimensions at the given indices
     */
    template <size_t... I>
    static constexpr size_t size_mul(const std::index_sequence<I...>& /*seq*/) {
        return (dim<I>() * ...);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return size_mul(std::make_index_sequence<D>());
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return D;
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
 * \brief 2D Max Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the 2D Max Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1 = C1, size_t S2 = C2, size_t P1 = 0, size_t P2 = 0, typename E>
pool_2d_expr<detail::build_type<E>, C1, C2, S1, S2, P1, P2, impl::max_pool_2d> max_pool_2d(E&& value) {
    return pool_2d_expr<detail::build_type<E>, C1, C2, S1, S2, P1, P2, impl::max_pool_2d>{value};
}

/*!
 * \brief 2D Average Pooling of the given matrix expression
 * \param value The matrix expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the 2D Average Pooling of the input expression.
 */
template <size_t C1, size_t C2, size_t S1 = C1, size_t S2 = C2, size_t P1 = 0, size_t P2 = 0, typename E>
pool_2d_expr<detail::build_type<E>, C1, C2, S1, S2, P1, P2, impl::avg_pool_2d> avg_pool_2d(E&& value) {
    return pool_2d_expr<detail::build_type<E>, C1, C2, S1, S2, P1, P2, impl::avg_pool_2d>{value};
}

/*!
 * \brief Probabilistic Max Pooling for pooling units
 * \param value The input expression
 * \tparam C1 The first pooling ratio
 * \tparam C2 The second pooling ratio
 * \return A expression representing the Probabilistic Max Pooling of pooling units
 */
template <size_t C1, size_t C2, typename E>
pool_2d_expr<detail::build_type<E>, C1, C2, C1, C2, 0, 0, impl::standard::pmp_p_impl> p_max_pool_p(E&& value) {
    validate_pmax_pooling<C1, C2>(value);
    return pool_2d_expr<detail::build_type<E>, C1, C2, C1, C2, 0, 0, impl::standard::pmp_p_impl>{value};
}

} //end of namespace etl
