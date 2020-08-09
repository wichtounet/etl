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

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, typename B, typename Impl>
struct dyn_pool_derivative_expr : base_temporary_expr_bin<dyn_pool_derivative_expr<A, B, Impl>, A, B> {
    using value_type  = value_t<A>;                               ///< The type of value of the expression
    using this_type   = dyn_pool_derivative_expr<A, B, Impl>;     ///< The type of this expression
    using base_type   = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using left_traits = decay_traits<A>;                          ///< The traits of the sub type

    static constexpr auto storage_order = left_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = false;

    const size_t c1; ///< The pooling ratio for the first dimension
    const size_t c2; ///< The pooling ratio for the second dimension
    const size_t c3; ///< The pooling ratio for the third dimension

    const size_t s1; ///< The stride for the first dimension
    const size_t s2; ///< The stride for the second dimension
    const size_t s3; ///< The stride for the third dimension

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit dyn_pool_derivative_expr(A a, B b, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3) : base_type(a, b), c1(c1), c2(c2), c3(c3), s1(s1), s2(s2), s3(s3) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template <typename C>
    void assign_to(C&& c) const {
        static_assert(all_etl_expr<A, B, C>, "pool_derivative only supported for ETL expressions");

        inc_counter("temp:assign");

        auto& a = this->a();
        auto& b = this->b();

        Impl::apply(smart_forward(a), smart_forward(b), c, c1, c2, c3, s1, s2, s3);
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
    friend std::ostream& operator<<(std::ostream& os, const dyn_pool_derivative_expr& expr) {
        return os << "pool_derivative(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B, typename Impl>
struct etl_traits<etl::dyn_pool_derivative_expr<A, B, Impl>> {
    using expr_t       = etl::dyn_pool_derivative_expr<A, B, Impl>; ///< The expression type
    using left_expr_t  = std::decay_t<A>;                           ///< The left sub expression type
    using right_expr_t = std::decay_t<B>;                           ///< The right sub expression type
    using left_traits  = etl_traits<left_expr_t>;                   ///< The left sub traits
    using right_traits = etl_traits<right_expr_t>;                  ///< The right sub traits
    using value_type   = value_t<A>;                                ///< The value type of the expression

    static constexpr bool is_etl         = true;                                          ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                         ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                         ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                         ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = left_traits::is_fast && right_traits::is_fast; ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                                         ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;                                          ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                                         ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = true;                                          ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                                         ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                                         ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = true;                                          ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = true;                                          ///< Indicates if the expression needs a evaluator visitor
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled;          ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = left_traits::storage_order;                    ///< The expression's storage order

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
        return left_traits::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        return left_traits::dim(e._a, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return left_traits::size(e._a);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return left_traits::size();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return left_traits::dimensions();
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
 * \brief Derivative of the 2D Max Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Derivative of 2D Max Pooling of the input expression.
 */
template <typename E, typename F>
dyn_pool_derivative_expr<detail::build_type<E>, F, impl::max_pool_derivative_2d> max_pool_derivative_2d(E&& input, F&& output, size_t c1, size_t c2) {
    return dyn_pool_derivative_expr<detail::build_type<E>, F, impl::max_pool_derivative_2d>{input, output, c1, c2, 0, c1, c2, 0};
}

/*!
 * \brief Derivative of the 2D Max Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Derivative of 2D Max Pooling of the input expression.
 */
template <typename E, typename F>
dyn_pool_derivative_expr<detail::build_type<E>, F, impl::max_pool_derivative_2d> max_pool_derivative_2d(
        E&& input, F&& output, size_t c1, size_t c2, size_t s1, size_t s2) {
    return dyn_pool_derivative_expr<detail::build_type<E>, F, impl::max_pool_derivative_2d>{input, output, c1, c2, 0, s1, s2, 0};
}

/*!
 * \brief Derivative of the 3D Max Pooling of the given matrix expression
 * \param input The input
 * \param output The output
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \param c3 The third pooling ratio
 * \return A expression representing the Derivative of 3D Max Pooling of the input expression.
 */
template <typename E, typename F>
dyn_pool_derivative_expr<detail::build_type<E>, F, impl::max_pool_derivative_3d> max_pool_derivative_3d(
    E&& input, F&& output, size_t c1, size_t c2, size_t c3) {
    return dyn_pool_derivative_expr<detail::build_type<E>, F, impl::max_pool_derivative_3d>{input, output, c1, c2, c3, c1, c2, c3};
}

} //end of namespace etl
