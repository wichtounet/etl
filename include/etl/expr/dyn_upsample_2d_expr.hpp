//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

#include "etl/impl/std/upsample.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <etl_expr A>
struct dyn_upsample_2d_expr : base_temporary_expr_un<dyn_upsample_2d_expr<A>, A, false> {
    using value_type = value_t<A>;                                  ///< The type of value of the expression
    using this_type  = dyn_upsample_2d_expr<A>;                     ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A, false>; ///< The base type
    using sub_traits = decay_traits<A>;                             ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = false;

    const size_t c1; ///< The pooling ratio for the first dimension
    const size_t c2; ///< The pooling ratio for the second dimension

    const size_t s1; ///< The stride for the first dimension
    const size_t s2; ///< The stride for the second dimension

    const size_t p1; ///< The padding for the first dimension
    const size_t p2; ///< The padding for the second dimension

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit dyn_upsample_2d_expr(A a, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) :
            base_type(a), c1(c1), c2(c2), s1(s1), s2(s2), p1(p1), p2(p2) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template <same_dimensions<A> L>
    void assign_to(L&& lhs) const {
        inc_counter("temp:assign");

        auto& a = this->a();

        impl::standard::upsample_2d::apply(smart_forward(a), lhs, c1, c2, s1, s2, p1, p2);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const dyn_upsample_2d_expr& expr) {
        return os << "upsample2(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <etl_expr A>
struct etl_traits<etl::dyn_upsample_2d_expr<A>> {
    using expr_t     = etl::dyn_upsample_2d_expr<A>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;              ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;       ///< The sub traits
    using value_type = value_t<A>;                   ///< The value type of the expression

    static constexpr size_t D = sub_traits::dimensions(); ///< The number of dimensions of this expressions

    static constexpr bool is_etl         = true;                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = false;                                ///< Indicates if the expression is fast
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
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        if (d == D - 2) {
            return (etl::dim(e._a, d) - 1) * e.s1 + e.c1 - 2 * e.p1;
        } else if (d == D - 1) {
            return (etl::dim(e._a, d) - 1) * e.s2 + e.c2 - 2 * e.p2;
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
 * \brief Upsample the given 2D matrix expression
 * \param value The input expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Upsampling of the given expression
 */
template <etl_expr E>
dyn_upsample_2d_expr<detail::build_type<E>> upsample_2d(E&& value, size_t c1, size_t c2) {
    return dyn_upsample_2d_expr<detail::build_type<E>>{value, c1, c2, c1, c2, 0, 0};
}

/*!
 * \brief Upsample the given 2D matrix expression
 * \param value The input expression
 * \param c1 The first pooling ratio
 * \param c2 The second pooling ratio
 * \return A expression representing the Upsampling of the given expression
 */
template <etl_expr E>
dyn_upsample_2d_expr<detail::build_type<E>> upsample_2d(E&& value, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    return dyn_upsample_2d_expr<detail::build_type<E>>{value, c1, c2, s1, s2, p1, p2};
}

} //end of namespace etl
