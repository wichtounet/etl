//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/std/convmtx2.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, size_t K1, size_t K2>
struct convmtx_2d_expr : base_temporary_expr_un<convmtx_2d_expr<A, K1, K2>, A> {
    using value_type = value_t<A>;                           ///< The type of value of the expression
    using this_type  = convmtx_2d_expr<A, K1, K2>;           ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>; ///< The base type
    using sub_traits = decay_traits<A>;                      ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit convmtx_2d_expr(A a) : base_type(a) {
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

        impl::standard::convmtx2_direct::template apply<K1, K2>(smart_forward(a), c);
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
    friend std::ostream& operator<<(std::ostream& os, const convmtx_2d_expr& expr) {
        return os << "convmtx2(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, size_t K1, size_t K2>
struct etl_traits<etl::convmtx_2d_expr<A, K1, K2>> {
    using expr_t     = etl::convmtx_2d_expr<A, K1, K2>; ///< The expression type
    using this_type  = etl_traits<expr_t>;              ///< The type of this traits
    using sub_expr_t = std::decay_t<A>;                 ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;          ///< The sub traits
    using value_type = value_t<A>;                      ///< The value type of the expression

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
        return DD == 0 ? ((sub_traits::template dim<0>() + K1 - 1) * (sub_traits::template dim<1>() + K2 - 1)) : K1 * K2;
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        if (d == 0) {
            return (etl::dim<0>(e._a) + K1 - 1) * (etl::dim<1>(e._a) + K2 - 1);
        } else {
            return K1 * K2;
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return this_type::dim(e, 0) * this_type::dim(e, 1);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return this_type::template dim<0>() * this_type::template dim<1>();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Construct a matrix to compute a 2D convolution by matrix-matrix multiplication
 * \param a The 2D matrix to transform (the input of the convolution)
 * \tparam K1 The first dimension of the kernel
 * \tparam K2 The second dimension of the kernel
 * \return a matrix expression for convolution
 */
template <size_t K1, size_t K2, typename A>
convmtx_2d_expr<detail::build_type<A>, K1, K2> convmtx2_direct(A&& a) {
    static_assert(is_etl_expr<A>, "Convolution matrices only supported for ETL expressions");
    static_assert(is_2d<A>, "Convolutional matrix only works in 2D");

    return convmtx_2d_expr<detail::build_type<A>, K1, K2>{a};
}

} //end of namespace etl
