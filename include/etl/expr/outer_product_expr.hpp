//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Include the implementations
#include "etl/impl/std/outer.hpp"
#include "etl/impl/blas/outer.hpp"
#include "etl/impl/cublas/outer.hpp"
#include "etl/impl/vec/outer.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <etl_expr A, etl_expr B>
struct outer_product_expr : base_temporary_expr_bin<outer_product_expr<A, B>, A, B> {
    using value_type  = value_t<A>;                               ///< The type of value of the expression
    using this_type   = outer_product_expr<A, B>;                 ///< The type of this expression
    using base_type   = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using left_traits = decay_traits<A>;                          ///< The traits of the sub type

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
    explicit outer_product_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Select the outer product implementation for an expression of type A and B
     *
     * This does not take the local context into account
     *
     * \tparam C The type of c expression
     * \return The implementation to use
     */
    template <etl_expr C>
    static constexpr etl::outer_impl select_default_outer_impl() {
        if (cblas_enabled) {
            return etl::outer_impl::BLAS;
        } else {
            return etl::outer_impl::STD;
        }
    }

#ifdef ETL_MANUAL_SELECT

    /*!
     * \brief Select the outer product implementation for an expression of type A and B
     * \tparam C The type of c expression
     * \return The implementation to use
     */
    template <etl_expr C>
    static etl::outer_impl select_outer_impl() {
        if (local_context().outer_selector.forced) {
            auto forced = local_context().outer_selector.impl;

            switch (forced) {
                //AVX cannot always be used
                case outer_impl::BLAS:
                    if (!cblas_enabled) {
                        std::cerr << "Forced selection to BLAS outer implementation, but not possible for this expression" << std::endl;
                        return select_default_outer_impl<C>();
                    }

                    return forced;

                //In other cases, simply use the forced impl
                default:
                    return forced;
            }
        }

        return select_default_outer_impl<C>();
    }

#else

    /*!
     * \brief Select the outer product implementation for an expression of type A and B
     *
     * \tparam C The type of c expression
     * \return The implementation to use
     */
    template <etl_expr C>
    static constexpr etl::outer_impl select_outer_impl() {
        return select_default_outer_impl<C>();
    }

#endif

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template <etl_expr C>
    void assign_to(C&& c) const {
        inc_counter("temp:assign");

        auto& a = this->a();
        auto& b = this->b();

        constexpr_select auto impl = select_outer_impl<C>();

        if
            constexpr_select(impl == etl::outer_impl::BLAS) {
                inc_counter("impl:blas");
                etl::impl::blas::outer(smart_forward(a), smart_forward(b), c);
            }
        else {
            inc_counter("impl:std");
            etl::impl::standard::outer(smart_forward(a), smart_forward(b), c);
        }
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
    friend std::ostream& operator<<(std::ostream& os, const outer_product_expr& expr) {
        return os << "out(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B>
struct etl_traits<etl::outer_product_expr<A, B>> {
    using expr_t       = etl::outer_product_expr<A, B>; ///< The expression type
    using left_expr_t  = std::decay_t<A>;               ///< The left sub expression type
    using right_expr_t = std::decay_t<B>;               ///< The right sub expression type
    using left_traits  = etl_traits<left_expr_t>;       ///< The left sub traits
    using right_traits = etl_traits<right_expr_t>;      ///< The right sub traits
    using value_type   = value_t<A>;                    ///< The value type of the expression

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
        return DD == 0 ? decay_traits<A>::template dim<0>() : decay_traits<B>::template dim<0>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        if (d == 0) {
            return etl::dim(e._a, 0);
        } else {
            return etl::dim(e._b, 0);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::dim(e._a, 0) * etl::dim(e._b, 0);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return decay_traits<A>::template dim<0>() * decay_traits<B>::template dim<0>();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 2;
    }

    /*!
     * \brief Estimate the complexity of computation
     * \return An estimation of the complexity of the expression
     */
    static constexpr int complexity() noexcept {
        return 4;
    }
};

/*!
 * \brief Outer product multiplication of two matrices
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <etl_expr A, etl_expr B>
outer_product_expr<detail::build_type<A>, detail::build_type<B>> outer(A&& a, B&& b) {
    return outer_product_expr<detail::build_type<A>, detail::build_type<B>>{a, b};
}

/*!
 * \brief Outer product multiplication of two matrices and store the result in c
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \param c The expression used to store the result
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <etl_expr A, etl_expr B, etl_expr C>
auto outer(A&& a, B&& b, C&& c) {
    c = outer(a, b);
    return c;
}

} //end of namespace etl
