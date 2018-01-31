//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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
template <typename A, typename B>
struct batch_outer_product_expr : base_temporary_expr_bin<batch_outer_product_expr<A, B>, A, B> {
    using value_type  = value_t<A>;                              ///< The type of value of the expression
    using this_type   = batch_outer_product_expr<A, B>;                   ///< The type of this expression
    using base_type   = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using left_traits = decay_traits<A>;                         ///< The traits of the sub type

    static constexpr auto storage_order = left_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = cublas_enabled;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit batch_outer_product_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Select the batch outer product implementation for an expression of type A and B
     *
     * This does not take the local context into account
     *
     * \tparam C The type of c expression
     * \return The implementation to use
     */
    template <typename C>
    static constexpr etl::outer_impl select_default_batch_outer_impl(bool no_gpu) {
        if (cublas_enabled && !no_gpu) {
            return etl::outer_impl::CUBLAS;
        }

        if (cblas_enabled) {
            return etl::outer_impl::BLAS;
        }

        if (vec_enabled) {
            return etl::outer_impl::VEC;
        }

        return etl::outer_impl::STD;
    }

#ifdef ETL_MANUAL_SELECT

    /*!
     * \brief Select the batch outer product implementation for an expression of type A and B
     * \tparam C The type of c expression
     * \return The implementation to use
     */
    template <typename C>
    static etl::outer_impl select_batch_outer_impl() {
        if (local_context().outer_selector.forced) {
            auto forced = local_context().outer_selector.impl;

            switch (forced) {
                //BLAS cannot always be used
                case outer_impl::BLAS:
                    if (!cblas_enabled) {
                        std::cerr << "Forced selection to BLAS outer implementation, but not possible for this expression" << std::endl;
                        return select_default_batch_outer_impl<C>(local_context().cpu);
                    }

                    return forced;

                //CUBLAS cannot always be used
                case outer_impl::CUBLAS:
                    if (!cublas_enabled || local_context().cpu) {
                        std::cerr << "Forced selection to CUBLAS outer implementation, but not possible for this expression" << std::endl;
                        return select_default_batch_outer_impl<C>(local_context().cpu);
                    }

                    return forced;

                //VEC cannot always be used
                case outer_impl::VEC:
                    if (!vec_enabled) {
                        std::cerr << "Forced selection to VEC outer implementation, but not possible for this expression" << std::endl;
                        return select_default_batch_outer_impl<C>(local_context().cpu);
                    }

                    return forced;

                //In other cases, simply use the forced impl
                default:
                    return forced;
            }
        }

        return select_default_batch_outer_impl<C>(local_context().cpu);
    }

#else

    /*!
     * \brief Select the batch_outer product implementation for an expression of type A and B
     *
     * \tparam C The type of c expression
     * \return The implementation to use
     */
    template <typename C>
    static constexpr etl::outer_impl select_batch_outer_impl() {
        return select_default_batch_outer_impl<C>(false);
    }

#endif

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template<typename C>
    void assign_to(C&& c)  const {
        static_assert(all_etl_expr<A, B, C>, "batch_outer_product only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        constexpr_select auto impl = select_batch_outer_impl<C>();

        if constexpr_select (impl == etl::outer_impl::STD) {
            etl::impl::standard::batch_outer(smart_forward(a), smart_forward(b), c);
        } else if constexpr_select (impl == etl::outer_impl::BLAS) {
            etl::impl::blas::batch_outer(smart_forward(a), smart_forward(b), c);
        } else if constexpr_select (impl == etl::outer_impl::CUBLAS) {
            etl::impl::cublas::batch_outer(smart_forward_gpu(a), smart_forward_gpu(b), c);
        } else if constexpr_select (impl == etl::outer_impl::VEC) {
            etl::impl::vec::batch_outer(smart_forward(a), smart_forward(b), c);
        } else {
            cpp_unreachable("Invalid batch_outer selection");
        }
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_add_to(L&& lhs)  const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_sub_to(L&& lhs)  const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mul_to(L&& lhs)  const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_div_to(L&& lhs)  const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mod_to(L&& lhs)  const {
        std_mod_evaluate(*this, lhs);
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const batch_outer_product_expr& expr) {
        return os << "batch_outer(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B>
struct etl_traits<etl::batch_outer_product_expr<A, B>> {
    using expr_t       = etl::batch_outer_product_expr<A, B>; ///< The expression type
    using left_expr_t  = std::decay_t<A>;                     ///< The left sub expression type
    using right_expr_t = std::decay_t<B>;                     ///< The right sub expression type
    using left_traits  = etl_traits<left_expr_t>;             ///< The left sub traits
    using right_traits = etl_traits<right_expr_t>;            ///< The right sub traits
    using value_type   = value_t<A>;                          ///< The value type of the expression

    static constexpr bool is_etl          = true;                                          ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer  = false;                                         ///< Indicates if the type is a transformer
    static constexpr bool is_view         = false;                                         ///< Indicates if the type is a view
    static constexpr bool is_magic_view   = false;                                         ///< Indicates if the type is a magic view
    static constexpr bool is_fast         = left_traits::is_fast && right_traits::is_fast; ///< Indicates if the expression is fast
    static constexpr bool is_linear       = false;                                          ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe  = true;                                          ///< Indicates if the expression is thread safe
    static constexpr bool is_value        = false;                                         ///< Indicates if the expression is of value type
    static constexpr bool is_direct       = true;                                          ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator    = false;                                         ///< Indicates if the expression is a generator
    static constexpr bool is_padded       = false;                                         ///< Indicates if the expression is padded
    static constexpr bool is_aligned      = true;                                          ///< Indicates if the expression is padded
    static constexpr bool is_temporary = true;                                          ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order  = left_traits::storage_order;                    ///< The expression's storage order
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled;                                         ///< Indicates if the expression can be computed on GPU

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
        return DD == 0 ? decay_traits<A>::template dim<1>()
                       : decay_traits<B>::template dim<1>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        if (d == 0){
            return etl::dim(e._a, 1);
        } else {
            return etl::dim(e._b, 1);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::dim(e._a, 1) * etl::dim(e._b, 1);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return decay_traits<A>::template dim<1>() * decay_traits<B>::template dim<1>();
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
 * \brief Batch Outer product multiplication of two matrices
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the outer product multiplication of a and b
 */
template <typename A, typename B>
batch_outer_product_expr<detail::build_type<A>, detail::build_type<B>> batch_outer(A&& a, B&& b) {
    return batch_outer_product_expr<detail::build_type<A>, detail::build_type<B>>{a, b};
}

/*!
 * \brief Batch Outer product multiplication of two matrices and store the result in c
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \param c The expression used to store the result
 * \return An expression representing the outer product multiplication of a and b
 */
template <typename A, typename B, typename C>
auto batch_outer(A&& a, B&& b, C&& c){
    c = batch_outer(a, b);
    return c;
}

} //end of namespace etl
