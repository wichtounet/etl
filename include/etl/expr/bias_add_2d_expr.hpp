//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

// Include the implementations
#include "etl/impl/std/bias_add.hpp"
#include "etl/impl/vec/bias_add.hpp"
#include "etl/impl/cudnn/bias_add.hpp"
#include "etl/impl/egblas/bias_add.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, typename B>
struct bias_add_2d_expr : base_temporary_expr_bin<bias_add_2d_expr<A, B>, A, B> {
    using value_type = value_t<A>;                               ///< The type of value of the expression
    using this_type  = bias_add_2d_expr<A, B>;                   ///< The type of this expression
    using base_type  = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using sub_traits = decay_traits<A>;                          ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = cudnn_enabled && all_floating<A, B> && all_homogeneous<A, B>;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit bias_add_2d_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const B& b, [[maybe_unused]] const C& c) {
        static_assert(etl::dimensions<A>() == 2, "The input of bias_add_2d is a 2D matrix");
        static_assert(etl::dimensions<B>() == 1, "The input of bias_add_2d is a vector of biases");
        static_assert(etl::dimensions<C>() == 2, "The output of bias_add_2d is a 2D matrix");

        if constexpr (all_fast<A, B, C>) {
            static_assert(etl::dim<1, A>() == etl::dim<0, B>(), "Invalid dimensions for bias_add_2d");

            static_assert(etl::dim<0, A>() == etl::dim<0, C>(), "Invalid dimensions for bias_add_2d");
            static_assert(etl::dim<1, A>() == etl::dim<1, C>(), "Invalid dimensions for bias_add_2d");
        } else {
            cpp_assert(etl::dim<1>(a) == etl::dim<0>(b), "Invalid dimensions for bias_add_2d");

            cpp_assert(etl::dim<0>(a) == etl::dim<0>(c), "Invalid dimensions for bias_add_2d");
            cpp_assert(etl::dim<1>(a) == etl::dim<1>(c), "Invalid dimensions for bias_add_2d");
        }
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        static_assert(all_etl_expr<A, L>, "bias_add_2d only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        constexpr_select auto impl = select_impl<L>();

        if constexpr_select (impl == bias_add_impl::VEC) {
            impl::vec::bias_add_2d(smart_forward(a), smart_forward(b), lhs);
        } else if constexpr_select (impl == bias_add_impl::STD) {
            impl::standard::bias_add_2d(smart_forward(a), smart_forward(b), lhs);
        } else if constexpr_select (impl == bias_add_impl::EGBLAS) {
            decltype(auto) e_x = smart_forward_gpu(a);
            decltype(auto) e_b = smart_forward_gpu(b);
            auto& e_y = lhs;

            e_x.ensure_gpu_up_to_date();
            e_b.ensure_gpu_up_to_date();
            e_y.ensure_gpu_allocated();

            impl::egblas::bias_add_2d(etl::dim<0>(a), etl::dim<1>(a), e_x.gpu_memory(), 1, e_b.gpu_memory(), 1, e_y.gpu_memory(), 1);

            e_y.validate_gpu();
            e_y.invalidate_cpu();
        } else if constexpr_select (impl == bias_add_impl::CUDNN) {
            impl::cudnn::bias_add_2d(smart_forward_gpu(a), smart_forward_gpu(b), lhs);
        } else {
            cpp_unreachable("Invalid bias_add_2d selection");
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
    friend std::ostream& operator<<(std::ostream& os, const bias_add_2d_expr& expr) {
        return os << "bias_add_2d(" << expr._a << "," << expr._b << ")";
    }

private:

    /*!
     * \brief Select the default implementation for this expression.
     *
     * This does not take the local context into account
     *
     * \tparam C The type of the result expression
     *
     * \return The implementation to use
     */
    template <typename C>
    static constexpr etl::bias_add_impl select_default_impl(bool no_gpu) {
        constexpr bool homo           = all_homogeneous<A, B, C>;
        constexpr bool vec_possible   = vec_enabled && vectorize_impl && all_vectorizable<vector_mode, A, B, C> && homo;
        constexpr bool cudnn_possible = cudnn_enabled && all_floating<A, B, C> && homo;

        if (homo && is_single_precision<A> && impl::egblas::has_sbias_add_2d) {
            return etl::bias_add_impl::EGBLAS;
        }

        if (homo && is_double_precision<A> && impl::egblas::has_dbias_add_2d) {
            return etl::bias_add_impl::EGBLAS;
        }

        if (cudnn_possible && !no_gpu) {
            return etl::bias_add_impl::CUDNN;
        }

        if(vec_possible){
            return etl::bias_add_impl::VEC;
        }

        return etl::bias_add_impl::STD;
    }

#ifdef ETL_MANUAL_SELECT

    /*!
     * \brief Select the implementation for this expression.
     * \tparam C The type of the result expression
     * \return The implementation to use
     */
    template <typename C>
    static etl::bias_add_impl select_impl() {
        auto def = select_default_impl<C>(local_context().cpu);

        if (local_context().bias_add_selector.forced) {
            auto forced = local_context().bias_add_selector.impl;

            switch (forced) {
                // EGBLAS cannot always be used
                case bias_add_impl::EGBLAS:
                    if (!all_homogeneous<A, B, C> || !((is_single_precision<A> && impl::egblas::has_sbias_add_2d) || (is_double_precision<A> && impl::egblas::has_sbias_add_2d)) || local_context().cpu) {
                        std::cerr << "Forced selection to EGBLAS bias_add implementation, but not possible for this expression" << std::endl;
                        return def;
                    }

                    return forced;

                //CUDNN cannot always be used
                case bias_add_impl::CUDNN:
                    if (!cudnn_enabled || !all_floating<A, B, C> || !all_homogeneous<A, B, C> || local_context().cpu) {
                        std::cerr << "Forced selection to cUDNN bias_add implementation, but not possible for this expression" << std::endl;
                        return def;
                    }

                    return forced;

                //VEC cannot always be used
                case bias_add_impl::VEC:
                    if (!vec_enabled || !vectorize_impl || !all_vectorizable<vector_mode, A, B, C> || !all_homogeneous<A, B, C>) {
                        std::cerr << "Forced selection to VEC bias_add_2d implementation, but not possible for this expression" << std::endl;
                        return def;
                    }

                    return forced;

                //In other cases, simply use the forced impl
                default:
                    return forced;
            }
        }

        return def;
    }

#else

    /*!
     * \brief Select the default implementation for this expression.
     *
     * \tparam C The type of the result expression
     *
     * \return The implementation to use
     */
    template <typename C>
    static constexpr etl::bias_add_impl select_impl() {
        return select_default_impl<C>(false);
    }

#endif
};

/*!
 * \brief Traits for a bias_add_2d expression
 * \tparam A The input type
 * \tparam B The biases type
 */
template <typename A, typename B>
struct etl_traits<etl::bias_add_2d_expr<A, B>> {
    using expr_t     = etl::bias_add_2d_expr<A, B>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;             ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;      ///< The sub traits
    using value_type = value_t<A>;                  ///< The value type of the expression

    static constexpr bool is_etl         = true;                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = all_fast<A, B>;                       ///< Indicates if the expression is fast
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
    static constexpr bool vectorizable = true;

    /*!
     * \brief Returns the DDth dimension of the expression
     * \return the DDth dimension of the expression
     */
    template <size_t DD>
    static constexpr size_t dim() {
        return sub_traits::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        return sub_traits::dim(e._a, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return sub_traits::size(e._a);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return sub_traits::size();
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
 * \brief Returns the result of adding the bias [K] to the 4D matrix [N1, K, N2, N3]
 * \param x The 4D matrix
 * \param biases The vector of biases
 * \return The transpose of the given expression.
 */
template <typename E, typename B>
bias_add_2d_expr<detail::build_type<E>, detail::build_type<B>> bias_add_2d(const E& x, const B& biases){
    static_assert(all_etl_expr<E, B>, "etl::bias_add_2d can only be used on ETL expressions");
    static_assert(is_2d<E>, "etl::bias_add_2d is only defined for 2D input");
    static_assert(is_1d<B>, "etl::bias_add_2d is only defined for 1D bias vector");

    return bias_add_2d_expr<detail::build_type<E>, detail::build_type<B>>{x, biases};
}

} //end of namespace etl
