//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

//Get the implementations
#include "etl/impl/pooling.hpp"

namespace etl {

/*!
 * \brief Base class for all 3D pooling derivative expressions
 */
template <typename T, size_t D, size_t C1, size_t C2, size_t C3, typename Impl>
struct basic_pool_derivative_3d_expr : impl_expr<basic_pool_derivative_3d_expr<T, D, C1, C2, C3, Impl>, T> {
    using value_type = T;                                                  ///< The type of values of this expression
    using this_type  = basic_pool_derivative_3d_expr<T, D, C1, C2, C3, Impl>; ///< The type of this expression

    /*!
     * \brief The result type for given sub types
     * \tparam A The sub expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, T, A, B>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and b and store the result in c
     * \param a The in(in) expression
     * \param b The in(out) expression
     * \param c The output expression
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "pool_derivative_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == decay_traits<A>::dimensions() && decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "pool_derivative_2d needs matrices of the same dimension");

        Impl::template apply<C1, C2, C3>(
            std::forward<A>(a),
            std::forward<B>(b),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "pool_derivative";
    }

    /*!
     * \brief Return the DDth dim of the expression
     * \tparam A The in(in) type
     * \tparam B The in(out) type
     * \tparam DD The dimension number
     * \return The DDth dimension of the epxression
     */
    template <typename A, typename B, size_t DD>
    static constexpr size_t dim() {
        return decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The in(in) expression
     * \param b The in(out) expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A, typename B>
    static size_t dim(const A& a, const B& b, size_t d) {
        cpp_unused(b);
        return etl::dim(a, d);
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in(in) expression
     * \param b The in(out) expression
     * \return the size of the expression given the input
     */
    template <typename A, typename B>
    static size_t size(const A& a, const B& b) {
        cpp_unused(b);
        return etl::size(a);
    }

    /*!
     * \brief Return the size of the expression given the input expression types
     * \tparam A The in(in) expression type
     * \tparam B The in(out) expression type
     * \return the size of the expression given the input types
     */
    template <typename A, typename B>
    static constexpr size_t size() {
        return decay_traits<A>::size();
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A, typename B>
    static constexpr etl::order order() {
        return etl::order::RowMajor;
    }

    /*!
     * \brief Return the number of dimensions of the expression
     * \return the nubmer of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Max Pooling Derivate 2D expression
 */
template <typename T, size_t D, size_t C1, size_t C2>
using max_pool_derivative_2d_expr = basic_pool_derivative_3d_expr<T, D, C1, C2, 0, impl::max_pool_derivative_2d>;

/*!
 * \brief Max Pooling Derivate 3D expression
 */
template <typename T, size_t D, size_t C1, size_t C2, size_t C3>
using max_pool_derivative_3d_expr = basic_pool_derivative_3d_expr<T, D, C1, C2, C3, impl::max_pool_derivative_3d>;

/*!
 * \brief Base class for all 3D pooling derivative expressions
 */
template <typename T, size_t D, typename Impl>
struct basic_dyn_pool_derivative_3d_expr : dyn_impl_expr<basic_dyn_pool_derivative_3d_expr<T, D, Impl>, T> {
    using value_type = T;                                                  ///< The type of values of this expression
    using this_type  = basic_dyn_pool_derivative_3d_expr<T, D, Impl>; ///< The type of this expression

    /*!
     * \brief The result type for given sub types
     * \tparam A The sub expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, T, A, B>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    const size_t c1; ///< First dimension pooling ratio
    const size_t c2; ///< Second dimension pooling ratio
    const size_t c3; ///< Third dimension pooling ratio

    /*!
     * \brief Construct a new basic_dyn_pool_derivative_3d_expr
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    basic_dyn_pool_derivative_3d_expr(size_t c1, size_t c2, size_t c3 = 0) : c1(c1), c2(c2), c3(c3) {
        // Nothing else to init
    }

    /*!
     * \brief Apply the expression on a and b and store the result in c
     * \param a The in(in) expression
     * \param b The in(out) expression
     * \param c The output expression
     */
    template <typename A, typename B, typename C>
    void apply(A&& a, B&& b, C&& c) const {
        static_assert(all_etl_expr<A, B, C>::value, "pool_derivative_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == decay_traits<A>::dimensions() && decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "pool_derivative_2d needs matrices of the same dimensions");

        Impl::template apply(
            std::forward<A>(a),
            std::forward<B>(b),
            std::forward<C>(c),
            c1, c2, c3);
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "pool_derivative_2d";
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The in(in) expression
     * \param b The in(out) expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A, typename B>
    static size_t dim(const A& a, const B& b, size_t d) {
        cpp_unused(b);
        return etl::dim(a, d);
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in(in) expression
     * \param b The in(out) expression
     * \return the size of the expression given the input
     */
    template <typename A, typename B>
    static size_t size(const A& a, const B& b) {
        cpp_unused(b);
        return etl::size(a);
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A, typename B>
    static constexpr etl::order order() {
        return etl::order::RowMajor;
    }

    /*!
     * \brief Return the number of dimensions of the expression
     * \return the nubmer of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Max Pooling Derivate 2D expression
 */
template <typename T, size_t D>
using dyn_max_pool_derivative_2d_expr = basic_dyn_pool_derivative_3d_expr<T, D, impl::max_pool_derivative_2d>;

/*!
 * \brief Max Pooling Derivate 3D expression
 */
template <typename T, size_t D>
using dyn_max_pool_derivative_3d_expr = basic_dyn_pool_derivative_3d_expr<T, D, impl::max_pool_derivative_3d>;

} //end of namespace etl
