//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

//Get the implementations
#include "etl/impl/pooling.hpp"

namespace etl {

/*!
 * \brief Base class for all 2D pooling derivative expressions
 */
template <typename T, std::size_t C1, std::size_t C2, typename Impl>
struct basic_pool_derivative_2d_expr : impl_expr<basic_pool_derivative_2d_expr<T, C1, C2, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");

    using value_type = T;                                              ///< The type of values of this expression
    using this_type  = basic_pool_derivative_2d_expr<T, C1, C2, Impl>; ///< The type of this expression

    /*!
     * \brief The result type for given sub types
     * \tparam A The in expression type
     * \tparam B The out expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, A, B>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and b and store the result in c
     * \param a The in(in) expression
     * \param b The in(out) expression
     * \param c The output expression
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "pool_derivative_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_derivative_2d needs 2D matrices");

        Impl::template apply<C1, C2>(
            std::forward<A>(a),
            std::forward<B>(b),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "pool_derivative_2d";
    }

    /*!
     * \brief Return the DDth dim of the expression
     * \tparam A The in(in) type
     * \tparam B The in(out) type
     * \tparam DD The dimension number
     * \return The DDth dimension of the epxression
     */
    template <typename A, typename B, std::size_t DD>
    static constexpr std::size_t dim() {
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
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
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
    static std::size_t size(const A& a, const B& b) {
        cpp_unused(b);
        return etl::dim<0>(a) * etl::dim<1>(a);
    }

    /*!
     * \brief Return the size of the expression given the input expression types
     * \tparam A The in(in) expression type
     * \tparam B The in(out) expression type
     * \return the size of the expression given the input types
     */
    template <typename A, typename B>
    static constexpr std::size_t size() {
        return this_type::template dim<A, B, 0>() * this_type::template dim<A, B, 1>();
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
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Max Pooling Derivate 2D expression
 */
template <typename T, std::size_t C1, std::size_t C2>
using max_pool_derivative_2d_expr = basic_pool_derivative_2d_expr<T, C1, C2, impl::max_pool_derivative_2d>;

/*!
 * \brief Base class for all 3D pooling derivative expressions
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t C3, typename Impl>
struct basic_pool_derivative_3d_expr : impl_expr<basic_pool_derivative_3d_expr<T, C1, C2, C3, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");

    using value_type = T;                                                  ///< The type of values of this expression
    using this_type  = basic_pool_derivative_3d_expr<T, C1, C2, C3, Impl>; ///< The type of this expression

    /*!
     * \brief The result type for given sub types
     * \tparam A The sub expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, A, B>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and b and store the result in c
     * \param a The in(in) expression
     * \param b The in(out) expression
     * \param c The output expression
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "pool_derivative_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_derivative_2d needs 2D matrices");

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
        return "pool_derivative_2d";
    }

    /*!
     * \brief Return the DDth dim of the expression
     * \tparam A The in(in) type
     * \tparam B The in(out) type
     * \tparam DD The dimension number
     * \return The DDth dimension of the epxression
     */
    template <typename A, typename B, std::size_t DD>
    static constexpr std::size_t dim() {
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
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
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
    static std::size_t size(const A& a, const B& b) {
        cpp_unused(b);
        return etl::dim<0>(a) * etl::dim<1>(a) * etl::dim<2>(a);
    }

    /*!
     * \brief Return the size of the expression given the input expression types
     * \tparam A The in(in) expression type
     * \tparam B The in(out) expression type
     * \return the size of the expression given the input types
     */
    template <typename A, typename B>
    static constexpr std::size_t size() {
        return this_type::template dim<A, B, 0>() * this_type::template dim<A, B, 1>() * this_type::template dim<A, B, 2>();
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
    static constexpr std::size_t dimensions() {
        return 3;
    }
};

/*!
 * \brief Max Pooling Derivate 3D expression
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t C3>
using max_pool_derivative_3d_expr = basic_pool_derivative_3d_expr<T, C1, C2, C3, impl::max_pool_derivative_3d>;

/*!
 * \brief Base class for all 2D pooling derivative expressions
 */
template <typename T, typename Impl>
struct basic_dyn_pool_derivative_2d_expr : dyn_impl_expr<basic_dyn_pool_derivative_2d_expr<T, Impl>> {
    using value_type = T;                                          ///< The type of values of this expression
    using this_type  = basic_dyn_pool_derivative_2d_expr<T, Impl>; ///< The type of this expression

    /*!
     * \brief The result type for given sub types
     * \tparam A The in expression type
     * \tparam B The out expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, A, B>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    const std::size_t c1; ///< First dimension pooling ratio
    const std::size_t c2; ///< Second dimension pooling ratio

    /*!
     * \brief Construct anew basic_dyn_pool_derivative_2d_expr
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     */
    basic_dyn_pool_derivative_2d_expr(std::size_t c1, std::size_t c2) : c1(c1), c2(c2) {
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
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_derivative_2d needs 2D matrices");

        Impl::template apply(
            std::forward<A>(a),
            std::forward<B>(b),
            std::forward<C>(c),
            c1, c2);
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
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
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
    static std::size_t size(const A& a, const B& b) {
        cpp_unused(b);
        return etl::dim<0>(a) * etl::dim<1>(a);
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
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Max Pooling Derivate 2D expression
 */
template <typename T>
using dyn_max_pool_derivative_2d_expr = basic_dyn_pool_derivative_2d_expr<T, impl::max_pool_derivative_2d>;

/*!
 * \brief Base class for all 3D pooling derivative expressions
 */
template <typename T, typename Impl>
struct basic_dyn_pool_derivative_3d_expr : dyn_impl_expr<basic_dyn_pool_derivative_3d_expr<T, Impl>> {
    using value_type = T;                                                  ///< The type of values of this expression
    using this_type  = basic_dyn_pool_derivative_3d_expr<T, Impl>; ///< The type of this expression

    /*!
     * \brief The result type for given sub types
     * \tparam A The sub expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, A, B>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    const std::size_t c1; ///< First dimension pooling ratio
    const std::size_t c2; ///< Second dimension pooling ratio
    const std::size_t c3; ///< Third dimension pooling ratio

    /*!
     * \brief Construct a new basic_dyn_pool_derivative_3d_expr
     * \param c1 The first dimension pooling ratio
     * \param c2 The second dimension pooling ratio
     * \param c3 The third dimension pooling ratio
     */
    basic_dyn_pool_derivative_3d_expr(std::size_t c1, std::size_t c2, std::size_t c3) : c1(c1), c2(c2), c3(c3) {
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
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_derivative_2d needs 2D matrices");

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
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
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
    static std::size_t size(const A& a, const B& b) {
        cpp_unused(b);
        return etl::dim<0>(a) * etl::dim<1>(a) * etl::dim<2>(a);
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
    static constexpr std::size_t dimensions() {
        return 3;
    }
};

/*!
 * \brief Max Pooling Derivate 3D expression
 */
template <typename T>
using dyn_max_pool_derivative_3d_expr = basic_dyn_pool_derivative_3d_expr<T, impl::max_pool_derivative_3d>;

} //end of namespace etl
