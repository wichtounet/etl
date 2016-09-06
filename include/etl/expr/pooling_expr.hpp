//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the pooling expressions.
*/

#pragma once

//Get the implementations
#include "etl/impl/pooling.hpp"

namespace etl {

/*!
 * \brief Base class for all 2D pooling expressions
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t S1, std::size_t S2, typename Impl>
struct basic_pool_2d_expr : impl_expr<basic_pool_2d_expr<T, C1, C2, S1, S2, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(S1 > 0, "S1 must be greater than 0");
    static_assert(S2 > 0, "S2 must be greater than 0");

    using value_type = T;                                           ///< Type of values of the expression
    using this_type  = basic_pool_2d_expr<T, C1, C2, S1, S2, Impl>; ///< The type of this expression

    /*!
     * \brief Compute the result type given the input type
     * \tparam A the input type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_2d needs 2D matrices");

        Impl::template apply<C1, C2, S1, S2>(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "pool_2d";
    }

    /*!
     * \brief Return the DDth dim of the expression
     * \tparam A The input type
     * \tparam DD The dimension number
     * \return The DDth dimension of the epxression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? (decay_traits<A>::template dim<0>() - C1) / S1 + 1
                       : (decay_traits<A>::template dim<1>() - C2) / S2 + 1;
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The input expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        if (d == 0) {
            return (etl::dim<0>(a) - C1) / S1 + 1;
        } else {
            return (etl::dim<1>(a) - C2) / S2 + 1;
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    static std::size_t size(const A& a) {
        return this_type::dim(a, 0) * this_type::dim(a, 1);
    }

    /*!
     * \brief Return the size of the expression given the input expression type
     * \tparam A The in expression type
     * \return the size of the expression given the input type
     */
    template <typename A>
    static constexpr std::size_t size() {
        return this_type::template dim<A, 0>() * this_type::template dim<A, 1>();
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A>
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
 * \brief Max Pooling 2D expression type
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t S1, std::size_t S2>
using max_pool_2d_expr = basic_pool_2d_expr<T, C1, C2, S1, S2, impl::max_pool_2d>;

/*!
 * \brief Average Pooling 2D expression type
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t S1, std::size_t S2>
using avg_pool_2d_expr = basic_pool_2d_expr<T, C1, C2, S1, S2, impl::avg_pool_2d>;

/*!
 * \brief Base class for all 3D pooling expressions
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t C3, typename Impl>
struct basic_pool_3d_expr : impl_expr<basic_pool_3d_expr<T, C1, C2, C3, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");

    using value_type = T;                                       ///< The type of values of the expression
    using this_type  = basic_pool_3d_expr<T, C1, C2, C3, Impl>; ///< The type of this expression

    /*!
     * \brief Compute the result type given the input type
     * \tparam A the input type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_3d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_3d needs 3D matrices");

        Impl::template apply<C1, C2, C3>(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "pool_3d";
    }

    /*!
     * \brief Return the DDth dim of the expression
     * \tparam A The input type
     * \tparam DD The dimension number
     * \return The DDth dimension of the epxression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? decay_traits<A>::template dim<0>() / C1
                       : DD == 1 ? decay_traits<A>::template dim<1>() / C2
                                 : decay_traits<A>::template dim<2>() / C3;
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The input expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        if (d == 0) {
            return etl::dim<0>(a) / C1;
        } else if (d == 1) {
            return etl::dim<1>(a) / C2;
        } else {
            return etl::dim<2>(a) / C3;
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    static std::size_t size(const A& a) {
        return (etl::dim<0>(a) / C1) * (etl::dim<1>(a) / C2) * (etl::dim<2>(a) / C3);
    }

    /*!
     * \brief Return the size of the expression given the input expression type
     * \tparam A The in expression type
     * \return the size of the expression given the input type
     */
    template <typename A>
    static constexpr std::size_t size() {
        return this_type::template dim<A, 0>() * this_type::template dim<A, 1>() * this_type::template dim<A, 2>();
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A>
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
 * \brief Max Pooling 3D expression type
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t C3>
using max_pool_3d_expr = basic_pool_3d_expr<T, C1, C2, C3, impl::max_pool_3d>;

/*!
 * \brief Average Pooling 3D expression type
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t C3>
using avg_pool_3d_expr = basic_pool_3d_expr<T, C1, C2, C3, impl::avg_pool_3d>;

/*!
 * \brief Base class for all dynamic 2D pooling expressions
 */
template <typename T, typename Impl>
struct basic_dyn_pool_2d_expr : dyn_impl_expr<basic_dyn_pool_2d_expr<T, Impl>> {
    using value_type = T;                               ///< Type of values of the expression
    using this_type  = basic_dyn_pool_2d_expr<T, Impl>; ///< The type of this expression

    /*!
     * \brief Compute the result type given the input type
     * \tparam A the input type
     */
    template <typename A>
    using result_type = dyn_matrix<value_t<A>, 2>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    const std::size_t c1; ///< First dimension pooling ratio
    const std::size_t c2; ///< Second dimension pooling ratio

    /*!
     * \brief Construct a new basic 2d pooling expression
     * \param c1 The first pooling factor
     * \param c2 The second pooling factor
     */
    basic_dyn_pool_2d_expr(std::size_t c1, std::size_t c2) : c1(c1), c2(c2) {
        // Nothing else to init
    }

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_2d needs 2D matrices");

        Impl::apply(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c),
            c1, c2);
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "pool_2d";
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The input expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A>
    std::size_t dim(const A& a, std::size_t d) const {
        if (d == 0) {
            return etl::dim<0>(a) / c1;
        } else {
            return etl::dim<1>(a) / c2;
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    std::size_t size(const A& a) const {
        return (etl::dim<0>(a) / c1) * (etl::dim<1>(a) / c2);
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A>
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
 * \brief Max Pooling 2D expression type
 */
template <typename T>
using dyn_max_pool_2d_expr = basic_dyn_pool_2d_expr<T, impl::max_pool_2d>;

/*!
 * \brief Average Pooling 2D expression type
 */
template <typename T>
using dyn_avg_pool_2d_expr = basic_dyn_pool_2d_expr<T, impl::avg_pool_2d>;

/*!
 * \brief Base class for all dynamic 3D pooling expressions
 */
template <typename T, typename Impl>
struct basic_dyn_pool_3d_expr : dyn_impl_expr<basic_dyn_pool_3d_expr<T, Impl>> {
    using value_type = T;                                       ///< The type of values of the expression
    using this_type  = basic_dyn_pool_3d_expr<T, Impl>; ///< The type of this expression

    /*!
     * \brief Compute the result type given the input type
     * \tparam A the input type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    const std::size_t c1; ///< First dimension pooling ratio
    const std::size_t c2; ///< Second dimension pooling ratio
    const std::size_t c3; ///< Third dimension pooling ratio

    /*!
     * \brief Construct a new basic 3d pooling expression
     * \param c1 The first pooling factor
     * \param c2 The second pooling factor
     * \param c3 The third pooling factor
     */
    basic_dyn_pool_3d_expr(std::size_t c1, std::size_t c2, std::size_t c3) : c1(c1), c2(c2), c3(c3) {
        // Nothing else to init
    }

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
        static_assert(all_etl_expr<A, C>::value, "pool_3d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_3d needs 3D matrices");

        Impl::apply(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c),
            c1, c2, c3);
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "pool_3d";
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The input expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A>
    std::size_t dim(const A& a, std::size_t d) const {
        if (d == 0) {
            return etl::dim<0>(a) / c1;
        } else if (d == 1) {
            return etl::dim<1>(a) / c2;
        } else {
            return etl::dim<2>(a) / c3;
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    std::size_t size(const A& a) const {
        return (etl::dim<0>(a) / c1) * (etl::dim<1>(a) / c2) * (etl::dim<2>(a) / c3);
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A>
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
 * \brief Max Pooling 3D expression type
 */
template <typename T>
using dyn_max_pool_3d_expr = basic_dyn_pool_3d_expr<T, impl::max_pool_3d>;

/*!
 * \brief Average Pooling 3D expression type
 */
template <typename T>
using dyn_avg_pool_3d_expr = basic_dyn_pool_3d_expr<T, impl::avg_pool_3d>;

} //end of namespace etl
