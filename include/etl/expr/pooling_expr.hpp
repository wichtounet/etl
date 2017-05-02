//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
template <typename T, size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename Impl>
struct basic_pool_2d_expr : impl_expr<basic_pool_2d_expr<T, C1, C2, S1, S2, P1, P2, Impl>, T> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(S1 > 0, "S1 must be greater than 0");
    static_assert(S2 > 0, "S2 must be greater than 0");

    using value_type = T;                                           ///< Type of values of the expression
    using this_type  = basic_pool_2d_expr<T, C1, C2, S1, S2, P1, P2, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_2d needs 2D matrices");

        Impl::template apply<C1, C2, S1, S2, P1, P2>(
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
    template <typename A, size_t DD>
    static constexpr size_t dim() {
        return DD == 0 ? (decay_traits<A>::template dim<0>() - C1 + 2 * P1) / S1 + 1
                       : (decay_traits<A>::template dim<1>() - C2 + 2 * P2) / S2 + 1;
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The input expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A>
    static size_t dim(const A& a, size_t d) {
        if (d == 0) {
            return (etl::dim<0>(a) - C1 + 2 * P1) / S1 + 1;
        } else {
            return (etl::dim<1>(a) - C2 + 2 * P1) / S2 + 1;
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    static size_t size(const A& a) {
        return this_type::dim(a, 0) * this_type::dim(a, 1);
    }

    /*!
     * \brief Return the size of the expression given the input expression type
     * \tparam A The in expression type
     * \return the size of the expression given the input type
     */
    template <typename A>
    static constexpr size_t size() {
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
    static constexpr size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Average Pooling 2D expression type
 */
template <typename T, size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2>
using avg_pool_2d_expr = basic_pool_2d_expr<T, C1, C2, S1, S2, P1, P2, impl::avg_pool_2d>;

/*!
 * \brief Base class for all 2D pooling expressions
 */
template <typename T, size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, size_t D, typename Impl>
struct basic_deep_pool_2d_expr : impl_expr<basic_deep_pool_2d_expr<T, C1, C2, S1, S2, P1, P2, D, Impl>, T> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(S1 > 0, "S1 must be greater than 0");
    static_assert(S2 > 0, "S2 must be greater than 0");

    using value_type = T;                                           ///< Type of values of the expression
    using this_type  = basic_deep_pool_2d_expr<T, C1, C2, S1, S2, P1, P2, D, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    static void apply(const A& a, C& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == D && decay_traits<C>::dimensions() == D, "pool_2d needs 2D matrices");

        Impl::template apply<C1, C2, S1, S2, P1, P2>(make_temporary(a), c);
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
    template <typename A, size_t DD>
    static constexpr size_t dim() {
        return DD == D - 2 ? (decay_traits<A>::template dim<DD>() - C1 + 2 * P1) / S1 + 1
             : DD == D - 1 ? (decay_traits<A>::template dim<DD>() - C2 + 2 * P2) / S2 + 1
                       : decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The input expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A>
    static size_t dim(const A& a, size_t d) {
        if (d == D - 2) {
            return (etl::dim(a, d) - C1 + 2 * P1) / S1 + 1;
        } else if (d == D - 1){
            return (etl::dim<1>(a, d) - C2 + 2 * P2) / S2 + 1;
        } else {
            return etl::dim(a, d);
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    static size_t size(const A& a) {
        size_t acc = 1;
        for (size_t i = 0; i < dimensions(); ++i) {
            acc *= this_type::dim(a, i);
        }
        return acc;
    }

    /*!
     * \brief Returns the multiplicative sum of the dimensions at the given indices
     * \return the multiplicative sum of the dimensions at the given indices
     */
    template <typename A, size_t... I>
    static constexpr size_t size_mul(const std::index_sequence<I...>& /*seq*/) {
        return mul_all<this_type::dim<A, I>()...>::value;
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A>
    static constexpr size_t size() {
        return size_mul<A>(std::make_index_sequence<dimensions()>());
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
    static constexpr size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Average Pooling 2D expression type
 */
template <typename T, size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, size_t D>
using deep_avg_pool_2d_expr = basic_deep_pool_2d_expr<T, C1, C2, S1, S2, P1, P2, D, impl::avg_pool_2d>;

/*!
 * \brief Base class for all 3D pooling expressions
 */
template <typename T, size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename Impl>
struct basic_pool_3d_expr : impl_expr<basic_pool_3d_expr<T, C1, C2, C3, S1, S2, S3, P1, P2, P3, Impl>, T> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");

    using value_type = T;                                                   ///< The type of values of the expression
    using this_type  = basic_pool_3d_expr<T, C1, C2, C3, S1, S2, S3, P1, P2, P3, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_3d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_3d needs 3D matrices");

        Impl::template apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(
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
    template <typename A, size_t DD>
    static constexpr size_t dim() {
        return DD == 0 ? (decay_traits<A>::template dim<0>() - C1 + 2 * P1) / S1 + 1
             : DD == 1 ? (decay_traits<A>::template dim<1>() - C2 + 2 * P2) / S2 + 1
                       : (decay_traits<A>::template dim<2>() - C3 + 2 * P3) / S3 + 1;
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The input expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A>
    static size_t dim(const A& a, size_t d) {
        if (d == 0) {
            return (etl::dim<0>(a) - C1 + 2 * P1) / S1 + 1;
        } else if (d == 1) {
            return (etl::dim<1>(a) - C2 + 2 * P2) / S2 + 1;
        } else {
            return (etl::dim<2>(a) - C3 + 2 * P3) / S3 + 1;
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    static size_t size(const A& a) {
        return this_type::dim(a, 0) * this_type::dim(a, 1) * this_type::dim(a, 2);
    }

    /*!
     * \brief Return the size of the expression given the input expression type
     * \tparam A The in expression type
     * \return the size of the expression given the input type
     */
    template <typename A>
    static constexpr size_t size() {
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
    static constexpr size_t dimensions() {
        return 3;
    }
};

/*!
 * \brief Max Pooling 3D expression type
 */
template <typename T, size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3>
using max_pool_3d_expr = basic_pool_3d_expr<T, C1, C2, C3, S1, S2, S3, P1, P2, P3, impl::max_pool_3d>;

/*!
 * \brief Average Pooling 3D expression type
 */
template <typename T, size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3>
using avg_pool_3d_expr = basic_pool_3d_expr<T, C1, C2, C3, S1, S2, S3, P1, P2, P3, impl::avg_pool_3d>;

/*!
 * \brief Base class for all deep 3D pooling expressions
 */
template <typename T, size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, size_t D, typename Impl>
struct basic_pool_deep_3d_expr : impl_expr<basic_pool_deep_3d_expr<T, C1, C2, C3, S1, S2, S3, P1, P2, P3, D, Impl>, T> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");
    static_assert(S1 > 0, "S1 must be greater than 0");
    static_assert(S2 > 0, "S2 must be greater than 0");
    static_assert(S3 > 0, "S3 must be greater than 0");

    using value_type = T;                                                   ///< The type of values of the expression
    using this_type  = basic_pool_deep_3d_expr<T, C1, C2, C3, S1, S2, S3, P1, P2, P3, D, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    static void apply(const A& a, C& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == D && decay_traits<C>::dimensions() == D, "pool_2d needs 2D matrices");

        Impl::template apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(make_temporary(a), c);
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
    template <typename A, size_t DD>
    static constexpr size_t dim() {
        return DD == D - 3 ? (decay_traits<A>::template dim<DD>() - C1 + 2 * P1) / S1 + 1
             : DD == D - 2 ? (decay_traits<A>::template dim<DD>() - C2 + 2 * P2) / S2 + 1
             : DD == D - 1 ? (decay_traits<A>::template dim<DD>() - C3 + 2 * P3) / S3 + 1
                       : decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Return the dth dim of the expression
     * \param a The input expression
     * \param d The dimension to get
     * \return The dth dimension
     */
    template <typename A>
    static size_t dim(const A& a, size_t d) {
        if (d == D - 3) {
            return (etl::dim(a, d) - C1 + 2 * P1) / S1 + 1;
        } else if (d == D - 2) {
            return (etl::dim(a, d) - C2 + 2 * P2) / S2 + 1;
        } else if(d == D - 1){
            return (etl::dim(a, d) - C3 + 2 * P3) / S3 + 1;
        } else {
            return etl::dim(a, d);
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    static size_t size(const A& a) {
        size_t acc = 1;
        for (size_t i = 0; i < dimensions(); ++i) {
            acc *= this_type::dim(a, i);
        }
        return acc;
    }

    /*!
     * \brief Returns the multiplicative sum of the dimensions at the given indices
     * \return the multiplicative sum of the dimensions at the given indices
     */
    template <typename A, size_t... I>
    static constexpr size_t size_mul(const std::index_sequence<I...>& /*seq*/) {
        return mul_all<this_type::dim<A, I>()...>::value;
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A>
    static constexpr size_t size() {
        return size_mul<A>(std::make_index_sequence<dimensions()>());
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
    static constexpr size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Max Pooling 3D expression type
 */
template <typename T, size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, size_t D>
using deep_max_pool_3d_expr = basic_pool_deep_3d_expr<T, C1, C2, C3, S1, S2, S3, P1, P2, P3, D, impl::max_pool_3d>;

/*!
 * \brief Average Pooling 3D expression type
 */
template <typename T, size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, size_t D>
using deep_avg_pool_3d_expr = basic_pool_deep_3d_expr<T, C1, C2, C3, S1, S2, S3, P1, P2, P3, D, impl::avg_pool_3d>;

/*!
 * \brief Base class for all dynamic 2D pooling expressions
 */
template <typename T, typename Impl>
struct basic_dyn_pool_2d_expr : dyn_impl_expr<basic_dyn_pool_2d_expr<T, Impl>, T> {
    using value_type = T;                               ///< Type of values of the expression
    using this_type  = basic_dyn_pool_2d_expr<T, Impl>; ///< The type of this expression

    /*!
     * \brief Compute the result type given the input type
     * \tparam A the input type
     */
    template <typename A>
    using result_type = dyn_matrix<value_t<A>, 2>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    const size_t c1; ///< First dimension pooling ratio
    const size_t c2; ///< Second dimension pooling ratio
    const size_t s1; ///< First dimension stride
    const size_t s2; ///< Second dimension stride
    const size_t p1; ///< First dimension padding
    const size_t p2; ///< Second dimension padding

    /*!
     * \brief Construct a new basic 2d pooling expression
     * \param c1 The first pooling factor
     * \param c2 The second pooling factor
     */
    basic_dyn_pool_2d_expr(size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) : c1(c1), c2(c2), s1(s1), s2(s2), p1(p1), p2(p2) {
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
            c1, c2, s1, s2, p1, p2);
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
    size_t dim(const A& a, size_t d) const {
        if (d == 0) {
            return (etl::dim<0>(a) - c1 + 2 * p1) / s1 + 1;
        } else {
            return (etl::dim<1>(a) - c2 + 2 * p2) / s2 + 1;
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    size_t size(const A& a) const {
        return dim(a, 0) * dim(a, 1);
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
    static constexpr size_t dimensions() {
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
 * \brief Base class for all dynamic 2D pooling expressions
 */
template <typename T, size_t D, typename Impl>
struct basic_dyn_deep_pool_2d_expr : dyn_impl_expr<basic_dyn_deep_pool_2d_expr<T, D, Impl>, T> {
    using value_type = T;                                       ///< Type of values of the expression
    using this_type  = basic_dyn_deep_pool_2d_expr<T, D, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = false; ///< no GPU implementation

    const size_t c1; ///< First dimension pooling ratio
    const size_t c2; ///< Second dimension pooling ratio
    const size_t s1; ///< First dimension stride
    const size_t s2; ///< Second dimension stride
    const size_t p1; ///< First dimension padding
    const size_t p2; ///< Second dimension padding

    /*!
     * \brief Construct a new basic 2d pooling expression
     * \param c1 The first pooling factor
     * \param c2 The second pooling factor
     */
    basic_dyn_deep_pool_2d_expr(size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) : c1(c1), c2(c2), s1(s1), s2(s2), p1(p1), p2(p2) {
        // Nothing else to init
    }

    /*!
     * \brief Apply the expression on a and store the result in c
     * \param a The input expression
     * \param c The output expression
     */
    template <typename A, typename C>
    void apply(const A& a, C& c) const {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == D && decay_traits<C>::dimensions() == D, "pool_2d needs 2D matrices");

        Impl::apply(make_temporary(a), c, c1, c2, s1, s2, p1, p2);
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
    size_t dim(const A& a, size_t d) const {
        if (d == D - 2) {
            return (etl::dim(a, d) - c1 + 2 * p1) / s1 + 1;
        } else if(d == D - 1){
            return (etl::dim(a, d) - c2 + 2 * p2) / s2 + 1;
        } else {
            return etl::dim(a, d);
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    size_t size(const A& a) const {
        size_t acc = 1;
        for (size_t i = 0; i < dimensions(); ++i) {
            acc *= this_type::dim(a, i);
        }
        return acc;
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
    static constexpr size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Max Pooling 2D expression type
 */
template <typename T, size_t D>
using dyn_deep_max_pool_2d_expr = basic_dyn_deep_pool_2d_expr<T, D, impl::max_pool_2d>;

/*!
 * \brief Average Pooling 2D expression type
 */
template <typename T, size_t D>
using dyn_deep_avg_pool_2d_expr = basic_dyn_deep_pool_2d_expr<T, D, impl::avg_pool_2d>;

/*!
 * \brief Base class for all dynamic 3D pooling expressions
 */
template <typename T, typename Impl>
struct basic_dyn_pool_3d_expr : dyn_impl_expr<basic_dyn_pool_3d_expr<T, Impl>, T> {
    using value_type = T;                                       ///< The type of values of the expression
    using this_type  = basic_dyn_pool_3d_expr<T, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = false; ///< no GPU implementation

    const size_t c1; ///< First dimension pooling ratio
    const size_t c2; ///< Second dimension pooling ratio
    const size_t c3; ///< Third dimension pooling ratio
    const size_t s1; ///< First dimension stride
    const size_t s2; ///< Second dimension stride
    const size_t s3; ///< Third dimension stride
    const size_t p1; ///< First dimension padding
    const size_t p2; ///< Second dimension padding
    const size_t p3; ///< Third dimension padding

    /*!
     * \brief Construct a new basic 3d pooling expression
     * \param c1 The first pooling factor
     * \param c2 The second pooling factor
     * \param c3 The third pooling factor
     */
    basic_dyn_pool_3d_expr(size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) : c1(c1), c2(c2), c3(c3), s1(s1), s2(s2), s3(s3), p1(p1), p2(p2), p3(p3) {
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
            c1, c2, c3, s1, s2, s3, p1, p2, p3);
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
    size_t dim(const A& a, size_t d) const {
        if (d == 0) {
            return (etl::dim<0>(a) - c1 + 2 * p1) / s1 + 1;
        } else if (d == 1) {
            return (etl::dim<1>(a) - c2 + 2 * p2) / s2 + 1;
        } else {
            return (etl::dim<2>(a) - c3 + 2 * p3) / s3 + 1;
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    size_t size(const A& a) const {
        return dim(a, 0) * dim(a, 1) * dim(a, 2);
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
    static constexpr size_t dimensions() {
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

/*!
 * \brief Base class for all dynamic 3D pooling expressions
 */
template <typename T, size_t D, typename Impl>
struct basic_dyn_deep_pool_3d_expr : dyn_impl_expr<basic_dyn_deep_pool_3d_expr<T, D, Impl>, T> {
    using value_type = T;                                       ///< The type of values of the expression
    using this_type  = basic_dyn_deep_pool_3d_expr<T, D, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = false; ///< no GPU implementation

    const size_t c1; ///< First dimension pooling ratio
    const size_t c2; ///< Second dimension pooling ratio
    const size_t c3; ///< Third dimension pooling ratio
    const size_t s1; ///< First dimension stride
    const size_t s2; ///< Second dimension stride
    const size_t s3; ///< Third dimension stride
    const size_t p1; ///< First dimension padding
    const size_t p2; ///< Second dimension padding
    const size_t p3; ///< Third dimension padding

    /*!
     * \brief Construct a new basic 3d pooling expression
     * \param c1 The first pooling factor
     * \param c2 The second pooling factor
     * \param c3 The third pooling factor
     */
    basic_dyn_deep_pool_3d_expr(size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) : c1(c1), c2(c2), c3(c3), s1(s1), s2(s2), s3(s3), p1(p1), p2(p2), p3(p3) {
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
        static_assert(decay_traits<A>::dimensions() == D && decay_traits<C>::dimensions() == D, "pool_3d needs 3D matrices");

        Impl::apply(make_temporary(a), c, c1, c2, c3, s1, s2, s3, p1, p2, p3);
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
    size_t dim(const A& a, size_t d) const {
        if (d == D - 3) {
            return (etl::dim(a, d) - c1 + 2 * p1) / s1 + 1;
        } else if (d == D - 2) {
            return (etl::dim(a, d) - c2 + 2 * p2) / s2 + 1;
        } else if(d == D - 1){
            return (etl::dim(a, d) - c3 + 2 * p3) / s3 + 1;
        } else {
            return etl::dim(a, d);
        }
    }

    /*!
     * \brief Return the size of the expression given the input expression
     * \param a The in expression
     * \return the size of the expression given the input
     */
    template <typename A>
    size_t size(const A& a) const {
        size_t acc = 1;
        for (size_t i = 0; i < dimensions(); ++i) {
            acc *= this_type::dim(a, i);
        }
        return acc;
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
    static constexpr size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Max Pooling 3D expression type
 */
template <typename T, size_t D>
using dyn_deep_max_pool_3d_expr = basic_dyn_deep_pool_3d_expr<T, D, impl::max_pool_3d>;

/*!
 * \brief Average Pooling 3D expression type
 */
template <typename T, size_t D>
using dyn_deep_avg_pool_3d_expr = basic_dyn_deep_pool_3d_expr<T, D, impl::avg_pool_3d>;

} //end of namespace etl
