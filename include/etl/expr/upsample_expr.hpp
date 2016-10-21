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
 * \brief Upsample configurable expression, in two dimensions
 */
template <typename T, std::size_t C1, std::size_t C2, typename Impl>
struct basic_upsample_2d_expr : impl_expr<basic_upsample_2d_expr<T, C1, C2, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");

    using this_type  = basic_upsample_2d_expr<T, C1, C2, Impl>; ///< The type of expression
    using value_type = T;                                       ///< The value type

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_2d needs 2D matrices");

        Impl::template apply<C1, C2>(
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
     * \brief Returns the DDth dimension of the expression
     * \tparam A The sub expression type
     * \tparam DD The dimension to get
     * \return the DDth dimension of the expression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? decay_traits<A>::template dim<0>() * C1
                       : decay_traits<A>::template dim<1>() * C2;
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        if (d == 0) {
            return etl::dim<0>(a) * C1;
        } else {
            return etl::dim<1>(a) * C2;
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    static std::size_t size(const A& a) {
        return (etl::dim<0>(a) * C1) * (etl::dim<1>(a) * C2);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
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
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

//Max Pool 2D

/*!
 * \brief Default Upsample 2D expression
 */
template <typename T, std::size_t C1, std::size_t C2>
using upsample_2d_expr = basic_upsample_2d_expr<T, C1, C2, impl::upsample_2d>;

/*!
 * \brief Upsample configurable expression, in three dimensions
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t C3, typename Impl>
struct basic_upsample_3d_expr : impl_expr<basic_upsample_3d_expr<T, C1, C2, C3, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");

    using this_type  = basic_upsample_3d_expr<T, C1, C2, C3, Impl>; ///< The type of expression
    using value_type = T;                                           ///< The value type

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
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
     * \brief Returns the DDth dimension of the expression
     * \tparam A The sub expression type
     * \tparam DD The dimension to get
     * \return the DDth dimension of the expression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? decay_traits<A>::template dim<0>() * C1
                       : DD == 1 ? decay_traits<A>::template dim<1>() * C2
                                 : decay_traits<A>::template dim<2>() * C3;
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        if (d == 0) {
            return etl::dim<0>(a) * C1;
        } else if (d == 1) {
            return etl::dim<1>(a) * C2;
        } else {
            return etl::dim<2>(a) * C3;
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    static std::size_t size(const A& a) {
        return (etl::dim<0>(a) * C1) * (etl::dim<1>(a) * C2) * (etl::dim<2>(a) * C3);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
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
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 3;
    }
};

/*!
 * \brief Default Upsample dD expression
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t C3>
using upsample_3d_expr = basic_upsample_3d_expr<T, C1, C2, C3, impl::upsample_3d>;

/*!
 * \brief Upsample configurable expression, in two dimensions
 */
template <typename T, typename Impl>
struct basic_dyn_upsample_2d_expr : dyn_impl_expr<basic_dyn_upsample_2d_expr<T, Impl>> {
    using this_type  = basic_dyn_upsample_2d_expr<T, Impl>; ///< The type of expression
    using value_type = T;                                       ///< The value type

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    std::size_t c1; ///< The first dimension upsampling factor
    std::size_t c2; ///< The second dimension upsampling factor

    /*!
     * \brief Create a new expression
     * \param c1 The first upsampling factor
     * \param c2 The second upsampling factor
     */
    basic_dyn_upsample_2d_expr(std::size_t c1, std::size_t c2) : c1(c1), c2(c2) {
        // Nothing else to init
    }

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_2d needs 2D matrices");

        Impl::template apply(
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
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    std::size_t dim(const A& a, std::size_t d) const {
        if (d == 0) {
            return etl::dim<0>(a) * c1;
        } else {
            return etl::dim<1>(a) * c2;
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    std::size_t size(const A& a) const {
        return (etl::dim<0>(a) * c1) * (etl::dim<1>(a) * c2);
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
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief Default Upsample 2D expression
 */
template <typename T>
using dyn_upsample_2d_expr = basic_dyn_upsample_2d_expr<T, impl::upsample_2d>;

/*!
 * \brief Upsample configurable expression, in three dimensions
 */
template <typename T, typename Impl>
struct basic_dyn_upsample_3d_expr : dyn_impl_expr<basic_dyn_upsample_3d_expr<T, Impl>> {
    using this_type  = basic_dyn_upsample_3d_expr<T, Impl>; ///< The type of expression
    using value_type = T;                                           ///< The value type

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    std::size_t c1; ///< The first dimension upsampling factor
    std::size_t c2; ///< The second dimension upsampling factor
    std::size_t c3; ///< The third dimension upsampling factor

    /*!
     * \brief Create a new expression
     * \param c1 The first upsampling factor
     * \param c2 The second upsampling factor
     * \param c3 The third upsampling factor
     */
    basic_dyn_upsample_3d_expr(std::size_t c1, std::size_t c2, std::size_t c3) : c1(c1), c2(c2), c3(c3) {
        // Nothing else to init
    }

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
        static_assert(all_etl_expr<A, C>::value, "pool_3d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_3d needs 3D matrices");

        Impl::template apply(
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
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    std::size_t dim(const A& a, std::size_t d) const {
        if (d == 0) {
            return etl::dim<0>(a) * c1;
        } else if (d == 1) {
            return etl::dim<1>(a) * c2;
        } else {
            return etl::dim<2>(a) * c3;
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    std::size_t size(const A& a) const {
        return (etl::dim<0>(a) * c1) * (etl::dim<1>(a) * c2) * (etl::dim<2>(a) * c3);
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
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 3;
    }
};

/*!
 * \brief Default Upsample dD expression
 */
template <typename T>
using dyn_upsample_3d_expr = basic_dyn_upsample_3d_expr<T, impl::upsample_3d>;

} //end of namespace etl
