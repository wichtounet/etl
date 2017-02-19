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
 * \brief Upsample configurable expression, in two dimensions
 */
template <typename T, size_t D, std::size_t C1, std::size_t C2, typename Impl>
struct basic_upsample_2d_expr : impl_expr<basic_upsample_2d_expr<T, D, C1, C2, Impl>, T> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");

    using this_type  = basic_upsample_2d_expr<T, D, C1, C2, Impl>; ///< The type of expression
    using value_type = T;                                       ///< The value type

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, T, A>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "upsample_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "upsample_2d needs 2D matrices");

        Impl::template apply<C1, C2>(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "upsample_2d";
    }

    /*!
     * \brief Returns the DDth dimension of the expression
     * \tparam A The sub expression type
     * \tparam DD The dimension to get
     * \return the DDth dimension of the expression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == D - 2 ? decay_traits<A>::template dim<DD>() * C1
             : DD == D - 1 ? decay_traits<A>::template dim<DD>() * C2
                           : decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        if (d == D - 2) {
            return etl::dim(a, d) * C1;
        } else if (d == D - 1) {
            return etl::dim(a, d) * C2;
        } else {
            return etl::dim(a, d);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    static std::size_t size(const A& a) {
        return etl::size(a) * (C1 * C2);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A>
    static constexpr std::size_t size() {
        return decay_traits<A>::size() * (C1 * C2);
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
        return D;
    }
};

/*!
 * \brief Default Upsample 2D expression
 */
template <typename T, size_t D, std::size_t C1, std::size_t C2>
using upsample_2d_expr = basic_upsample_2d_expr<T, D, C1, C2, impl::upsample_2d>;

/*!
 * \brief Upsample configurable expression, in three dimensions
 */
template <typename T, size_t D, std::size_t C1, std::size_t C2, std::size_t C3, typename Impl>
struct basic_upsample_3d_expr : impl_expr<basic_upsample_3d_expr<T, D, C1, C2, C3, Impl>, T> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");

    using this_type  = basic_upsample_3d_expr<T, D, C1, C2, C3, Impl>; ///< The type of expression
    using value_type = T;                                           ///< The value type

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, T, A>;

    static constexpr bool is_gpu = false; ///< no GPU implementation

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "upsample_3d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() && decay_traits<C>::dimensions(), "upsample_3d needs 3D matrices");

        Impl::template apply<C1, C2, C3>(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "upsample_3d";
    }

    /*!
     * \brief Returns the DDth dimension of the expression
     * \tparam A The sub expression type
     * \tparam DD The dimension to get
     * \return the DDth dimension of the expression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == D - 3 ? decay_traits<A>::template dim<DD>() * C1
             : DD == D - 2 ? decay_traits<A>::template dim<DD>() * C2
             : DD == D - 1 ? decay_traits<A>::template dim<DD>() * C3
                           : decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        if (d == D - 3) {
            return etl::dim(a, d) * C1;
        } else if (d == D - 2) {
            return etl::dim(a, d) * C2;
        } else if (d == D - 1) {
            return etl::dim(a, d) * C3;
        } else {
            return etl::dim(a, d);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    static std::size_t size(const A& a) {
        return etl::size(a) * (C1 * C2 * C3);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A>
    static constexpr std::size_t size() {
        return decay_traits<A>::size() * (C1 * C2 * C3);
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
        return D;
    }
};

/*!
 * \brief Default Upsample dD expression
 */
template <typename T, size_t D, std::size_t C1, std::size_t C2, std::size_t C3>
using upsample_3d_expr = basic_upsample_3d_expr<T, D, C1, C2, C3, impl::upsample_3d>;

/*!
 * \brief Upsample configurable expression, in two dimensions
 */
template <typename T, size_t D, typename Impl>
struct basic_dyn_upsample_2d_expr : dyn_impl_expr<basic_dyn_upsample_2d_expr<T, D, Impl>, T> {
    using this_type  = basic_dyn_upsample_2d_expr<T, D, Impl>; ///< The type of expression
    using value_type = T;                                       ///< The value type

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, T, A>;

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
        static_assert(all_etl_expr<A, C>::value, "upsample_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "upsample_2d needs 2D matrices");

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
        return "upsample_2d";
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    std::size_t dim(const A& a, std::size_t d) const {
        if (d == D - 3) {
            return etl::dim(a, d) * c1;
        } else if (d == D - 2) {
            return etl::dim(a, d) * c2;
        } else {
            return etl::dim(a, d);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    std::size_t size(const A& a) const {
        return etl::size(a) * (c1 * c2);
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
        return D;
    }
};

/*!
 * \brief Default Upsample 2D expression
 */
template <typename T, size_t D>
using dyn_upsample_2d_expr = basic_dyn_upsample_2d_expr<T, D, impl::upsample_2d>;

/*!
 * \brief Upsample configurable expression, in three dimensions
 */
template <typename T, size_t D, typename Impl>
struct basic_dyn_upsample_3d_expr : dyn_impl_expr<basic_dyn_upsample_3d_expr<T, D, Impl>, T> {
    using this_type  = basic_dyn_upsample_3d_expr<T, D, Impl>; ///< The type of expression
    using value_type = T;                                           ///< The value type

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, T, A>;

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
        static_assert(all_etl_expr<A, C>::value, "upsample_3d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "upsample_3d needs 3D matrices");

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
        return "upsample_3d";
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    std::size_t dim(const A& a, std::size_t d) const {
        if (d == D - 3) {
            return etl::dim(a, d) * c1;
        } else if (d == D - 2) {
            return etl::dim(a, d) * c2;
        } else if (d == D - 1) {
            return etl::dim(a, d) * c3;
        } else {
            return etl::dim(a, d);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    std::size_t size(const A& a) const {
        return etl::size(a) * (c1 * c2 * c3);
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
        return D;
    }
};

/*!
 * \brief Default Upsample 3D expression
 */
template <typename T, size_t D>
using dyn_upsample_3d_expr = basic_dyn_upsample_3d_expr<T, D, impl::upsample_3d>;

} //end of namespace etl
