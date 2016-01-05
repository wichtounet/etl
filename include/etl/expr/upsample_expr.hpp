//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

#include "etl/expr/detail.hpp"

//Get the implementations
#include "etl/impl/pooling.hpp"

namespace etl {

template <typename T, std::size_t C1, std::size_t C2, template <typename...> class Impl>
struct basic_upsample_2d_expr : impl_expr<basic_upsample_2d_expr<T, C1, C2, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");

    using this_type  = basic_upsample_2d_expr<T, C1, C2, Impl>;
    using value_type = T;

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_2d needs 2D matrices");

        Impl<decltype(make_temporary(std::forward<A>(a))), C>::template apply<C1, C2>(
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
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

//Max Pool 2D

template <typename T, std::size_t C1, std::size_t C2>
using upsample_2d_expr = basic_upsample_2d_expr<T, C1, C2, impl::upsample_2d>;

template <typename T, std::size_t C1, std::size_t C2, std::size_t C3, template <typename...> class Impl>
struct basic_upsample_3d_expr  : impl_expr<basic_upsample_3d_expr<T, C1, C2, C3, Impl>>{
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");

    using this_type = basic_upsample_3d_expr<T, C1, C2, C3, Impl>;
    using value_type = T;

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_3d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_3d needs 3D matrices");

        Impl<decltype(make_temporary(std::forward<A>(a))), C>::template apply<C1, C2, C3>(
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

    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? decay_traits<A>::template dim<0>() * C1
                       : DD == 1 ? decay_traits<A>::template dim<1>() * C2
                                 : decay_traits<A>::template dim<2>() * C3;
    }

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

    template <typename A>
    static std::size_t size(const A& a) {
        return (etl::dim<0>(a) * C1) * (etl::dim<1>(a) * C2) * (etl::dim<2>(a) * C3);
    }

    template <typename A>
    static constexpr std::size_t size() {
        return this_type::template dim<A, 0>() * this_type::template dim<A, 1>() * this_type::template dim<A, 2>();
    }

    static constexpr std::size_t dimensions() {
        return 3;
    }
};

template <typename T, std::size_t C1, std::size_t C2, std::size_t C3>
using upsample_3d_expr = basic_upsample_3d_expr<T, C1, C2, C3, impl::upsample_3d>;

} //end of namespace etl
