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

template <typename T, std::size_t C1, std::size_t C2, template <typename...> class Impl>
struct basic_pool_derivative_2d_expr : impl_expr<basic_pool_derivative_2d_expr<T, C1, C2, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");

    using this_type = basic_pool_derivative_2d_expr<T, C1, C2, Impl>;
    using value_type = T;

    /*!
     * \brief The result type for given sub types
     * \tparam A The sub expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, A, B>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "pool_derivative_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_derivative_2d needs 2D matrices");

        Impl<A, B, C>::template apply<C1, C2>(
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

    template <typename A, typename B, std::size_t DD>
    static constexpr std::size_t dim() {
        return decay_traits<A>::template dim<DD>();
    }

    template <typename A, typename B>
    static std::size_t dim(const A& a, const B& /*b*/, std::size_t d) {
        return etl::dim(a, d);
    }

    template <typename A, typename B>
    static std::size_t size(const A& a, const B& /*b*/) {
        return etl::dim<0>(a) * etl::dim<1>(a);
    }

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

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

//Max Pool 2D

template <typename T, std::size_t C1, std::size_t C2>
using max_pool_derivative_2d_expr = basic_pool_derivative_2d_expr<T, C1, C2, impl::max_pool_derivative_2d>;

template <typename T, std::size_t C1, std::size_t C2, std::size_t C3, template <typename...> class Impl>
struct basic_pool_derivative_3d_expr : impl_expr<basic_pool_derivative_3d_expr<T, C1, C2, C3, Impl>> {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");

    using this_type = basic_pool_derivative_3d_expr<T, C1, C2, C3, Impl>;
    using value_type = T;

    /*!
     * \brief The result type for given sub types
     * \tparam A The sub expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, A, B>;

    static constexpr const bool is_gpu = false; ///< no GPU implementation

    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "pool_derivative_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_derivative_2d needs 2D matrices");

        Impl<A, B, C>::template apply<C1, C2, C3>(
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

    template <typename A, typename B, std::size_t DD>
    static constexpr std::size_t dim() {
        return decay_traits<A>::template dim<DD>();
    }

    template <typename A, typename B>
    static std::size_t dim(const A& a, const B& /*b*/, std::size_t d) {
        return etl::dim(a, d);
    }

    template <typename A, typename B>
    static std::size_t size(const A& a, const B& /*b*/) {
        return etl::dim<0>(a) * etl::dim<1>(a) * etl::dim<2>(a);
    }

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

    static constexpr std::size_t dimensions() {
        return 3;
    }
};

//Max pool_derivative 2D

template <typename T, std::size_t C1, std::size_t C2, std::size_t C3>
using max_pool_derivative_3d_expr = basic_pool_derivative_3d_expr<T, C1, C2, C3, impl::max_pool_derivative_3d>;

} //end of namespace etl
