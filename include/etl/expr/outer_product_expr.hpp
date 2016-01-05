//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

#include "etl/expr/detail.hpp"

#include "etl/impl/outer_product.hpp"

#include "etl/temporary.hpp"

namespace etl {

template <typename T>
struct outer_product_expr : impl_expr<outer_product_expr<T>> {
    using value_type = T;
    using this_type  = outer_product_expr<T>;

    /*!
     * \brief The result type for given sub types
     * \tparam A The sub expression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, A, B>;

    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "Outer product only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 1 && decay_traits<B>::dimensions() == 1 && decay_traits<C>::dimensions() == 2, "Invalid dimensions for outer product");

        detail::outer_product_impl<A, B, C>::apply(std::forward<A>(a), std::forward<B>(b), std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "outer_product";
    }

    template <typename A, typename B>
    static std::size_t size(const A& a, const B& b) {
        return etl::dim<0>(a) * etl::dim<0>(b);
    }

    template <typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
        return d == 0 ? etl::dim<0>(a) : etl::dim<0>(b);
    }

    template <typename A, typename B>
    static constexpr std::size_t size() {
        return etl_traits<A>::template dim<0>() * etl_traits<B>::template dim<0>();
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A, typename B>
    static constexpr etl::order order() {
        return etl::order::RowMajor;
    }

    template <typename A, typename B, std::size_t D>
    static constexpr std::size_t dim() {
        return D == 0 ? decay_traits<A>::template dim<0>() : decay_traits<B>::template dim<0>();
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

} //end of namespace etl
