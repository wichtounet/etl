//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "etl/expr/detail.hpp"

//Get the implementations
#include "etl/impl/convmtx2.hpp"

namespace etl {

template <typename T, std::size_t K1, std::size_t K2, template <typename...> class Impl>
struct basic_convmtx2_expr : impl_expr<basic_convmtx2_expr<T, K1, K2, Impl>> {
    static_assert(K1 > 0, "K1 must be greater than 0");
    static_assert(K2 > 0, "K2 must be greater than 0");

    using value_type = T;
    using this_type  = basic_convmtx2_expr<T, K1, K2, Impl>;

    /*!
     * \brief The result type for given sub types
     * \tparam A The sub expression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "convmtx2 only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "convmtx2 needs 2D matrices");

        Impl<decltype(make_temporary(std::forward<A>(a))), C>::template apply<K1, K2>(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "convmtx2";
    }

    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? ((decay_traits<A>::template dim<0>() + K1 - 1) * (decay_traits<A>::template dim<1>() + K2 - 1))
                       : K1 * K2;
    }

    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        if (d == 0) {
            return (etl::dim<0>(a) + K1 - 1) * (etl::dim<1>(a) + K2 - 1);
        } else {
            return K1 * K2;
        }
    }

    template <typename A>
    static std::size_t size(const A& a) {
        return (K1 * K2) * ((etl::dim<0>(a) + K1 - 1) * (etl::dim<1>(a) + K2 - 1));
    }

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

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

//Direct convmtx2

template <typename T, std::size_t K1, std::size_t K2>
using direct_convmtx2_expr = basic_convmtx2_expr<T, K1, K2, detail::convmtx2_direct>;

} //end of namespace etl
