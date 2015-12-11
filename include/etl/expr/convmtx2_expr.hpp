//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

//Get the implementations
#include "etl/impl/convmtx2.hpp"

namespace etl {

template <typename T, std::size_t K1, std::size_t K2, template <typename...> class Impl>
struct basic_convmtx2_expr {
    static_assert(K1 > 0, "K1 must be greater than 0");
    static_assert(K2 > 0, "K2 must be greater than 0");

    using this_type = basic_convmtx2_expr<T, K1, K2, Impl>;

    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? ((decay_traits<A>::template dim<0>() + K1 - 1) * (decay_traits<A>::template dim<1>() + K2 - 1))
                       : K1 * K2;
    }

    template <typename A, class Enable = void>
    struct result_type_builder {
        using type = dyn_matrix<value_t<A>, 2>;
    };

    template <typename A>
    struct result_type_builder<A, std::enable_if_t<all_fast<A>::value>> {
        using type = fast_dyn_matrix<value_t<A>, this_type::template dim<A, 0>(), this_type::template dim<A, 1>()>;
    };

    template <typename A>
    using result_type = typename result_type_builder<A>::type;

    template <typename A, cpp_enable_if(all_fast<A>::value)>
    static result_type<A>* allocate(A&& /*a*/) {
        return new result_type<A>();
    }

    template <typename A, cpp_disable_if(all_fast<A>::value)>
    static result_type<A>* allocate(A&& a) {
        auto c_height = (etl::dim<0>(a) + K1 - 1) * (etl::dim<1>(a) + K2 - 1);
        auto c_width = K1 * K2;
        return new result_type<A>(c_height, c_width);
    }

    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "convmtx2 only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "convmtx2 needs 2D matrices");

        Impl<decltype(make_temporary(std::forward<A>(a))), C>::template apply<K1, K2>(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    static std::string desc() noexcept {
        return "convmtx2";
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

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

//Direct convmtx2

template <typename T, std::size_t K1, std::size_t K2>
using direct_convmtx2_expr = basic_convmtx2_expr<T, K1, K2, detail::convmtx2_direct>;

} //end of namespace etl
