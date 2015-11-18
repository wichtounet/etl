//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the pooling expressions.
*/

#pragma once

#include <algorithm>

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "etl/traits_lite.hpp"

//Get the implementations
#include "etl/impl/pooling.hpp"

namespace etl {

/*!
 * \brief Base class for all 2D pooling expressions
 */
template <typename T, std::size_t C1, std::size_t C2, template <typename...> class Impl>
struct basic_pool_2d_expr {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");

    using this_type = basic_pool_2d_expr<T, C1, C2, Impl>;

    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? decay_traits<A>::template dim<0>() / C1
                       : decay_traits<A>::template dim<1>() / C2;
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
        return new result_type<A>(etl::dim<0>(a) / C2, etl::dim<1>(a) / C2);
    }

    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_2d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "pool_2d needs 2D matrices");

        Impl<decltype(make_temporary(std::forward<A>(a))), C>::template apply<C1, C2>(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    static std::string desc() noexcept {
        return "pool_2d";
    }

    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        if (d == 0) {
            return etl::dim<0>(a) / C1;
        } else {
            return etl::dim<1>(a) / C2;
        }
    }

    template <typename A>
    static std::size_t size(const A& a) {
        return (etl::dim<0>(a) / C1) * (etl::dim<1>(a) / C2);
    }

    template <typename A>
    static constexpr std::size_t size() {
        return this_type::template dim<A, 0>() * this_type::template dim<A, 1>();
    }

    static constexpr std::size_t dimensions() {
        return 2;
    }
};

//Max Pool 2D

template <typename T, std::size_t C1, std::size_t C2>
using max_pool_2d_expr = basic_pool_2d_expr<T, C1, C2, impl::max_pool_2d>;

template <typename T, std::size_t C1, std::size_t C2>
using avg_pool_2d_expr = basic_pool_2d_expr<T, C1, C2, impl::avg_pool_2d>;

/*!
 * \brief Base class for all 3D pooling expressions
 */
template <typename T, std::size_t C1, std::size_t C2, std::size_t C3, template <typename...> class Impl>
struct basic_pool_3d_expr {
    static_assert(C1 > 0, "C1 must be greater than 0");
    static_assert(C2 > 0, "C2 must be greater than 0");
    static_assert(C3 > 0, "C3 must be greater than 0");

    using this_type = basic_pool_3d_expr<T, C1, C2, C3, Impl>;

    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? decay_traits<A>::template dim<0>() / C1
                       : DD == 1 ? decay_traits<A>::template dim<1>() / C2
                                 : decay_traits<A>::template dim<2>() / C3;
    }

    template <typename A, class Enable = void>
    struct result_type_builder {
        using type = dyn_matrix<value_t<A>, 3>;
    };

    template <typename A>
    struct result_type_builder<A, std::enable_if_t<all_fast<A>::value>> {
        using type = fast_dyn_matrix<value_t<A>, this_type::template dim<A, 0>(), this_type::template dim<A, 1>(), this_type::template dim<A, 2>()>;
    };

    template <typename A>
    using result_type = typename result_type_builder<A>::type;

    template <typename A, cpp_enable_if(all_fast<A>::value)>
    static result_type<A>* allocate(A&& /*a*/) {
        return new result_type<A>();
    }

    template <typename A, cpp_disable_if(all_fast<A>::value)>
    static result_type<A>* allocate(A&& a) {
        return new result_type<A>(etl::dim<0>(a) / C2, etl::dim<1>(a) / C2, etl::dim<2>(a) / C3);
    }

    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "pool_3d only supported for ETL expressions");
        static_assert(decay_traits<A>::dimensions() == 3 && decay_traits<C>::dimensions() == 3, "pool_3d needs 3D matrices");

        Impl<decltype(make_temporary(std::forward<A>(a))), C>::template apply<C1, C2, C3>(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    static std::string desc() noexcept {
        return "pool_3d";
    }

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

    template <typename A>
    static std::size_t size(const A& a) {
        return (etl::dim<0>(a) / C1) * (etl::dim<1>(a) / C2) * (etl::dim<2>(a) / C3);
    }

    template <typename A>
    static constexpr std::size_t size() {
        return this_type::template dim<A, 0>() * this_type::template dim<A, 1>() * this_type::template dim<A, 2>();
    }

    static constexpr std::size_t dimensions() {
        return 3;
    }
};

//Max Pool 2D

template <typename T, std::size_t C1, std::size_t C2, std::size_t C3>
using max_pool_3d_expr = basic_pool_3d_expr<T, C1, C2, C3, impl::max_pool_3d>;

template <typename T, std::size_t C1, std::size_t C2, std::size_t C3>
using avg_pool_3d_expr = basic_pool_3d_expr<T, C1, C2, C3, impl::avg_pool_3d>;

} //end of namespace etl
