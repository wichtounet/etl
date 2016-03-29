//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains some TMP utilities for expressions
*/

#pragma once

#include "etl/temporary.hpp"

namespace etl {

namespace detail {

template <typename E, typename I, typename... Subs>
struct fast_result_type_builder;

template <typename E, std::size_t... I, typename... Subs>
struct fast_result_type_builder<E, std::index_sequence<I...>, Subs...> {
    using value_type = typename E::value_type;
    using type       = fast_matrix_impl<value_type, std::vector<value_type>, E::template order<Subs...>(), E::template dim<Subs..., I>()...>;
};

template <typename E, bool Fast, typename... Subs>
struct expr_result;

template <typename E, typename... Subs>
struct expr_result<E, false, Subs...> {
    using type = dyn_matrix_impl<typename E::value_type, E::template order<Subs...>(), E::dimensions()>;
};

template <typename E, typename... Subs>
struct expr_result<E, true, Subs...> {
    using type = typename fast_result_type_builder<E, std::make_index_sequence<E::dimensions()>, Subs...>::type;
};

template <typename E, typename... Subs>
using expr_result_t = typename expr_result<E, all_fast<Subs...>::value, Subs...>::type;

} // end of namespace detail

template <typename D>
struct impl_expr {
    using derived_t = D;

    template <typename... Subs>
    using result_type = detail::expr_result_t<derived_t, Subs...>;

    /*!
     * \brief Allocate the temporary for the expression
     * \param args The sub expressions
     * \return a pointer to the temporary
     */
    template <typename... Subs, cpp_enable_if(all_fast<Subs...>::value)>
    static result_type<Subs...>* allocate(__attribute__((unused)) Subs&&... args) {
        return new result_type<Subs...>();
    }

    template <typename... Subs, std::size_t... I>
    static result_type<Subs...>* dyn_allocate(std::index_sequence<I...> /*seq*/, Subs&&... subs) {
        return new result_type<Subs...>(derived_t::dim(subs..., I)...);
    }

    /*!
     * \brief Allocate the temporary for the expression
     * \param args The sub expressions
     * \return a pointer to the temporary
     */
    template <typename... Subs, cpp_disable_if(all_fast<Subs...>::value)>
    static result_type<Subs...>* allocate(Subs&&... args) {
        return dyn_allocate(std::make_index_sequence<derived_t::dimensions()>(), std::forward<Subs>(args)...);
    }
};

} //end of namespace etl
