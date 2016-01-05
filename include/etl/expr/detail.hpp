//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains some TMP utilities for expressions
*/

#pragma once

#include "cpp_utils/assert.hpp"
#include "cpp_utils/tmp.hpp"

#include "etl/temporary.hpp"

namespace etl {

namespace detail {

template <typename E, typename I, typename... Subs>
struct fast_result_type_builder;

template <typename E, std::size_t... I, typename... Subs>
struct fast_result_type_builder<E, std::index_sequence<I...>, Subs...> {
    using type = fast_dyn_matrix<typename E::value_type, E::template dim<Subs..., I>()...>;
};

template <typename E, bool Fast, typename... Subs>
struct expr_result;

template <typename E, typename... Subs>
struct expr_result<E, false, Subs...> {
    using type = dyn_matrix<typename E::value_type, E::dimensions()>;
};

template <typename E, typename... Subs>
struct expr_result<E, true, Subs...> {
    using type = typename fast_result_type_builder<E, std::make_index_sequence<E::dimensions()>, Subs...>::type;
};

template <typename E, typename... Subs>
using expr_result_t = typename expr_result<E, all_fast<Subs...>::value, Subs...>::type;

} // end of namespace detail

} //end of namespace etl
