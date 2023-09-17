//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <concepts>
#include <type_traits>

namespace etl {

namespace concepts_detail {

} // end of namespace traits_detail

template<typename T>
concept etl_expr = etl::decay_traits<T>::is_etl;

template <typename T>
concept unary_expr_c = cpp::is_specialization_of_v<etl::unary_expr, std::decay_t<T>>;

template <typename T>
concept sub_view_c = traits_detail::is_sub_view<std::decay_t<T>>::value;

template <typename T>
concept slice_view_c = cpp::is_specialization_of_v<etl::slice_view, std::decay_t<T>>;

template <typename T>
concept dyn_matrix_view_c = traits_detail::is_dyn_matrix_view<T>::value;

template <typename T>
concept simple_lhs = is_etl_value_class<T> || unary_expr_c<T> || sub_view_c<T> || slice_view_c<T> || dyn_matrix_view_c<T>;

// Complement the standard library

template<typename T>
concept arithmetic = std::floating_point<T> || std::integral<T>;

} //end of namespace etl
