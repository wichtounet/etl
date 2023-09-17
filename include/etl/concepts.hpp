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
concept etl_expr = decay_traits<T>::is_etl;

template <typename T>
concept etl_complex = cpp::is_specialization_of_v<etl::complex, std::decay_t<T>>;

template <typename T>
concept std_complex = cpp::is_specialization_of_v<std::complex, std::decay_t<T>>;

template <typename T>
concept complex_c = etl_complex<T> || std_complex<T>;

template<typename T>
concept etl_complex_expr = etl_expr<T> && complex_c<value_t<T>>;

template <typename T>
concept unary_expr_c = cpp::is_specialization_of_v<etl::unary_expr, std::decay_t<T>>;

template <typename T>
concept sub_view_c = traits_detail::is_sub_view<std::decay_t<T>>::value;

template <typename T>
concept slice_view_c = cpp::is_specialization_of_v<etl::slice_view, std::decay_t<T>>;

template <typename T>
concept dyn_matrix_view_c = traits_detail::is_dyn_matrix_view<T>::value;

template <typename T>
concept fast_matrix_c = traits_detail::is_fast_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept custom_fast_matrix_c = traits_detail::is_custom_fast_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept dyn_matrix_c = traits_detail::is_dyn_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept gpu_dyn_matrix_c = traits_detail::is_gpu_dyn_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept custom_dyn_matrix_c = traits_detail::is_custom_dyn_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept sparse_matrix_c = traits_detail::is_sparse_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept etl_value_class = fast_matrix_c<T> || custom_fast_matrix_c<T> || dyn_matrix_c<T> || custom_dyn_matrix_c<T> || sparse_matrix_c<T> || gpu_dyn_matrix_c<T>;

template <typename T>
concept simple_lhs = etl_value_class<T> || unary_expr_c<T> || sub_view_c<T> || slice_view_c<T> || dyn_matrix_view_c<T>;

template <typename T>
concept etl_1d = etl_expr<T> && decay_traits<T>::dimensions() == 1;

template <typename T>
concept etl_2d = etl_expr<T> && decay_traits<T>::dimensions() == 2;

template <typename T>
concept etl_3d = etl_expr<T> && decay_traits<T>::dimensions() == 3;

template <typename T>
concept etl_4d = etl_expr<T> && decay_traits<T>::dimensions() == 4;

template <typename T>
concept deep_mat = etl_expr<T> && decay_traits<T>::dimensions() >= 3;

template <typename T>
concept mat_or_vec = etl_expr<T> && (etl_1d<T> || etl_2d<T>);

template <typename T>
concept matrix = etl_expr<T> && decay_traits<T>::dimensions() > 1;

// Complement the standard library

template<typename T>
concept arithmetic = std::floating_point<T> || std::integral<T>;

} //end of namespace etl
