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

namespace detail {

template <typename T>
concept fast_matrix_impl = traits_detail::is_fast_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept custom_fast_matrix_impl = traits_detail::is_custom_fast_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept dyn_matrix_impl = traits_detail::is_dyn_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept gpu_dyn_matrix_impl = traits_detail::is_gpu_dyn_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept custom_dyn_matrix_impl = traits_detail::is_custom_dyn_matrix_impl<std::decay_t<T>>::value;

template <typename T>
concept sparse_matrix_impl = traits_detail::is_sparse_matrix_impl<std::decay_t<T>>::value;

} // end of namespace detail

template<typename T>
concept fast = decay_traits<T>::is_fast;

template<typename T>
concept dyn = !decay_traits<T>::is_fast;

template<typename T>
concept generator = etl_expr<T> && decay_traits<T>::is_generator;

// TODO: Maybe rename into a generic concept (not not-something)
template<typename T>
concept not_generator = etl_expr<T> && !decay_traits<T>::is_generator;

template<typename T>
concept dyn_expr = etl_expr<T> && dyn<T>;

template<typename T>
concept fast_expr = etl_expr<T> && fast<T>;

template <typename T>
concept etl_complex = cpp::specialization_of<etl::complex, T>;

template <typename T>
concept std_complex = cpp::specialization_of<std::complex, T>;

template <typename T>
concept complex_c = etl_complex<T> || std_complex<T>;

template<typename T>
concept etl_complex_expr = etl_expr<T> && complex_c<value_t<T>>;

template <typename T>
concept unary_expr_c = cpp::specialization_of<etl::unary_expr, T>;

template <typename T>
concept sub_view_c = traits_detail::is_sub_view<std::decay_t<T>>::value;

template <typename T>
concept slice_view_c = cpp::specialization_of<etl::slice_view, T>;

template <typename T>
concept dyn_matrix_view_c = traits_detail::is_dyn_matrix_view<T>::value;

template <typename T>
concept etl_value_class =
        detail::fast_matrix_impl<T> || detail::custom_fast_matrix_impl<T> || detail::dyn_matrix_impl<T> || detail::custom_dyn_matrix_impl<T> || detail::sparse_matrix_impl<T> || detail::gpu_dyn_matrix_impl<T>;

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
concept etl_4d_and_plus = etl_expr<T> && decay_traits<T>::dimensions() >= 4;

template <typename T>
concept etl_5d = etl_expr<T> && decay_traits<T>::dimensions() == 5;

template <typename T>
concept etl_5d_and_plus = etl_expr<T> && decay_traits<T>::dimensions() >= 5;

template <typename T>
concept deep_mat = etl_expr<T> && decay_traits<T>::dimensions() >= 3;

template <typename T>
concept mat_or_vec = etl_expr<T> && (etl_1d<T> || etl_2d<T>);

template <typename T>
concept matrix = etl_expr<T> && decay_traits<T>::dimensions() > 1;

template <typename T>
concept fast_matrix_c = fast<T> && matrix<T>;

template <typename T, typename E>
concept same_dimensions = etl_expr<T> && decay_traits<T>::dimensions() == decay_traits<E>::dimensions();

template <typename T>
concept fast_2d = fast<T> && etl_2d<T>;

template <typename T>
concept fast_3d = fast<T> && etl_3d<T>;

template <typename T>
concept fast_4d = fast<T> && etl_4d<T>;

template <typename T>
concept dyn_2d = dyn<T> && etl_2d<T>;

template <typename T>
concept dyn_3d = dyn<T> && etl_3d<T>;

template <typename T>
concept dyn_4d = dyn<T> && etl_4d<T>;

template <typename T>
concept dyn_matrix_c = dyn<T> && matrix<T>;

template <typename T, typename VT>
concept convertible_expr = etl_expr<T> && std::convertible_to<value_t<T>, VT>;

template <typename T>
concept optimized_expr_c = cpp::specialization_of<etl::optimized_expr, T>;

template <typename T>
concept serial_expr_c = cpp::specialization_of<etl::serial_expr, T>;

template <typename T>
concept selected_expr_c = traits_detail::is_selected_expr_impl<std::decay_t<T>>::value;

template <typename T>
concept parallel_expr_c = cpp::specialization_of<etl::parallel_expr, T>;

template <typename T>
concept timed_expr_c = cpp::specialization_of<etl::timed_expr, T>;

template <typename T>
concept wrapper_expr = optimized_expr_c<T> || selected_expr_c<T> || serial_expr_c<T> || parallel_expr_c<T> || timed_expr_c<T>;

template <typename T>
concept sub_capable = matrix<T> || generator<T>;

template <typename T>
concept single_precision = std::same_as<T, float>;

template <typename T>
concept etl_single_precision = etl_expr<T> && single_precision<value_t<T>>;

template <typename T>
concept double_precision = std::same_as<T, double>;

template <typename T>
concept etl_double_precision = etl_expr<T> && double_precision<value_t<T>>;

template <typename T>
concept etl_dma = etl_expr<T> && decay_traits<T>::is_direct;

template <typename T>
concept etl_dma_single_precision = etl_dma<T> && etl_single_precision<T>;

template <typename T>
concept etl_dma_double_precision = etl_dma<T> && etl_double_precision<T>;

template <typename T>
concept gpu_computable = etl_expr<T> && decay_traits<T>::gpu_computable;

template <typename T>
concept gpu_computable_single_precision = gpu_computable<T> && etl_single_precision<T>;

template <typename T>
concept gpu_computable_double_precision = gpu_computable<T> && etl_double_precision<T>;

// Complement the standard library

template<typename T>
concept arithmetic = std::floating_point<T> || std::integral<T>;

template<typename T>
concept size_c = std::convertible_to<T, size_t>;

template<typename T>
concept std_container = requires(T a) {
    requires !etl_expr<T>;
    typename T::value_type;
    { a.begin() };
    { a.end() };
};

} // namespace etl
