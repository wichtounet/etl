//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace traits_detail {

/*!
 * \brief Special traits helper to detect if type is a fast_matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_fast_matrix_impl : std::false_type {};

/*!
 * \copydoc is_fast_matrix_impl
 */
template <typename V1, typename V2, order V3, size_t... R>
struct is_fast_matrix_impl<fast_matrix_impl<V1, V2, V3, R...>> : std::true_type {};

/*!
 * \brief Special traits helper to detect if type is a custom_fast_matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_custom_fast_matrix_impl : std::false_type {};

/*!
 * \copydoc is_custom_fast_matrix_impl
 */
template <typename V1, typename V2, order V3, size_t... R>
struct is_custom_fast_matrix_impl<custom_fast_matrix_impl<V1, V2, V3, R...>> : std::true_type {};

/*!
 * \brief Special traits helper to detect if type is a dyn_matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_dyn_matrix_impl : std::false_type {};

/*!
 * \copydoc is_dyn_matrix_impl
 */
template <typename V1, order V2, size_t V3>
struct is_dyn_matrix_impl<dyn_matrix_impl<V1, V2, V3>> : std::true_type {};

/*!
 * \brief Special traits helper to detect if type is a gpu_dyn_matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_gpu_dyn_matrix_impl : std::false_type {};

/*!
 * \copydoc is_gpu_dyn_matrix_impl
 */
template <typename V1, order V2, size_t V3>
struct is_gpu_dyn_matrix_impl<gpu_dyn_matrix_impl<V1, V2, V3>> : std::true_type {};

/*!
 * \brief Special traits helper to detect if type is a custom_dyn_matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_custom_dyn_matrix_impl : std::false_type {};

/*!
 * \copydoc is_custom_dyn_matrix_impl
 */
template <typename V1, order V2, size_t V3>
struct is_custom_dyn_matrix_impl<custom_dyn_matrix_impl<V1, V2, V3>> : std::true_type {};

/*!
 * \brief Special traits helper to detect if type is a sparse_matrix
 * \tparam T The type to test
 */
template <typename T>
struct is_sparse_matrix_impl : std::false_type {};

/*!
 * \copydoc is_sparse_matrix_impl
 */
template <typename V1, sparse_storage V2, size_t V3>
struct is_sparse_matrix_impl<sparse_matrix_impl<V1, V2, V3>> : std::true_type {};

/*!
 * \brief Special traits helper to detect if type is a dyn_matrix_view
 * \tparam T The type to test
 */
template <typename T>
struct is_dyn_matrix_view : std::false_type {};

/*!
 * \copydoc is_dyn_matrix_view
 */
template <typename E, size_t D, typename Enable>
struct is_dyn_matrix_view<dyn_matrix_view<E, D, Enable>> : std::true_type {};

/*!
 * \brief Special traits helper to detect if type is a sub_view
 * \tparam T The type to test
 */
template <typename T>
struct is_sub_view : std::false_type {};

/*!
 * \copydoc is_sub_view
 */
template <typename E, bool Aligned>
struct is_sub_view<sub_view<E, Aligned>> : std::true_type {};

/*!
 * \brief Special traits helper to detect if type is a selected_expr
 * \tparam T The type to test
 */
template <typename T>
struct is_selected_expr_impl : std::false_type {};

/*!
 * \copydoc is_selected_expr_impl
 */
template <typename Selector, Selector V, typename Expr>
struct is_selected_expr_impl<selected_expr<Selector, V, Expr>> : std::true_type {};

/*!
 * \brief Implementation of is_base_of_template_tb
 */
template <template <typename, bool> typename BTE, typename T, bool B>
std::true_type is_base_of_template_tb_impl(const BTE<T, B>*);

/*!
 * \brief Implementation of is_base_of_template_tb
 */
template <template <typename, bool> typename BTE>
std::false_type is_base_of_template_tb_impl(...);

/*!
 * \brief Traits to test if a type if inheriting from a given template.
 */
template <typename T, template <typename, bool> typename C>
constexpr bool is_base_of_template_tb = decltype(is_base_of_template_tb_impl<C>(std::declval<T*>()))::value;

/*!
 * \brief Helper traits to test if E is a non-GPU temporary expression.
 */
template <typename E, typename Enable = void>
struct is_nongpu_temporary_impl : std::false_type {};

/*!
 * \brief Helper traits to test if E is a non-GPU temporary expression.
 */
template <typename E>
struct is_nongpu_temporary_impl<E, std::enable_if_t<is_base_of_template_tb<std::decay_t<E>, etl::base_temporary_expr> && !std::decay_t<E>::gpu_computable>>
        : std::true_type {};

/*!
 * \brief Helper traits to test if E is a GPU temporary expression.
 */
template <typename E, typename Enable = void>
struct is_gpu_temporary_impl : std::false_type {};

/*!
 * \brief Helper traits to test if E is a GPU temporary expression.
 */
template <typename E>
struct is_gpu_temporary_impl<E, std::enable_if_t<is_base_of_template_tb<std::decay_t<E>, etl::base_temporary_expr> && std::decay_t<E>::gpu_computable>>
        : std::true_type {};

} // end of namespace traits_detail

/*!
 * \brief Traits to get information about ETL types
 *
 * For non-ETL types, is_etl is false and in that case, no other fields should be used on the traits.
 *
 * \tparam T the type to introspect
 */
template <typename T, typename Enable>
struct etl_traits {
    static constexpr bool is_etl         = false; ///< Indicates if T is an ETL type
    static constexpr bool is_transformer = false; ///< Indicates if T is a transformer
    static constexpr bool is_view        = false; ///< Indicates if T is a view
    static constexpr bool is_magic_view  = false; ///< Indicates if T is a magic view
    static constexpr bool is_fast        = false; ///< Indicates if T is a fast structure
    static constexpr bool is_generator   = false; ///< Indicates if T is a generator expression

    /*!
     * \brief Return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 0;
    }
};

/*!
 * \brief Traits indicating if the given ETL type is a fast matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_fast_matrix = traits_detail::is_fast_matrix_impl<std::decay_t<T>>::value;

/*!
 * \brief Traits indicating if the given ETL type is a fast matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_custom_fast_matrix = traits_detail::is_custom_fast_matrix_impl<std::decay_t<T>>::value;

/*!
 * \brief Traits indicating if the given ETL type is a dyn matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_dyn_matrix = traits_detail::is_dyn_matrix_impl<std::decay_t<T>>::value;

/*!
 * \brief Traits indicating if the given ETL type is a GPU dyn matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_gpu_dyn_matrix = traits_detail::is_gpu_dyn_matrix_impl<std::decay_t<T>>::value;

/*!
 * \brief Traits indicating if the given ETL type is a custom dyn matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_custom_dyn_matrix = traits_detail::is_custom_dyn_matrix_impl<std::decay_t<T>>::value;

/*!
 * \brief Traits indicating if the given ETL type is a sparse matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_sparse_matrix = traits_detail::is_sparse_matrix_impl<std::decay_t<T>>::value;

/*!
 * \brief Traits indicating if the given ETL type is a symmetric matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_symmetric_matrix = cpp::is_specialization_of_v<etl::symmetric_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a hermitian matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_hermitian_matrix = cpp::is_specialization_of_v<etl::hermitian_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a diagonal matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_diagonal_matrix = cpp::is_specialization_of_v<etl::diagonal_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is an upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_upper_matrix = cpp::is_specialization_of_v<etl::upper_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_lower_matrix = cpp::is_specialization_of_v<etl::lower_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a strictly lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_strictly_lower_matrix = cpp::is_specialization_of_v<etl::strictly_lower_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a strictly upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_strictly_upper_matrix = cpp::is_specialization_of_v<etl::strictly_upper_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a uni lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_uni_lower_matrix = cpp::is_specialization_of_v<etl::uni_lower_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a uni upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_uni_upper_matrix = cpp::is_specialization_of_v<etl::uni_upper_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a unary expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_unary_expr = cpp::is_specialization_of_v<etl::unary_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a binary expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_binary_expr = cpp::is_specialization_of_v<etl::binary_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a generator expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_generator_expr = cpp::is_specialization_of_v<etl::generator_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is an optimized expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_optimized_expr = cpp::is_specialization_of_v<etl::optimized_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a serial expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_serial_expr = cpp::is_specialization_of_v<etl::serial_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a selector expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_selected_expr = traits_detail::is_selected_expr_impl<std::decay_t<T>>::value;

/*!
 * \brief Traits indicating if the given ETL type is a parallel expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_parallel_expr = cpp::is_specialization_of_v<etl::parallel_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a timed expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_timed_expr = cpp::is_specialization_of_v<etl::timed_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a wrapper expression (optimized, serial, ...).
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_wrapper_expr = is_optimized_expr<T> || is_selected_expr<T> || is_serial_expr<T> || is_parallel_expr<T> || is_timed_expr<T>;

/*!
 * \brief Traits to test if the given expression is a sub_view
 */
template <typename T>
constexpr bool is_sub_view = traits_detail::is_sub_view<std::decay_t<T>>::value;

/*!
 * \brief Traits to test if the given expression is a slice_view
 */
template <typename T>
constexpr bool is_slice_view = cpp::is_specialization_of_v<etl::slice_view, std::decay_t<T>>;

/*!
 * \brief Traits to test if the given expression is a dyn_matrix_view
 */
template <typename T>
constexpr bool is_dyn_matrix_view = traits_detail::is_dyn_matrix_view<T>::value;

/*!
 * \brief Traits indicating if the given ETL type is a transformer expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_transformer = decay_traits<T>::is_transformer;

/*!
 * \brief Traits indicating if the given ETL type is a view expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_view = decay_traits<T>::is_view;

/*!
 * \brief Traits indicating if the given ETL type is a magic view expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_magic_view = decay_traits<T>::is_magic_view;

/*!
 * \brief Traits indicating if the given type is an ETL type.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_etl_expr = decay_traits<T>::is_etl;

/*!
 * \brief Traits indicating if the given type is a transpose expr.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_transpose_expr = cpp::is_specialization_of_v<etl::transpose_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given type is a temporary expression.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_temporary_expr = traits_detail::is_base_of_template_tb<std::decay_t<T>, etl::base_temporary_expr>;

/*!
 * \brief Traits indicating if the given ETL type is a value type.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_etl_value = decay_traits<T>::is_value;

/*!
 * \brief Traits indicating if the given ETL type is from a value class.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_etl_value_class =
    is_fast_matrix<T> || is_custom_fast_matrix<T> || is_dyn_matrix<T> || is_custom_dyn_matrix<T> || is_sparse_matrix<T> || is_gpu_dyn_matrix<T>;

/*!
 * \brief Traits indicating if the given ETL type can be left hand side type
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_lhs = is_etl_value<T> || is_unary_expr<T>;

/*!
 * \brief Traits indicating if the given ETL type is a simple left hand side type.
 * Adapter types are not taken from this because they do more operations.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_simple_lhs = is_etl_value_class<T> || is_unary_expr<T> || is_sub_view<T> || is_slice_view<T> || is_dyn_matrix_view<T>;

/*!
 * \brief Traits to test if a type is a scalar type
 * \tparam T The type to test.
 */
template <typename T>
constexpr bool is_scalar = cpp::is_specialization_of_v<etl::scalar, std::decay_t<T>>;

/*!
 * \brief Traits to test if the given type is single precision type.
 * \tparam T The type
 */
template <typename T>
constexpr bool is_single_precision_t = std::is_same_v<T, float>;

/*!
 * \brief Traits to test if the given ETL expresion contains single precision numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_single_precision = is_single_precision_t<value_t<T>>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains single precision numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_single_precision = (is_single_precision<E> && ...);

/*!
 * \brief Traits to test if the given type is double precision type.
 * \tparam T The type
 */
template <typename T>
constexpr bool is_double_precision_t = std::is_same_v<T, double>;

/*!
 * \brief Traits to test if the type is boolean.
 * \tparam T The type.
 */
template <typename T>
constexpr bool is_bool_t = std::is_same_v<T, bool>;

/*!
 * \brief Traits to test if the given ETL expresion contains double precision numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_double_precision = is_double_precision_t<value_t<T>>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains double precision numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_double_precision = (is_double_precision<E> && ...);

/*!
 * \brief Traits to test if the given ETL expresion contains floating point numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_floating = is_single_precision<T> || is_double_precision<T>;

/*!
 * \brief Traits to test if the type is floating point numbers.
 * \tparam T The type.
 */
template <typename T>
constexpr bool is_floating_t = is_single_precision_t<T> || is_double_precision_t<T>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains floating point numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_floating = (is_floating<E> && ...);

/*!
 * \brief Traits to test if all the given types are floating point numbers.
 * \tparam E The types.
 */
template <typename... E>
constexpr bool all_floating_t = (is_floating_t<E> && ...);

/*!
 * \brief Traits to test if a type is a complex number type
 * \tparam T The type to test.
 */
template <typename T>
constexpr bool is_complex_t = cpp::is_specialization_of_v<std::complex, std::decay_t<T>> || cpp::is_specialization_of_v<etl::complex, std::decay_t<T>>;

/*!
 * \brief Traits to test if a type is a single precision complex number type
 * \tparam T The type to test.
 */
template <typename T>
constexpr bool is_complex_single_t = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, etl::complex<float>>;

/*!
 * \brief Traits to test if a type is a double precision complex number type
 * \tparam T The type to test.
 */
template <typename T>
constexpr bool is_complex_double_t = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, etl::complex<double>>;

/*!
 * \brief Traits to test if the given ETL expresion type contains single precision complex numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_complex_single_precision = is_complex_single_t<value_t<T>>;

/*!
 * \brief Traits to test if the given ETL expresion type contains double precision complex numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_complex_double_precision = is_complex_double_t<value_t<T>>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains single precision complex numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_complex_single_precision = (is_complex_single_precision<E> && ...);

/*!
 * \brief Traits to test if all the given ETL expresion types contains double precision complex numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_complex_double_precision = (is_complex_double_precision<E> && ...);

/*!
 * \brief Traits to test if the given ETL expresion type contains complex numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_complex = is_complex_single_precision<T> || is_complex_double_precision<T>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains complex numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_complex = (is_complex<E> && ...);

/*!
 * \brief Traits to test if the given ETL expresion type contains
 * single precision floating point or single precision complex numbers.
 *
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_deep_single_precision = is_complex_single_precision<T> || is_single_precision<T>;

/*!
 * \brief Traits to test if the given ETL expresion type contains
 * double precision floating point or double precision complex numbers.
 *
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_deep_double_precision = is_complex_double_precision<T> || is_double_precision<T>;

/*!
 * \brief Traits to test if the given type contains a type that can
 * be computed on a GPU.
 *
 * Currently, GPU on ETL only supports floating points and complex
 * numbers.
 *
 * \tparam T The type.
 */
template <typename T>
constexpr bool is_gpu_t = is_floating_t<T> || is_complex_t<T> || is_bool_t<T>;

/*!
 * \brief Traits indicating if the given ETL type has direct memory access.
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_dma = decay_traits<T>::is_direct;

/*!
 * \brief Traits to test if all the given ETL expresion types have direct memory access (DMA).
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_dma = (is_dma<E> && ...);

/*!
 * \brief Traits to test if all the given ETL expresion types are row-major.
 * \tparam E The ETL expression types.
 */
template <typename E>
constexpr bool is_row_major = decay_traits<E>::storage_order == order::RowMajor;

/*!
 * \brief Traits to test if all the given ETL expresion types are row-major.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_row_major = (is_row_major<E> & ...);

/*!
 * \brief Traits to test if all the given ETL expresion types are column-major.
 * \tparam E The ETL expression types.
 */
template <typename E>
constexpr bool is_column_major = decay_traits<E>::storage_order == order::ColumnMajor;

/*!
 * \brief Traits to test if all the given ETL expresion types are column-major.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_column_major = (is_column_major<E> && ...);

/*!
 * \brief Traits to test if the given ETL expresion type is fast (sizes known at compile-time)
 * \tparam E The ETL expression type.
 */
template <typename E>
constexpr bool is_fast = decay_traits<E>::is_fast;

/*!
 * \brief Traits to test if all the given ETL expresion types are fast (sizes known at compile-time)
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_fast = (decay_traits<E>::is_fast && ...);

/*!
 * \brief Traits to test if all the given types are ETL types.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_etl_expr = (is_etl_expr<E> && ...);

/*!
 * \brief Traits to test if the given expression type is 1D
 * \tparam T The ETL expression type
 */
template <typename T>
constexpr bool is_1d = decay_traits<T>::dimensions() == 1;

/*!
 * \brief Traits to test if all the given expression types are 1D
 * \tparam T The ETL expression type
 */
template <typename... T>
constexpr bool all_1d = (is_1d<T> && ...);

/*!
 * \brief Traits to test if the given expression type is 2D
 * \tparam T The ETL expression type
 */
template <typename T>
constexpr bool is_2d = decay_traits<T>::dimensions() == 2;

/*!
 * \brief Traits to test if all the given expression types are 2D
 * \tparam T The ETL expression type
 */
template <typename... T>
constexpr bool all_2d = (is_2d<T> && ...);

/*!
 * \brief Traits to test if the given expression type is 3D
 * \tparam T The ETL expression type
 */
template <typename T>
constexpr bool is_3d = decay_traits<T>::dimensions() == 3;

/*!
 * \brief Traits to test if all the given expression types are 3D
 * \tparam T The ETL expression type
 */
template <typename... T>
constexpr bool all_3d = (is_3d<T> && ...);

/*!
 * \brief Traits to test if the given expression type is 4D
 * \tparam T The ETL expression type
 */
template <typename T>
constexpr bool is_4d = decay_traits<T>::dimensions() == 4;

/*!
 * \brief Traits to test if all the given expression types are 4D
 * \tparam T The ETL expression type
 */
template <typename... T>
constexpr bool all_4d = (is_4d<T> && ...);

/*!
 * \brief Traits to test if all the given ETL expresion types are vectorizable.
 * \tparam E The ETL expression types.
 */
template <vector_mode_t V, typename... E>
constexpr bool all_vectorizable = (decay_traits<E>::template vectorizable<V> && ...);

/*!
 * \brief Traits to test if the given type are vectorizable types.
 * \tparam E The type.
 */
template <vector_mode_t V, typename E>
static constexpr bool vectorizable_t = get_intrinsic_traits<V>::template type<value_t<E>>::vectorizable;

/*!
 * \brief Traits to test if all the given types are vectorizable types.
 * \tparam E The types.
 */
template <vector_mode_t V, typename... E>
constexpr bool all_vectorizable_t = (vectorizable_t<V, E> & ...);

/*!
 * \brief Traits to test if the given ETL expresion type is
 * thread safe.
 * \tparam E The ETL expression type
 */
template <typename E>
constexpr bool is_thread_safe = decay_traits<E>::is_thread_safe;

/*!
 * \brief Traits to test if all the given ETL expresion types are
 * thread safe.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_thread_safe = (decay_traits<E>::is_thread_safe && ...);

/*!
 * \brief Traits to test if the givn ETL expression is a padded value class.
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_padded_value = is_dyn_matrix<T> || is_fast_matrix<T>;

/*!
 * \brief Traits to test if the givn ETL expression is an aligned value class.
 * \tparam T The ETL expression type.
 */
template <typename T>
constexpr bool is_aligned_value = is_dyn_matrix<T> || is_fast_matrix<T>;

/*!
 * \brief Traits to test if all the given ETL expresion types are padded.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_padded = (decay_traits<E>::is_padded && ...);

/*!
 * \brief Traits indicating if the given ETL expression's type is computable
 * on GPU.
 *
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_gpu_computable = decay_traits<T>::gpu_computable;

/*!
 * \brief Traits indicating if all the given ETL expresion types are computable
 * on GPU.
 *
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_gpu_computable = (decay_traits<E>::gpu_computable && ...);

/*!
 * \brief Traits to test if all the given ETL expresion types are padded.
 * \tparam E The ETL expression types.
 */
template <typename... E>
constexpr bool all_homogeneous = cpp::is_homogeneous_v<value_t<E>...>;

/*!
 * \brief Simple utility traits indicating if a light subview can be created out
 * of this type.
 */
template <typename T>
constexpr bool fast_sub_view_able = is_dma<T>&& decay_traits<T>::storage_order == order::RowMajor;

/*!
 * \brief Simple utility traits indicating if a light sub_matrix can be created out
 * of this type.
 */
template <typename T>
constexpr bool fast_sub_matrix_able = is_dma<T>;

/*!
 * \brief Simple utility traits indicating if a light slice view can be created out
 * of this type.
 */
template <typename T>
constexpr bool fast_slice_view_able = fast_sub_view_able<T>;

namespace traits_detail {

/*!
 * \brief Traits to test if an expression is inplace sub transpose-able.
 *
 * Sub-transpose able means that the last two dimensions can be transposed in place.
 *
 * \tparam T The type to test
 */
template <typename T, typename Enable = void>
struct inplace_sub_transpose_able_impl;

/*!
 * \copydoc inplace_sub_transpose_able_impl
 */
template <typename T>
struct inplace_sub_transpose_able_impl<T, std::enable_if_t<is_fast<T> && is_3d<T>>> {
    /*!
     * \brief Indicates if T is inplace sub-transpose-able
     */
    static constexpr bool value = decay_traits<T>::template dim<1>() == decay_traits<T>::template dim<2>();
};

/*!
 * \copydoc inplace_sub_transpose_able_impl
 */
template <typename T>
struct inplace_sub_transpose_able_impl<T, std::enable_if_t<!is_fast<T> && is_3d<T>>> {
    /*!
     * \brief Indicates if T is inplace sub-transpose-able
     */
    static constexpr bool value = true;
};

/*!
 * \copydoc inplace_sub_transpose_able_impl
 */
template <typename T>
struct inplace_sub_transpose_able_impl<T, std::enable_if_t<!is_3d<T>>> {
    /*!
     * \brief Indicates if T is inplace sub-transpose-able
     */
    static constexpr bool value = false;
};

/*!
 * \brief Traits to test if an expression is inplace transpose-able
 * \tparam T The type to test
 */
template <typename T, typename Enable = void>
struct inplace_transpose_able_impl;

/*!
 * \copydoc inplace_transpose_able_impl
 */
template <typename T>
struct inplace_transpose_able_impl<T, std::enable_if_t<is_fast<T> && is_2d<T>>> {
    /*!
     * \brief Indicates if T is inplace transpose-able
     */
    static constexpr bool value = decay_traits<T>::template dim<0>() == decay_traits<T>::template dim<1>();
};

/*!
 * \copydoc inplace_transpose_able_impl
 */
template <typename T>
struct inplace_transpose_able_impl<T, std::enable_if_t<!is_fast<T> && is_2d<T>>> {
    /*!
     * \brief Indicates if T is inplace transpose-able
     */
    static constexpr bool value = true;
};

/*!
 * \copydoc inplace_transpose_able_impl
 */
template <typename T>
struct inplace_transpose_able_impl<T, std::enable_if_t<!is_2d<T>>> {
    /*!
     * \brief Indicates if T is inplace transpose-able
     */
    static constexpr bool value = false;
};

/*!
 * \brief Traits to test if a matrix is a square matrix, if this can be defined.
 */
template <typename Matrix, typename Enable = void>
struct is_square_matrix_impl {
    /*!
     * \brief The value of the traits. True if the matrix is square, false otherwise
     */
    static constexpr bool value = false;
};

/*!
 * \copydoc is_square_matrix_impl
 */
template <typename Matrix>
struct is_square_matrix_impl<Matrix, std::enable_if_t<is_fast<Matrix> && is_2d<Matrix>>> {
    /*!
     * \brief The value of the traits. True if the matrix is square, false otherwise
     */
    static constexpr bool value = etl_traits<Matrix>::template dim<0>() == etl_traits<Matrix>::template dim<1>();
};

/*!
 * \copydoc is_square_matrix_impl
 */
template <typename Matrix>
struct is_square_matrix_impl<Matrix, std::enable_if_t<!is_fast<Matrix> && is_2d<Matrix>>> {
    /*!
     * \brief The value of the traits. True if the matrix is square, false otherwise
     */
    static constexpr bool value = true;
};

} //end of namespace traits_detail

/*!
 * \brief Traits to test if an expression is inplace transpose-able
 * \tparam T The type to test
 */
template <typename T>
constexpr bool inplace_transpose_able = traits_detail::inplace_transpose_able_impl<T>::value;

/*!
 * \brief Traits to test if an expression is inplace sub transpose-able.
 *
 * Sub-transpose able means that the last two dimensions can be transposed in place.
 *
 * \tparam T The type to test
 */
template <typename T>
constexpr bool inplace_sub_transpose_able = traits_detail::inplace_sub_transpose_able_impl<T>::value;

/*!
 * \brief Traits to test if a matrix is a square matrix, if this can be defined.
 */
template <typename Matrix>
constexpr bool is_square_matrix = traits_detail::is_square_matrix_impl<Matrix>::value;

/*!
 * \brief Traits to test if an expression is a temporary expression with non-GPU
 * capabilities
 */
template <typename E>
constexpr bool is_nongpu_temporary = traits_detail::is_nongpu_temporary_impl<E>::value;

/*!
 * \brief Traits to test if an expression is a temporary expression with GPU
 * capabilities
 */
template <typename E>
constexpr bool is_gpu_temporary = traits_detail::is_gpu_temporary_impl<E>::value;

/*!
 * \brief Traits indicating if it's more efficient to use smart_gpu_compute(x)
 * instead of smart_gpu_compute(x, y) for an expression of type E.
 */
template <typename E>
constexpr bool should_gpu_compute_direct = is_etl_value<E> || is_nongpu_temporary<E> || (is_dma<E> && !is_gpu_temporary<E>);

/*!
 * Builder to construct the type returned by a view.
 */
template <typename T, typename S>
using return_helper = std::conditional_t<
    std::is_const_v<std::remove_reference_t<S>>,
    const value_t<T>&,
    std::conditional_t<std::is_lvalue_reference_v<S> && !std::is_const_v<T>, value_t<T>&, value_t<T>>>;

/*!
 * Builder to construct the const type returned by a view.
 */
template <typename T, typename S>
using const_return_helper = std::conditional_t<std::is_lvalue_reference_v<S>, const value_t<T>&, value_t<T>>;

/*!
 * \brief Specialization for value structures
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<is_etl_value_class<T>>> {
    using value_type = typename T::value_type; ///< The value type of the expression

    static constexpr bool is_etl         = true;                                          ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                         ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                         ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                         ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = is_fast_matrix<T> || is_custom_fast_matrix<T>; ///< Indicates if the expression is fast
    static constexpr bool is_value       = true;                                          ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = !is_sparse_matrix<T>;                          ///< Indicates if the expression has direct memory access
    static constexpr bool is_thread_safe = true;                                          ///< Indicates if the expression is thread safe
    static constexpr bool is_linear      = true;                                          ///< Indicates if the expression is linear
    static constexpr bool is_generator   = false;                                         ///< Indicates if the expression is a generator expression
    static constexpr bool is_temporary   = false;                                         ///< Indicates if the expression needs an evaluator visitor
    static constexpr bool is_padded      = is_padded_value<T>;                            ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = is_aligned_value<T>;                           ///< Indicates if the expression is aligned
    static constexpr order storage_order = T::storage_order;                              ///< The expression storage order
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled;          ///< Indicates if the expression can be computed on GPU

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = get_intrinsic_traits<V>::template type<value_type>::vectorizable && !is_sparse_matrix<T> && !is_gpu_dyn_matrix<T>;

    /*!
     * \brief Return the size of the given epxression
     * \param v The expression to get the size from
     * \return the size of the given expressio
     */
    static size_t size(const T& v) {
        return v.size();
    }

    /*!
     * \brief Return the dth dimension of the given epxression
     * \param v The expression to get the size from
     * \param d The dimension to get
     * \return the dth dimension of the given expressio
     */
    static size_t dim(const T& v, size_t d) {
        return v.dim(d);
    }

    /*!
     * \brief Return the size of an expression of the given type
     * \return the size of an expression of the given type
     */
    static constexpr size_t size() {
        static_assert(is_fast, "Only fast_matrix have compile-time access to the dimensions");

        return T::size();
    }

    /*!
     * \brief Return the Dth dimension of the given epxression
     * \tparam D The dimension to get
     * \return the Dth dimension of the given expressio
     */
    template <size_t D>
    static constexpr size_t dim() {
        static_assert(is_fast, "Only fast_matrix have compile-time access to the dimensions");

        return T::template dim<D>();
    }

    /*!
     * \brief Return the number of dimensions of an expression of the given type
     * \return the number of dimensions of an expression of the given type
     */
    static constexpr size_t dimensions() {
        return T::n_dimensions;
    }

    /*!
     * \brief Estimate the complexity of computation
     * \return An estimation of the complexity of the expression
     */
    static constexpr int complexity() noexcept {
        return -1;
    }
};

/*!
 * \brief Return the number of dimensions of the given ETL expression
 * \param expr The expression to get the number of dimensions for
 * \return The number of dimensions of the given expression.
 */
template <typename E>
constexpr size_t dimensions([[maybe_unused]] const E& expr) noexcept {
    return etl_traits<E>::dimensions();
}

/*!
 * \brief Return the number of dimensions of the given ETL type
 * \tparam E The expression type to get the number of dimensions for
 * \return The number of dimensions of the given type.
 */
template <typename E>
constexpr size_t dimensions() noexcept {
    return decay_traits<E>::dimensions();
}

/*!
 * \brief Return the complexity of the expression
 * \param expr The expression to get the complexity for
 * \return The estimated complexity of the given expression.
 */
template <typename E>
constexpr int complexity([[maybe_unused]] const E& expr) noexcept {
    return etl_traits<E>::complexity();
}

/*!
 * \brief Return the complexity of the expression
 * \tparam E The expression type to get the complexity for
 * \return The estimated complexity of the given type.
 */
template <typename E>
constexpr int complexity() noexcept {
    return decay_traits<E>::complexity();
}

/*!
 * \brief Returns the number of rows of the given ETL expression.
 * \param expr The expression to get the number of rows from.
 * \return The number of rows of the given expression.
 */
template <typename E, cpp_disable_iff(decay_traits<E>::is_fast)>
size_t rows(const E& expr) {
    return etl_traits<E>::dim(expr, 0);
}

/*!
 * \brief Returns the number of rows of the given ETL expression.
 * \param expr The expression to get the number of rows from.
 * \return The number of rows of the given expression.
 */
template <typename E, cpp_enable_iff(decay_traits<E>::is_fast)>
constexpr size_t rows(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::template dim<0>();
}

/*!
 * \brief Returns the number of columns of the given ETL expression.
 * \param expr The expression to get the number of columns from.
 * \return The number of columns of the given expression.
 */
template <typename E, cpp_disable_iff(decay_traits<E>::is_fast)>
size_t columns(const E& expr) {
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return etl_traits<E>::dim(expr, 1);
}

/*!
 * \brief Returns the number of columns of the given ETL expression.
 * \param expr The expression to get the number of columns from.
 * \return The number of columns of the given expression.
 */
template <typename E, cpp_enable_iff(decay_traits<E>::is_fast)>
constexpr size_t columns(const E& expr) noexcept {
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return (void)expr, etl_traits<E>::template dim<1>();
}

/*!
 * \brief Returns the size of the given ETL expression.
 * \param expr The expression to get the size from.
 * \return The size of the given expression.
 */
template <typename E, cpp_disable_iff(decay_traits<E>::is_fast)>
size_t size(const E& expr) {
    return etl_traits<E>::size(expr);
}

/*!
 * \brief Returns the size of the given ETL expression.
 * \param expr The expression to get the size from.
 * \return The size of the given expression.
 */
template <typename E, cpp_enable_iff(decay_traits<E>::is_fast)>
constexpr size_t size(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::size();
}

/*!
 * \brief Returns the sub-size of the given ETL expression, i.e. the size not considering the first dimension.
 * \param expr The expression to get the sub-size from.
 * \return The sub-size of the given expression.
 */
template <typename E, cpp_disable_iff(decay_traits<E>::is_fast)>
size_t subsize(const E& expr) {
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return etl_traits<E>::size(expr) / etl_traits<E>::dim(expr, 0);
}

/*!
 * \brief Returns the sub-size of the given ETL expression, i.e. the size not considering the first dimension.
 * \param expr The expression to get the sub-size from.
 * \return The sub-size of the given expression.
 */
template <typename E, cpp_enable_iff(decay_traits<E>::is_fast)>
constexpr size_t subsize(const E& expr) noexcept {
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return (void)expr, etl_traits<E>::size() / etl_traits<E>::template dim<0>();
}

/*!
 * \brief Return the D dimension of e
 * \param e The expression to get the dimensions from
 * \tparam D The dimension to get
 * \return the Dth dimension of e
 */
template <size_t D, typename E, cpp_disable_iff(decay_traits<E>::is_fast)>
size_t dim(const E& e) noexcept {
    return etl_traits<E>::dim(e, D);
}

/*!
 * \brief Return the d dimension of e
 * \param e The expression to get the dimensions from
 * \param d The dimension to get
 * \return the dth dimension of e
 */
template <typename E>
size_t dim(const E& e, size_t d) noexcept {
    return etl_traits<E>::dim(e, d);
}

/*!
 * \brief Return the D dimension of e
 * \param e The expression to get the dimensions from
 * \tparam D The dimension to get
 * \return the Dth dimension of e
 */
template <size_t D, typename E, cpp_enable_iff(decay_traits<E>::is_fast)>
constexpr size_t dim(const E& e) noexcept {
    return (void)e, etl_traits<E>::template dim<D>();
}

/*!
 * \brief Return the D dimension of E
 * \return the Dth dimension of E
 */
template <size_t D, typename E>
constexpr size_t dim() noexcept {
    static_assert(is_fast<E>, "dim<D, E>() can only be used on statically-sized ETL expression");
    return decay_traits<E>::template dim<D>();
}

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E, typename Enable = void>
struct safe_dimensions_impl;

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E>
struct safe_dimensions_impl<E, std::enable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<size_t, std::numeric_limits<size_t>::max()> {};

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E>
struct safe_dimensions_impl<E, cpp::disable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<size_t, etl_traits<E>::dimensions()> {};

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E, typename Enable = void>
constexpr size_t safe_dimensions = safe_dimensions_impl<E>::value;

/*!
 * \brief Convert a flat index into a 2D index
 * \param sub The matrix expression
 * \param i The flat index
 * \return a pair of indices for the equivalent 2D index
 */
template <typename E>
constexpr std::pair<size_t, size_t> index_to_2d(E&& sub, size_t i) {
    return decay_traits<E>::storage_order == order::RowMajor ? std::make_pair(i / dim<1>(sub), i % dim<1>(sub))
                                                             : std::make_pair(i % dim<0>(sub), i / dim<0>(sub));
}

/*!
 * \brief Returns the row stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the row stride of the given ETL matrix expression
 */
template <typename E>
size_t row_stride(E&& expr) {
    static_assert(is_2d<E>, "row_stride() only makes sense on 2D matrices");
    return decay_traits<E>::storage_order == order::RowMajor ? etl::dim<1>(expr) : 1;
}

/*!
 * \brief Returns the column stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the column stride of the given ETL matrix expression
 */
template <typename E>
size_t col_stride(E&& expr) {
    static_assert(is_2d<E>, "col_stride() only makes sense on 2D matrices");
    return decay_traits<E>::storage_order == order::RowMajor ? 1 : etl::dim<0>(expr);
}

/*!
 * \brief Returns the minor stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the minor stride of the given ETL matrix expression
 */
template <typename E>
size_t minor_stride(E&& expr) {
    static_assert(is_2d<E>, "minor_stride() only makes sense on 2D matrices");
    return decay_traits<E>::storage_order == order::RowMajor ? etl::dim<0>(expr) : etl::dim<1>(expr);
}

/*!
 * \brief Returns the major stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the major stride of the given ETL matrix expression
 */
template <typename E>
size_t major_stride(E&& expr) {
    static_assert(is_2d<E>, "major_stride() only makes sense on 2D matrices");
    return decay_traits<E>::storage_order == order::RowMajor ? etl::dim<1>(expr) : etl::dim<0>(expr);
}

/*!
 * \brief Test if two memory ranges overlap.
 * \param a_begin The beginning of the first range
 * \param a_end The end (exclusive) of the first range
 * \param b_begin The beginning of the second range
 * \param b_end The end (exclusive) of the second range
 *
 * The ranges must be ordered (begin <= end). This function is optimized so that only two comparisons are performed.
 *
 * \return true if the two ranges overlap, false otherwise
 */
template <typename P1, typename P2>
bool memory_alias(const P1* a_begin, const P1* a_end, const P2* b_begin, const P2* b_end) {
    cpp_assert(a_begin <= a_end, "memory_alias works on ordered ranges");
    cpp_assert(b_begin <= b_end, "memory_alias works on ordered ranges");

    return reinterpret_cast<uintptr_t>(a_begin) < reinterpret_cast<uintptr_t>(b_end)
           && reinterpret_cast<uintptr_t>(a_end) > reinterpret_cast<uintptr_t>(b_begin);
}

/*!
 * \brief Ensure that the CPU is up to date.
 *
 * \param expr The expression
 */
template <typename E>
void safe_ensure_cpu_up_to_date(E&& expr) {
    expr.ensure_cpu_up_to_date();
}

/*!
 * \brief Indicates if the CPU memory is up to date. If the expression does
 * not have direct memory access, return true
 *
 * \param expr The expression
 */
template <typename E>
bool safe_is_cpu_up_to_date(E&& expr) {
    if constexpr (is_dma<E>) {
        return expr.is_cpu_up_to_date();
    } else {
        return true;
    }
}

/*!
 * \brief Indicates if the GPU memory is up to date. If the expression does
 * not have direct memory access, return false
 *
 * \param expr The expression
 */
template <typename E>
bool safe_is_gpu_up_to_date(E&& expr) {
    if constexpr (is_dma<E>) {
        return expr.is_gpu_up_to_date();
    } else {
        return false;
    }
}

/*!
 * \brief Smart forwarding for a temporary expression.
 *
 * This is guaranteed to produce a DMA expression in the most
 * efficient way possible.
 *
 * \param expr the Expresison from which to create a temporary.
 *
 * \return a direct expression
 */
template <typename E>
decltype(auto) smart_forward(E& expr) {
    if constexpr (is_temporary_expr<E>) {
        return force_temporary(expr);
    } else {
        return make_temporary(expr);
    }
}

/*!
 * \brief Smart forwarding for a temporary expression that will be
 * computed in GPU.
 *
 * This is guaranteed to produce a DMA expression in the most
 * efficient way possible.
 *
 * \param expr the Expresison from which to create a temporary.
 *
 * \return a direct GPU-able expression
 */
template <typename E>
decltype(auto) smart_forward_gpu(E& expr) {
    if constexpr (is_temporary_expr<E>) {
        if constexpr (E::gpu_computable) {
            return force_temporary_gpu(expr);
        } else {
            return force_temporary(expr);
        }
    } else {
        return make_temporary(expr);
    }
}

// Unary smart_gpu_compute

/*!
 * \brief Compute the expression into a representation that is GPU up to date.
 *
 * This function tries to minimize the number of copies and evaluations that is
 * performed.
 *
 * \param expr The expression that must be evaluated
 *
 * \return A gpu-computed expression reprensenting the results of the input expr
 */
template <typename E, typename Y>
decltype(auto) smart_gpu_compute_hint(E& expr, Y& y) {
    if constexpr (is_temporary_expr<E>) {
        if constexpr (E::gpu_computable) {
            return force_temporary_gpu(expr);
        } else {
            auto t = force_temporary(expr);
            t.ensure_gpu_up_to_date();
            return t;
        }
    } else {
        return expr.gpu_compute_hint(y);
    }
}

// Binary smart_gpu_compute

/*!
 * \brief Compute the expression into a representation that is GPU up to date
 * and store this representation in y.
 *
 * This function tries to minimize the number of copies and evaluations that is
 * performed. Ideally, the result will be directly computed inside y.
 *
 * \param x The expression that must be evaluated
 * \param y The expression into which store the GPU result of x
 *
 * \return y
 */
template <typename X, typename Y>
decltype(auto) smart_gpu_compute(X& x, Y& y) {
    if constexpr (is_temporary_expr<X>) {
        if constexpr (X::gpu_computable) {
            return y = x;
        } else {
            auto t = force_temporary(x);
            t.ensure_gpu_up_to_date();
            y = t;
            return y;
        }
    } else {
        if constexpr (is_dma<X>) {
            x.ensure_gpu_up_to_date();
            y.ensure_gpu_allocated();
            y.gpu_copy_from(x.gpu_memory());
            return y;
        } else {
            return x.gpu_compute(y);
        }
    }
}

// Select version

/*!
 * \brief Compute the expression into a representation that is GPU up to date
 * and possibly store this representation in y.
 *
 * This function tries to minimize the number of copies and evaluations that is
 * performed. Ideally, the result will be directly computed inside y.
 *
 * \param x The expression that must be evaluated
 * \param y The expression into which store the GPU result of x
 *
 * \return either a temporary of the result of x (possibly x) or y
 */
template <typename X, typename Y>
decltype(auto) select_smart_gpu_compute(X& x, Y& y) {
    if constexpr (should_gpu_compute_direct<X>) {
        return smart_gpu_compute_hint(x, y);
    } else {
        return smart_gpu_compute(x, y);
    }
}

/*!
 * \brief The space between two elements in GPU for the given type.
 *
 * This is 1 for every type but for scalar which are 0.
 */
template <typename T>
constexpr size_t gpu_inc = is_scalar<T> ? 0 : 1;

} //end of namespace etl
