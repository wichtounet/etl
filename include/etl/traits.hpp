//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace traits_detail {

template <typename T>
struct is_fast_matrix_impl : std::false_type {};

template <typename V1, typename V2, order V3, std::size_t... R>
struct is_fast_matrix_impl<fast_matrix_impl<V1, V2, V3, R...>> : std::true_type {};

template <typename T>
struct is_custom_fast_matrix_impl : std::false_type {};

template <typename V1, typename V2, order V3, std::size_t... R>
struct is_custom_fast_matrix_impl<custom_fast_matrix_impl<V1, V2, V3, R...>> : std::true_type {};

template <typename T>
struct is_dyn_matrix_impl : std::false_type {};

template <typename V1, order V2, std::size_t V3>
struct is_dyn_matrix_impl<dyn_matrix_impl<V1, V2, V3>> : std::true_type {};

template <typename T>
struct is_custom_dyn_matrix_impl : std::false_type {};

template <typename V1, order V2, std::size_t V3>
struct is_custom_dyn_matrix_impl<custom_dyn_matrix_impl<V1, V2, V3>> : std::true_type {};

template <typename T>
struct is_sparse_matrix_impl : std::false_type {};

template <typename V1, sparse_storage V2, std::size_t V3>
struct is_sparse_matrix_impl<sparse_matrix_impl<V1, V2, V3>> : std::true_type {};

template <typename T>
struct is_dyn_matrix_view : std::false_type {};

template <typename E, std::size_t D, typename Enable>
struct is_dyn_matrix_view<dyn_matrix_view<E, D, Enable>> : std::true_type {};

template <typename T>
struct is_selected_expr_impl : std::false_type {};

template <typename Selector, Selector V, typename Expr>
struct is_selected_expr_impl<selected_expr<Selector, V, Expr>> : std::true_type {};

} //end of namespace traits_detail

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
};

/*!
 * \brief Traits indicating if the given ETL type is a fast matrix
 * \tparam T The type to test
 */
template <typename T>
using is_fast_matrix = traits_detail::is_fast_matrix_impl<std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a fast matrix
 * \tparam T The type to test
 */
template <typename T>
using is_custom_fast_matrix = traits_detail::is_custom_fast_matrix_impl<std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a dyn matrix
 * \tparam T The type to test
 */
template <typename T>
using is_dyn_matrix = traits_detail::is_dyn_matrix_impl<std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a custom dyn matrix
 * \tparam T The type to test
 */
template <typename T>
using is_custom_dyn_matrix = traits_detail::is_custom_dyn_matrix_impl<std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a sparse matrix
 * \tparam T The type to test
 */
template <typename T>
using is_sparse_matrix = traits_detail::is_sparse_matrix_impl<std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a symmetric matrix
 * \tparam T The type to test
 */
template <typename T>
using is_symmetric_matrix = cpp::is_specialization_of<etl::symmetric_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a hermitian matrix
 * \tparam T The type to test
 */
template <typename T>
using is_hermitian_matrix = cpp::is_specialization_of<etl::hermitian_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a diagonal matrix
 * \tparam T The type to test
 */
template <typename T>
using is_diagonal_matrix = cpp::is_specialization_of<etl::diagonal_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is an upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
using is_upper_matrix = cpp::is_specialization_of<etl::upper_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
using is_lower_matrix = cpp::is_specialization_of<etl::lower_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a strictly lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
using is_strictly_lower_matrix = cpp::is_specialization_of<etl::strictly_lower_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a strictly upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
using is_strictly_upper_matrix = cpp::is_specialization_of<etl::strictly_upper_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a uni lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
using is_uni_lower_matrix = cpp::is_specialization_of<etl::uni_lower_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a uni upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
using is_uni_upper_matrix = cpp::is_specialization_of<etl::uni_upper_matrix, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a unary expression.
 * \tparam T The type to test
 */
template <typename T>
using is_unary_expr = cpp::is_specialization_of<etl::unary_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a binary expression.
 * \tparam T The type to test
 */
template <typename T>
using is_binary_expr = cpp::is_specialization_of<etl::binary_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a generator expression.
 * \tparam T The type to test
 */
template <typename T>
using is_generator_expr = cpp::is_specialization_of<etl::generator_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is an optimized expression.
 * \tparam T The type to test
 */
template <typename T>
using is_optimized_expr = cpp::is_specialization_of<etl::optimized_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a serial expression.
 * \tparam T The type to test
 */
template <typename T>
using is_serial_expr = cpp::is_specialization_of<etl::serial_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a selector expression.
 * \tparam T The type to test
 */
template <typename T>
using is_selected_expr = traits_detail::is_selected_expr_impl<std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a parallel expression.
 * \tparam T The type to test
 */
template <typename T>
using is_parallel_expr = cpp::is_specialization_of<etl::parallel_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a timed expression.
 * \tparam T The type to test
 */
template <typename T>
using is_timed_expr = cpp::is_specialization_of<etl::timed_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a wrapper expression (optimized, serial, ...).
 * \tparam T The type to test
 */
template <typename T>
using is_wrapper_expr = cpp::or_c<is_optimized_expr<T>, is_selected_expr<T>, is_serial_expr<T>, is_parallel_expr<T>, is_timed_expr<T>>;

/*!
 * \brief Traits to test if the given expression is a sub_view
 */
template <typename T>
using is_sub_view = cpp::is_specialization_of<etl::sub_view, std::decay_t<T>>;

/*!
 * \brief Traits to test if the given expression is a slice_view
 */
template <typename T>
using is_slice_view = cpp::is_specialization_of<etl::slice_view, std::decay_t<T>>;

/*!
 * \brief Traits to test if the given expression is a dyn_matrix_view
 */
template <typename T>
using is_dyn_matrix_view = traits_detail::is_dyn_matrix_view<T>;

/*!
 * \brief Traits indicating if the given ETL type is a temporary unary expression.
 * \tparam T The type to test
 */
template <typename T>
using is_temporary_unary_expr = cpp::or_c<
    cpp::is_specialization_of<etl::temporary_unary_expr, std::decay_t<T>>,
    cpp::is_specialization_of<etl::temporary_unary_expr_state, std::decay_t<T>>>;

/*!
 * \brief Traits indicating if the given ETL type is a temporary binary expression.
 * \tparam T The type to test
 */
template <typename T>
using is_temporary_binary_expr = cpp::or_c<
    cpp::is_specialization_of<etl::temporary_binary_expr, std::decay_t<T>>,
    cpp::is_specialization_of<etl::temporary_binary_expr_state, std::decay_t<T>>>;

/*!
 * \brief Traits indicating if the given ETL type is a temporary expression.
 * \tparam T The type to test
 */
template <typename T>
using is_temporary_expr = cpp::or_c<is_temporary_unary_expr<T>, is_temporary_binary_expr<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a transformer expression.
 * \tparam T The type to test
 */
template <typename T>
using is_transformer = cpp::bool_constant<decay_traits<T>::is_transformer>;

/*!
 * \brief Traits indicating if the given ETL type is a view expression.
 * \tparam T The type to test
 */
template <typename T>
using is_view = cpp::bool_constant<decay_traits<T>::is_view>;

/*!
 * \brief Traits indicating if the given ETL type is a magic view expression.
 * \tparam T The type to test
 */
template <typename T>
using is_magic_view = cpp::bool_constant<decay_traits<T>::is_magic_view>;

/*!
 * \brief Traits indicating if the given type is an ETL type.
 * \tparam T The type to test
 */
template <typename T>
using is_etl_expr = cpp::bool_constant<decay_traits<T>::is_etl>;

/*!
 * \brief Traits indicating if the given type is a transpose expr.
 * \tparam T The type to test
 */
template <typename T>
using is_transpose_expr = cpp::is_specialization_of<etl::transpose_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a value type.
 * \tparam T The type to test
 */
template <typename T>
using is_etl_value = cpp::bool_constant<decay_traits<T>::is_value>;

/*!
 * \brief Traits indicating if the given ETL type is from a value class.
 * \tparam T The type to test
 */
template <typename T>
using is_etl_value_class = cpp::or_c<is_fast_matrix<T>, is_custom_fast_matrix<T>, is_dyn_matrix<T>, is_custom_dyn_matrix<T>, is_sparse_matrix<T>>;

/*!
 * \brief Traits indicating if the given ETL type can be left hand side type
 * \tparam T The type to test
 */
template <typename T>
using is_lhs = cpp::or_c<is_etl_value<T>, is_unary_expr<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a simple left hand side type.
 * Adapter types are not taken from this because they do more operations.
 * \tparam T The type to test
 */
template <typename T>
using is_simple_lhs = cpp::or_c<is_etl_value_class<T>, is_unary_expr<T>, is_sub_view<T>, is_slice_view<T>, is_dyn_matrix_view<T>>;

/*!
 * \brief Traits indicating if the given ETL type has direct memory access.
 * \tparam T The type to test
 */
template <typename T>
using has_direct_access = cpp::bool_constant<decay_traits<T>::is_direct>;

/*!
 * \brief Traits to test if the given type is single precision type.
 * \tparam T The type
 */
template <typename T>
using is_single_precision_t = std::is_same<T, float>;

/*!
 * \brief Traits to test if the given ETL expresion contains single precision numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_single_precision = is_single_precision_t<value_t<T>>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains single precision numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_single_precision = cpp::and_c<is_single_precision<E>...>;

/*!
 * \brief Traits to test if the given type is double precision type.
 * \tparam T The type
 */
template <typename T>
using is_double_precision_t = std::is_same<T, double>;

/*!
 * \brief Traits to test if the given ETL expresion contains double precision numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_double_precision = is_double_precision_t<value_t<T>>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains double precision numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_double_precision = cpp::and_c<is_double_precision<E>...>;

/*!
 * \brief Traits to test if the given ETL expresion contains floating point numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_floating = cpp::or_c<is_single_precision<T>, is_double_precision<T>>;

/*!
 * \brief Traits to test if the type is floating point numbers.
 * \tparam T The type.
 */
template <typename T>
using is_floating_t = cpp::or_c<is_single_precision_t<T>, is_double_precision_t<T>>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains floating point numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_floating = cpp::and_c<is_floating<E>...>;

/*!
 * \brief Traits to test if a type is a complex number type
 * \tparam T The type to test.
 */
template <typename T>
using is_complex_t = cpp::or_c<cpp::is_specialization_of<std::complex, std::decay_t<T>>, cpp::is_specialization_of<etl::complex, std::decay_t<T>>>;

/*!
 * \brief Traits to test if a type is a single precision complex number type
 * \tparam T The type to test.
 */
template <typename T>
using is_complex_single_t = cpp::or_c<std::is_same<T, std::complex<float>>, std::is_same<T, etl::complex<float>>>;

/*!
 * \brief Traits to test if a type is a double precision complex number type
 * \tparam T The type to test.
 */
template <typename T>
using is_complex_double_t = cpp::or_c<std::is_same<T, std::complex<double>>, std::is_same<T, etl::complex<double>>>;

/*!
 * \brief Traits to test if the given ETL expresion type contains single precision complex numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_complex_single_precision = is_complex_single_t<value_t<T>>;

/*!
 * \brief Traits to test if the given ETL expresion type contains double precision complex numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_complex_double_precision = is_complex_double_t<value_t<T>>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains single precision complex numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_complex_single_precision = cpp::and_c<is_complex_single_precision<E>...>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains double precision complex numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_complex_double_precision = cpp::and_c<is_complex_double_precision<E>...>;

/*!
 * \brief Traits to test if the given ETL expresion type contains complex numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_complex = cpp::or_c<is_complex_single_precision<T>, is_complex_double_precision<T>>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains complex numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_complex = cpp::and_c<is_complex<E>...>;

/*!
 * \brief Traits to test if the given ETL expresion type has direct memory access (DMA).
 * \tparam E The ETL expression type.
 */
template <typename E>
using is_dma = has_direct_access<E>;

/*!
 * \brief Traits to test if all the given ETL expresion types have direct memory access (DMA).
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_dma = cpp::and_c<has_direct_access<E>...>;

/*!
 * \brief Traits to test if all the given ETL expresion types are row-major.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_row_major = cpp::and_u<(decay_traits<E>::storage_order == order::RowMajor)...>;

/*!
 * \brief Traits to test if all the given ETL expresion types are column-major.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_column_major = cpp::and_u<(decay_traits<E>::storage_order == order::ColumnMajor)...>;

/*!
 * \brief Traits to test if all the given ETL expresion types are fast (sizes known at compile-time)
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_fast = cpp::and_u<decay_traits<E>::is_fast...>;

/*!
 * \brief Traits to test if all the given types are ETL types.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_etl_expr = cpp::and_c<is_etl_expr<E>...>;

/*!
 * \brief Traits to test if the given expression type is 1D
 * \tparam T The ETL expression type
 */
template <typename T>
using is_1d = cpp::bool_constant<decay_traits<T>::dimensions() == 1>;

/*!
 * \brief Traits to test if the given expression type is 2D
 * \tparam T The ETL expression type
 */
template <typename T>
using is_2d = cpp::bool_constant<decay_traits<T>::dimensions() == 2>;

/*!
 * \brief Traits to test if the given expression type is 3D
 * \tparam T The ETL expression type
 */
template <typename T>
using is_3d = cpp::bool_constant<decay_traits<T>::dimensions() == 3>;

/*!
 * \brief Traits to test if the given expression type is 4D
 * \tparam T The ETL expression type
 */
template <typename T>
using is_4d = cpp::bool_constant<decay_traits<T>::dimensions() == 4>;

/*!
 * \brief Traits to test if all the given ETL expresion types are vectorizable.
 * \tparam E The ETL expression types.
 */
template <vector_mode_t V, typename... E>
using all_vectorizable = cpp::and_u<decay_traits<E>::template vectorizable<V>::value...>;

/*!
 * \brief Traits to test if all the given ETL expresion types are
 * thread safe.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_thread_safe = cpp::and_u<decay_traits<E>::is_thread_safe...>;

/*!
 * \brief Traits to test if the givn ETL expression is a padded value class.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_padded_value = cpp::or_u<is_dyn_matrix<T>::value, is_fast_matrix<T>::value>;

/*!
 * \brief Traits to test if the givn ETL expression is an aligned value class.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_aligned_value = cpp::or_u<is_dyn_matrix<T>::value, is_fast_matrix<T>::value>;

/*!
 * \brief Traits to test if all the given ETL expresion types are padded.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_padded = cpp::and_u<decay_traits<E>::is_padded...>;

/*!
 * \brief Simple utility traits indicating if a light subview can be created out
 * of this type.
 */
template <typename T>
using fast_sub_view_able = cpp::and_u<has_direct_access<T>::value, decay_traits<T>::storage_order == order::RowMajor>;

/*!
 * \brief Simple utility traits indicating if a light slice view can be created out
 * of this type.
 */
template <typename T>
using fast_slice_view_able = fast_sub_view_able<T>;

/*!
 * \brief Traits to test if an expression is inplace transpose-able
 * \tparam T The type to test
 */
template <typename T, typename Enable = void>
struct inplace_transpose_able;

/*!
 * \copydoc inplace_transpose_able
 */
template <typename T>
struct inplace_transpose_able<T, std::enable_if_t<all_fast<T>::value && is_2d<T>::value>> {
    /*!
     * \brief Indicates if T is inplace transpose-able
     */
    static constexpr bool value = decay_traits<T>::template dim<0>() == decay_traits<T>::template dim<1>();
};

/*!
 * \copydoc inplace_transpose_able
 */
template <typename T>
struct inplace_transpose_able<T, std::enable_if_t<!all_fast<T>::value && is_2d<T>::value>> {
    /*!
     * \brief Indicates if T is inplace transpose-able
     */
    static constexpr bool value = true;
};

/*!
 * \copydoc inplace_transpose_able
 */
template <typename T>
struct inplace_transpose_able<T, std::enable_if_t<!is_2d<T>::value>> {
    /*!
     * \brief Indicates if T is inplace transpose-able
     */
    static constexpr bool value = false;
};

/*!
 * \brief Traits to test if an expression is inplace sub transpose-able.
 *
 * Sub-transpose able means that the last two dimensions can be transposed in place.
 *
 * \tparam T The type to test
 */
template <typename T, typename Enable = void>
struct inplace_sub_transpose_able;

/*!
 * \copydoc inplace_sub_transpose_able
 */
template <typename T>
struct inplace_sub_transpose_able<T, std::enable_if_t<all_fast<T>::value && is_3d<T>::value>> {
    /*!
     * \brief Indicates if T is inplace sub-transpose-able
     */
    static constexpr bool value = decay_traits<T>::template dim<1>() == decay_traits<T>::template dim<2>();
};

/*!
 * \copydoc inplace_sub_transpose_able
 */
template <typename T>
struct inplace_sub_transpose_able<T, std::enable_if_t<!all_fast<T>::value && is_3d<T>::value>> {
    /*!
     * \brief Indicates if T is inplace sub-transpose-able
     */
    static constexpr bool value = true;
};

/*!
 * \copydoc inplace_sub_transpose_able
 */
template <typename T>
struct inplace_sub_transpose_able<T, std::enable_if_t<!is_3d<T>::value>> {
    /*!
     * \brief Indicates if T is inplace sub-transpose-able
     */
    static constexpr bool value = false;
};


/*!
 * \brief Traits to test if a matrix is a square matrix, if this can be defined.
 */
template <typename Matrix, typename Enable = void>
struct is_square_matrix {
    /*!
     * \brief The value of the traits. True if the matrix is square, false otherwise
     */
    static constexpr bool value = false;
};

/*!
 * \copydoc is_square_matrix
 */
template <typename Matrix>
struct is_square_matrix <Matrix, std::enable_if_t<all_fast<Matrix>::value && is_2d<Matrix>::value>> {
    /*!
     * \brief The value of the traits. True if the matrix is square, false otherwise
     */
    static constexpr bool value = etl_traits<Matrix>::template dim<0>() == etl_traits<Matrix>::template dim<1>();
};

/*!
 * \copydoc is_square_matrix
 */
template <typename Matrix>
struct is_square_matrix <Matrix, std::enable_if_t<!all_fast<Matrix>::value && is_2d<Matrix>::value>> {
    /*!
     * \brief The value of the traits. True if the matrix is square, false otherwise
     */
    static constexpr bool value = true;
};

/*!
 * Builder to construct the type returned by a view.
 */
template <typename T, typename S>
using return_helper =
    std::conditional_t<
        std::is_const<std::remove_reference_t<S>>::value,
        const value_t<T>&,
        std::conditional_t<
            cpp::and_u<
                std::is_lvalue_reference<S>::value,
                cpp::not_u<std::is_const<T>::value>::value>::value,
            value_t<T>&,
            value_t<T>>>;

/*!
 * Builder to construct the const type returned by a view.
 */
template <typename T, typename S>
using const_return_helper = std::conditional_t<
    std::is_lvalue_reference<S>::value,
    const value_t<T>&,
    value_t<T>>;

/*!
 * \brief Specialization for value structures
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<is_etl_value_class<T>::value>> {
    using value_type = typename T::value_type; ///< The value type of the expression

    static constexpr bool is_etl                  = true;                                                        ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                                                       ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                                                       ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                                       ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = is_fast_matrix<T>::value || is_custom_fast_matrix<T>::value; ///< Indicates if the expression is fast
    static constexpr bool is_value                = true;                                                        ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = !is_sparse_matrix<T>::value;                                 ///< Indicates if the expression has direct memory access
    static constexpr bool is_thread_safe          = true;                                                        ///< Indicates if the expression is thread safe
    static constexpr bool is_linear               = true;                                                        ///< Indicates if the expression is linear
    static constexpr bool is_generator            = false;                                                       ///< Indicates if the expression is a generator expression
    static constexpr bool needs_evaluator_visitor = false;                                                       ///< Indicates if the expression needs an evaluator visitor
    static constexpr bool is_padded               = is_padded_value<T>::value;                                   ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = is_aligned_value<T>::value;                                  ///< Indicates if the expression is aligned
    static constexpr order storage_order          = T::storage_order;                                            ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = cpp::bool_constant<
                    get_intrinsic_traits<V>::template type<value_type>::vectorizable
                &&  !is_sparse_matrix<T>::value
            >;

    /*!
     * \brief Return the size of the given epxression
     * \param v The expression to get the size from
     * \return the size of the given expressio
     */
    static std::size_t size(const T& v) {
        return v.size();
    }

    /*!
     * \brief Return the dth dimension of the given epxression
     * \param v The expression to get the size from
     * \param d The dimension to get
     * \return the dth dimension of the given expressio
     */
    static std::size_t dim(const T& v, std::size_t d) {
        return v.dim(d);
    }

    /*!
     * \brief Return the size of an expression of the given type
     * \return the size of an expression of the given type
     */
    static constexpr std::size_t size() {
        static_assert(is_fast, "Only fast_matrix have compile-time access to the dimensions");

        return T::size();
    }

    /*!
     * \brief Return the Dth dimension of the given epxression
     * \tparam D The dimension to get
     * \return the Dth dimension of the given expressio
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        static_assert(is_fast, "Only fast_matrix have compile-time access to the dimensions");

        return T::template dim<D>();
    }

    /*!
     * \brief Return the number of dimensions of an expression of the given type
     * \return the number of dimensions of an expression of the given type
     */
    static constexpr std::size_t dimensions() {
        return T::n_dimensions;
    }
};

/*!
 * \brief Return the number of dimensions of the given ETL expression
 * \param expr The expression to get the number of dimensions for
 * \return The number of dimensions of the given expression.
 */
template <typename E>
constexpr std::size_t dimensions(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::dimensions();
}

/*!
 * \brief Return the number of dimensions of the given ETL type
 * \tparam E The expression type to get the number of dimensions for
 * \return The number of dimensions of the given type.
 */
template <typename E>
constexpr std::size_t dimensions() noexcept {
    return decay_traits<E>::dimensions();
}

/*!
 * \brief Returns the number of rows of the given ETL expression.
 * \param expr The expression to get the number of rows from.
 * \return The number of rows of the given expression.
 */
template <typename E, cpp_disable_if(decay_traits<E>::is_fast)>
std::size_t rows(const E& expr) {
    return etl_traits<E>::dim(expr, 0);
}

/*!
 * \brief Returns the number of rows of the given ETL expression.
 * \param expr The expression to get the number of rows from.
 * \return The number of rows of the given expression.
 */
template <typename E, cpp_enable_if(decay_traits<E>::is_fast)>
constexpr std::size_t rows(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::template dim<0>();
}

/*!
 * \brief Returns the number of columns of the given ETL expression.
 * \param expr The expression to get the number of columns from.
 * \return The number of columns of the given expression.
 */
template <typename E, cpp_disable_if(decay_traits<E>::is_fast)>
std::size_t columns(const E& expr) {
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return etl_traits<E>::dim(expr, 1);
}

/*!
 * \brief Returns the number of columns of the given ETL expression.
 * \param expr The expression to get the number of columns from.
 * \return The number of columns of the given expression.
 */
template <typename E, cpp_enable_if(decay_traits<E>::is_fast)>
constexpr std::size_t columns(const E& expr) noexcept {
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return (void)expr, etl_traits<E>::template dim<1>();
}

/*!
 * \brief Returns the size of the given ETL expression.
 * \param expr The expression to get the size from.
 * \return The size of the given expression.
 */
template <typename E, cpp_disable_if(decay_traits<E>::is_fast)>
std::size_t size(const E& expr) {
    return etl_traits<E>::size(expr);
}

/*!
 * \brief Returns the size of the given ETL expression.
 * \param expr The expression to get the size from.
 * \return The size of the given expression.
 */
template <typename E, cpp_enable_if(decay_traits<E>::is_fast)>
constexpr std::size_t size(const E& expr) noexcept {
    return (void)expr, etl_traits<E>::size();
}

/*!
 * \brief Returns the sub-size of the given ETL expression, i.e. the size not considering the first dimension.
 * \param expr The expression to get the sub-size from.
 * \return The sub-size of the given expression.
 */
template <typename E, cpp_disable_if(decay_traits<E>::is_fast)>
std::size_t subsize(const E& expr) {
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return etl_traits<E>::size(expr) / etl_traits<E>::dim(expr, 0);
}

/*!
 * \brief Returns the sub-size of the given ETL expression, i.e. the size not considering the first dimension.
 * \param expr The expression to get the sub-size from.
 * \return The sub-size of the given expression.
 */
template <typename E, cpp_enable_if(decay_traits<E>::is_fast)>
constexpr std::size_t subsize(const E& expr) noexcept {
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return (void)expr, etl_traits<E>::size() / etl_traits<E>::template dim<0>();
}

/*!
 * \brief Return the D dimension of e
 * \param e The expression to get the dimensions from
 * \tparam D The dimension to get
 * \return the Dth dimension of e
 */
template <std::size_t D, typename E, cpp_disable_if(decay_traits<E>::is_fast)>
std::size_t dim(const E& e) noexcept {
    return etl_traits<E>::dim(e, D);
}

/*!
 * \brief Return the d dimension of e
 * \param e The expression to get the dimensions from
 * \param d The dimension to get
 * \return the dth dimension of e
 */
template <typename E>
std::size_t dim(const E& e, std::size_t d) noexcept {
    return etl_traits<E>::dim(e, d);
}

/*!
 * \brief Return the D dimension of e
 * \param e The expression to get the dimensions from
 * \tparam D The dimension to get
 * \return the Dth dimension of e
 */
template <std::size_t D, typename E, cpp_enable_if(decay_traits<E>::is_fast)>
constexpr std::size_t dim(const E& e) noexcept {
    return (void)e, etl_traits<E>::template dim<D>();
}

/*!
 * \brief Return the D dimension of E
 * \return the Dth dimension of E
 */
template <std::size_t D, typename E, cpp_enable_if(decay_traits<E>::is_fast)>
constexpr std::size_t dim() noexcept {
    return decay_traits<E>::template dim<D>();
}

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E, typename Enable = void>
struct safe_dimensions;

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E>
struct safe_dimensions<E, std::enable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<size_t, std::numeric_limits<size_t>::max()> {};

/*!
 * \brief Utility to get the dimensions of an expressions, with support for generator
 */
template <typename E>
struct safe_dimensions<E, cpp::disable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<size_t, etl_traits<E>::dimensions()> {};

/*!
 * \brief Convert a flat index into a 2D index
 * \param sub The matrix expression
 * \param i The flat index
 * \return a pair of indices for the equivalent 2D index
 */
template <typename E>
constexpr std::pair<std::size_t, std::size_t> index_to_2d(E&& sub, std::size_t i) {
    return decay_traits<E>::storage_order == order::RowMajor
               ? std::make_pair(i / dim<0>(sub), i % dim<0>(sub))
               : std::make_pair(i % dim<0>(sub), i / dim<0>(sub));
}

/*!
 * \brief Returns the row stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the row stride of the given ETL matrix expression
 */
template <typename E>
std::size_t row_stride(E&& expr) {
    static_assert(decay_traits<E>::dimensions() == 2, "row_stride() only makes sense on 2D matrices");
    return decay_traits<E>::storage_order == order::RowMajor
               ? etl::dim<1>(expr)
               : 1;
}

/*!
 * \brief Returns the column stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the column stride of the given ETL matrix expression
 */
template <typename E>
std::size_t col_stride(E&& expr) {
    static_assert(decay_traits<E>::dimensions() == 2, "col_stride() only makes sense on 2D matrices");
    return decay_traits<E>::storage_order == order::RowMajor
               ? 1
               : etl::dim<0>(expr);
}

/*!
 * \brief Returns the minor stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the minor stride of the given ETL matrix expression
 */
template <typename E>
std::size_t minor_stride(E&& expr) {
    static_assert(decay_traits<E>::dimensions() == 2, "minor_stride() only makes sense on 2D matrices");
    return decay_traits<E>::storage_order == order::RowMajor
               ? etl::dim<0>(expr)
               : etl::dim<1>(expr);
}

/*!
 * \brief Returns the major stride of the given ETL matrix expression
 * \param expr The ETL expression.
 * \return the major stride of the given ETL matrix expression
 */
template <typename E>
std::size_t major_stride(E&& expr) {
    static_assert(decay_traits<E>::dimensions() == 2, "major_stride() only makes sense on 2D matrices");
    return decay_traits<E>::storage_order == order::RowMajor
               ? etl::dim<1>(expr)
               : etl::dim<0>(expr);
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

    return reinterpret_cast<uintptr_t>(a_begin) < reinterpret_cast<uintptr_t>(b_end) && reinterpret_cast<uintptr_t>(a_end) > reinterpret_cast<uintptr_t>(b_begin);
}

/*!
 * \brief Ensure that the CPU is up to date. If the expression does
 * not have direct memory access, has no effect.
 *
 * \param expr The expression
 */
template <typename E, cpp_enable_if(all_dma<E>::value)>
void safe_ensure_cpu_up_to_date(E&& expr){
    expr.ensure_cpu_up_to_date();
}

/*!
 * \brief Ensure that the CPU is up to date. If the expression does
 * not have direct memory access, has no effect.
 *
 * \param expr The expression
 */
template <typename E, cpp_disable_if(all_dma<E>::value)>
void safe_ensure_cpu_up_to_date(E&& expr){
    cpp_unused(expr);
}

/*!
 * \brief Indicates if the GPU memory is up to date. If the expression does
 * not have direct memory access, return false
 *
 * \param expr The expression
 */
template <typename E, cpp_enable_if(all_dma<E>::value)>
bool safe_is_gpu_up_to_date(E&& expr){
    return expr.is_gpu_up_to_date();
}

/*!
 * \brief Indicates if the GPU memory is up to date. If the expression does
 * not have direct memory access, return false
 *
 * \param expr The expression
 */
template <typename E, cpp_disable_if(all_dma<E>::value)>
bool safe_is_gpu_up_to_date(E&& expr){
    cpp_unused(expr);
    return false;
}

} //end of namespace etl
