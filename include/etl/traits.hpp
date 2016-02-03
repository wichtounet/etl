//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
struct is_dyn_matrix_impl : std::false_type {};

template <typename V1, order V2, std::size_t V3>
struct is_dyn_matrix_impl<dyn_matrix_impl<V1, V2, V3>> : std::true_type {};

template <typename T>
struct is_sparse_matrix_impl : std::false_type {};

template <typename V1, sparse_storage V2, std::size_t V3>
struct is_sparse_matrix_impl<sparse_matrix_impl<V1, V2, V3>> : std::true_type {};

} //end of namespace traits_detail

/*!
 * \brief Traits to get information about ETL types
 *
 * For non-ETL types, is_etl is false and in that case, no other fields should be used on the traits.
 *
 * \tparam T the type to introspect
 */
template <typename T, typename Enable = void>
struct etl_traits {
    static constexpr const bool is_etl         = false; ///< Indicates if T is an ETL type
    static constexpr const bool is_transformer = false; ///< Indicates if T is a transformer
    static constexpr const bool is_view        = false; ///< Indicates if T is a view
    static constexpr const bool is_magic_view  = false; ///< Indicates if T is a magic view
    static constexpr const bool is_fast        = false; ///< Indicates if T is a fast structure
    static constexpr const bool is_generator   = false; ///< Indicates if T is a generator expression
};

/*!
 * \brief Traits helper to get information about ETL types, the type is first decayed.
 * \tparam E the type to introspect
 */
template <typename E>
using decay_traits = etl_traits<std::decay_t<E>>;

/*!
 * \brief Traits indicating if the given ETL type is a fast matrix
 * \tparam T The type to test
 */
template <typename T>
using is_fast_matrix = traits_detail::is_fast_matrix_impl<std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a dyn matrix
 * \tparam T The type to test
 */
template <typename T>
using is_dyn_matrix = traits_detail::is_dyn_matrix_impl<std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a sparse matrix
 * \tparam T The type to test
 */
template <typename T>
using is_sparse_matrix = traits_detail::is_sparse_matrix_impl<std::decay_t<T>>;

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
 * \brief Traits indicating if the given ETL type is a temporary unary expression.
 * \tparam T The type to test
 */
template <typename T>
using is_temporary_unary_expr = cpp::is_specialization_of<etl::temporary_unary_expr, std::decay_t<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a temporary binary expression.
 * \tparam T The type to test
 */
template <typename T>
using is_temporary_binary_expr = cpp::is_specialization_of<etl::temporary_binary_expr, std::decay_t<T>>;

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
 * \brief Traits indicating if the given ETL type is a value type with direct access.
 * \tparam T The type to test
 */
template <typename T>
using is_etl_direct_value = cpp::or_c<is_fast_matrix<T>, is_dyn_matrix<T>>;

/*!
 * \brief Traits indicating if the given ETL type is a value type.
 * \tparam T The type to test
 */
template <typename T>
using is_etl_value = cpp::or_c<is_fast_matrix<T>, is_dyn_matrix<T>, is_sparse_matrix<T>>;

/*!
 * \brief Traits indicating if the given ETL type can be left hand side type
 * \tparam T The type to test
 */
template <typename T>
using is_lhs = cpp::or_c<is_etl_value<T>, is_unary_expr<T>>;

/*!
 * \brief Traits indicating if the given ETL type has direct memory access.
 * \tparam T The type to test
 */
template <typename T, typename DT = std::decay_t<T>>
struct has_direct_access;

/*!
 * \brief Traits indicating if the given ETL type is a sub view with direct access to memory
 * \tparam T The type to test
 */
template <typename T>
struct is_direct_sub_view : std::false_type {};

/*!
 * \copydoc is_direct_sub_view
 */
template <typename T>
struct is_direct_sub_view<sub_view<T>> : cpp::and_u<has_direct_access<T>::value, decay_traits<T>::storage_order == order::RowMajor> {};

/*!
 * \brief Traits indicating if the given ETL type is a dim view with direct access to memory
 * \tparam T The type to test
 */
template <typename T>
struct is_direct_dim_view : std::false_type {};

template <typename T>
struct is_direct_dim_view<dim_view<T, 1>> : has_direct_access<T> {};

/*!
 * \brief Traits indicating if the given ETL type is a fast matrix view with direct access to memory
 * \tparam T The type to test
 */
template <typename T>
struct is_direct_fast_matrix_view : std::false_type {};

template <typename T, std::size_t... Dims>
struct is_direct_fast_matrix_view<fast_matrix_view<T, Dims...>> : has_direct_access<T> {};

/*!
 * \brief Traits indicating if the given ETL type is a dyn matrix view with direct access to memory
 * \tparam T The type to test
 */
template <typename T>
struct is_direct_dyn_matrix_view : std::false_type {};

template <typename T>
struct is_direct_dyn_matrix_view<dyn_matrix_view<T>> : has_direct_access<T> {};

/*!
 * \brief Traits indicating if the given ETL type is a dyn vector view with direct access to memory
 * \tparam T The type to test
 */
template <typename T>
struct is_direct_dyn_vector_view : std::false_type {};

template <typename T>
struct is_direct_dyn_vector_view<dyn_vector_view<T>> : has_direct_access<T> {};

/*!
 * \brief Traits indicating if the given ETL type is an identity view with direct access to memory
 * \tparam T The type to test
 */
template <typename T>
struct is_direct_identity_view : std::false_type {};

template <typename T, typename V>
struct is_direct_identity_view<etl::unary_expr<T, V, identity_op>> : has_direct_access<V> {};

template <typename T, typename DT>
struct has_direct_access : cpp::or_c<
                               is_etl_direct_value<DT>, is_temporary_unary_expr<DT>, is_temporary_binary_expr<DT>, is_direct_identity_view<DT>, is_direct_sub_view<DT>, is_direct_dim_view<DT>, is_direct_fast_matrix_view<DT>, is_direct_dyn_matrix_view<DT>, is_direct_dyn_vector_view<DT>> {};

/*!
 * \brief Traits to test if the given ETL expresion contains single precision numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_single_precision = std::is_same<typename std::decay_t<T>::value_type, float>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains single precision numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_single_precision = cpp::and_c<is_single_precision<E>...>;

/*!
 * \brief Traits to test if the given ETL expresion contains double precision numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_double_precision = std::is_same<typename std::decay_t<T>::value_type, double>;

/*!
 * \brief Traits to test if all the given ETL expresion types contains double precision numbers.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_double_precision = cpp::and_c<is_double_precision<E>...>;

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
using is_complex_single_precision = is_complex_single_t<typename std::decay_t<T>::value_type>;

/*!
 * \brief Traits to test if the given ETL expresion type contains double precision complex numbers.
 * \tparam T The ETL expression type.
 */
template <typename T>
using is_complex_double_precision = is_complex_double_t<typename std::decay_t<T>::value_type>;

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
 * \brief Traits to test if all the given ETL expresion types are vectorizable.
 * \tparam E The ETL expression types.
 */
template <typename... E>
using all_vectorizable = cpp::and_u<decay_traits<E>::vectorizable...>;

template <typename T, typename Enable = void>
struct inplace_transpose_able;

template <typename T>
struct inplace_transpose_able<T, std::enable_if_t<all_fast<T>::value && is_2d<T>::value>> {
    static constexpr const bool value = decay_traits<T>::template dim<0>() == decay_traits<T>::template dim<1>();
};

template <typename T>
struct inplace_transpose_able<T, std::enable_if_t<!all_fast<T>::value && is_2d<T>::value>> {
    static constexpr const bool value = true;
};

template <typename T>
struct inplace_transpose_able<T, std::enable_if_t<!is_2d<T>::value>> {
    static constexpr const bool value = false;
};

/*!
 * \brief Specialization for value structures
 */
template <typename T>
struct etl_traits<T, std::enable_if_t<is_etl_value<T>::value>> {
    static constexpr const bool is_etl                  = true;                        ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                       ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;                       ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                       ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = is_fast_matrix<T>::value;    ///< Indicates if the expression is fast
    static constexpr const bool is_value                = true;                        ///< Indicates if the expression is of value type
    static constexpr const bool is_linear               = true;                        ///< Indicates if the expression is linear
    static constexpr const bool is_generator            = false;                       ///< Indicates if the expression is a generator expression
    static constexpr const bool vectorizable            = !is_sparse_matrix<T>::value; ///< Indicates if the expression is vectorizable
    static constexpr const bool needs_temporary_visitor = false;                       ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = false;                       ///< Indicaes if the expression needs an evaluator visitor
    static constexpr const order storage_order          = T::storage_order;            ///< The expression storage order

    static std::size_t size(const T& v) {
        return v.size();
    }

    static std::size_t dim(const T& v, std::size_t d) {
        return v.dim(d);
    }

    static constexpr std::size_t size() {
        static_assert(is_fast, "Only fast_matrix have compile-time access to the dimensions");

        return T::size();
    }

    template <std::size_t D>
    static constexpr std::size_t dim() {
        static_assert(is_fast, "Only fast_matrix have compile-time access to the dimensions");

        return T::template dim<D>();
    }

    static constexpr std::size_t dimensions() {
        return T::n_dimensions;
    }
};

/*
 * \brief Return the number of dimensions of the given ETL expression
 * \param expr The expression to get the number of dimensions for
 * \return The number of dimensions of the given expression.
 */
template <typename E>
constexpr std::size_t dimensions(const E& expr) noexcept {
    return (void) expr, etl_traits<E>::dimensions();
}

/*
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
template <typename E, cpp_disable_if(etl_traits<E>::is_fast)>
std::size_t rows(const E& expr) {
    return etl_traits<E>::dim(expr, 0);
}

/*!
 * \brief Returns the number of rows of the given ETL expression.
 * \param expr The expression to get the number of rows from.
 * \return The number of rows of the given expression.
 */
template <typename E, cpp_enable_if(etl_traits<E>::is_fast)>
constexpr std::size_t rows(const E& expr) noexcept {
    return (void) expr, etl_traits<E>::template dim<0>();
}

/*!
 * \brief Returns the number of columns of the given ETL expression.
 * \param expr The expression to get the number of columns from.
 * \return The number of columns of the given expression.
 */
template <typename E, cpp_disable_if(etl_traits<E>::is_fast)>
std::size_t columns(const E& expr) {
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return etl_traits<E>::dim(expr, 1);
}

/*!
 * \brief Returns the number of columns of the given ETL expression.
 * \param expr The expression to get the number of columns from.
 * \return The number of columns of the given expression.
 */
template <typename E, cpp_enable_if(etl_traits<E>::is_fast)>
constexpr std::size_t columns(const E& expr) noexcept {
    static_assert(etl_traits<E>::dimensions() > 1, "columns() can only be used on 2D+ matrices");
    return (void) expr, etl_traits<E>::template dim<1>();
}

/*!
 * \brief Returns the size of the given ETL expression.
 * \param expr The expression to get the size from.
 * \return The size of the given expression.
 */
template <typename E, cpp_disable_if(etl_traits<E>::is_fast)>
std::size_t size(const E& expr) {
    return etl_traits<E>::size(expr);
}

/*!
 * \brief Returns the size of the given ETL expression.
 * \param expr The expression to get the size from.
 * \return The size of the given expression.
 */
template <typename E, cpp_enable_if(etl_traits<E>::is_fast)>
constexpr std::size_t size(const E& expr) noexcept {
    return (void) expr, etl_traits<E>::size();
}

/*!
 * \brief Returns the sub-size of the given ETL expression, i.e. the size not considering the first dimension.
 * \param expr The expression to get the sub-size from.
 * \return The sub-size of the given expression.
 */
template <typename E, cpp_disable_if(etl_traits<E>::is_fast)>
std::size_t subsize(const E& expr) {
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return etl_traits<E>::size(expr) / etl_traits<E>::dim(expr, 0);
}

/*!
 * \brief Returns the sub-size of the given ETL expression, i.e. the size not considering the first dimension.
 * \param expr The expression to get the sub-size from.
 * \return The sub-size of the given expression.
 */
template <typename E, cpp_enable_if(etl_traits<E>::is_fast)>
constexpr std::size_t subsize(const E& expr) noexcept {
    static_assert(etl_traits<E>::dimensions() > 1, "Only 2D+ matrices have a subsize");
    return (void) expr, etl_traits<E>::size() / etl_traits<E>::template dim<0>();
}

/*!
 * \brief Return the D dimension of e
 * \param e The expression to get the dimensions from
 * \tparam D The dimension to get
 * \return the Dth dimension of e
 */
template <std::size_t D, typename E, cpp_disable_if(etl_traits<E>::is_fast)>
std::size_t dim(const E& e) {
    return etl_traits<E>::dim(e, D);
}

/*!
 * \brief Return the d dimension of e
 * \param e The expression to get the dimensions from
 * \param d The dimension to get
 * \return the dth dimension of e
 */
template <typename E>
std::size_t dim(const E& e, std::size_t d) {
    return etl_traits<E>::dim(e, d);
}

/*!
 * \brief Return the D dimension of e
 * \param e The expression to get the dimensions from
 * \tparam D The dimension to get
 * \return the Dth dimension of e
 */
template <std::size_t D, typename E, cpp_enable_if(etl_traits<E>::is_fast)>
constexpr std::size_t dim(const E& e) noexcept {
    return (void) e, etl_traits<E>::template dim<D>();
}

/*!
 * \brief Return the D dimension of E
 * \return the Dth dimension of E
 */
template <std::size_t D, typename E, cpp_enable_if(etl_traits<E>::is_fast)>
constexpr std::size_t dim() noexcept {
    return decay_traits<E>::template dim<D>();
}

template <typename E, typename Enable = void>
struct sub_size_compare;

template <typename E>
struct sub_size_compare<E, std::enable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<std::size_t, std::numeric_limits<std::size_t>::max()> {};

template <typename E>
struct sub_size_compare<E, cpp::disable_if_t<etl_traits<E>::is_generator>> : std::integral_constant<std::size_t, etl_traits<E>::dimensions()> {};

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
template<typename P1, typename P2>
bool memory_alias(const P1* a_begin, const P1* a_end, const P2* b_begin, const P2* b_end){
    cpp_assert(a_begin <= a_end, "memory_alias works on ordered ranges");
    cpp_assert(b_begin <= b_end, "memory_alias works on ordered ranges");

    return reinterpret_cast<uintptr_t>(a_begin) < reinterpret_cast<uintptr_t>(b_end)
        && reinterpret_cast<uintptr_t>(a_end) > reinterpret_cast<uintptr_t>(b_begin);
}

} //end of namespace etl
