//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains forward declarations and using declarations for
 * the various value types.
 */

#pragma once

#include <cstddef>
#include "cpp_utils/array_wrapper.hpp"
#include "cpp_utils/aligned_vector.hpp"
#include "cpp_utils/aligned_array.hpp"

namespace etl {

template <typename T>
static constexpr size_t alloc_size_vec(size_t size) {
    return padding
        ? size + (size % default_intrinsic_traits<T>::size == 0 ? 0 : (default_intrinsic_traits<T>::size - (size % default_intrinsic_traits<T>::size)))
        : size;
}

#ifdef ETL_ADVANCED_PADDING

template <typename T>
static constexpr size_t alloc_size_mat(size_t size, size_t last) {
    return size == 0 ? 0 :
        (padding
        ? (size / last) * (last + (last % default_intrinsic_traits<T>::size == 0 ? 0 : (default_intrinsic_traits<T>::size - last % default_intrinsic_traits<T>::size)))
        : size);
}

#else

template <typename T>
static constexpr size_t alloc_size_mat(size_t size, size_t /*last*/) {
    return size == 0 ? 0 : alloc_size_vec<T>(size);
}

#endif

template <typename T, size_t... Dims>
static constexpr size_t alloc_size_mat() {
    return alloc_size_mat<T>(mul_all<Dims...>(), nth_size<sizeof...(Dims) - 1, 0, Dims...>::value);
}

template <typename T, typename ST, order SO, std::size_t... Dims>
struct fast_matrix_impl;

template <typename T, typename ST, order SO, std::size_t... Dims>
struct custom_fast_matrix_impl;

template <typename T, order SO, std::size_t D = 2>
struct dyn_matrix_impl;

template <typename T, order SO, std::size_t D = 2>
struct custom_dyn_matrix_impl;

template <typename T, sparse_storage SS, std::size_t D>
struct sparse_matrix_impl;

template <typename Stream>
struct serializer;

template <typename Stream>
struct deserializer;

template <typename Matrix>
struct sym_matrix;

/*!
 * \brief A static matrix with fixed dimensions, in row-major order
 */
template <typename T, std::size_t... Dims>
using fast_matrix = fast_matrix_impl<T, cpp::aligned_array<T, alloc_size_mat<T, Dims...>(), default_intrinsic_traits<T>::alignment>, order::RowMajor, Dims...>;

/*!
 * \brief A static matrix with fixed dimensions, in column-major order
 */
template <typename T, std::size_t... Dims>
using fast_matrix_cm = fast_matrix_impl<T, cpp::aligned_array<T, alloc_size_mat<T, Dims...>(), default_intrinsic_traits<T>::alignment>, order::ColumnMajor, Dims...>;

/*!
 * \brief A static vector with fixed dimensions, in row-major order
 */
template <typename T, std::size_t Rows>
using fast_vector = fast_matrix_impl<T, cpp::aligned_array<T, alloc_size_vec<T>(Rows), default_intrinsic_traits<T>::alignment>, order::RowMajor, Rows>;

/*!
 * \brief A static vector with fixed dimensions, in column-major order
 */
template <typename T, std::size_t Rows>
using fast_vector_cm = fast_matrix_impl<T, cpp::aligned_array<T, alloc_size_vec<T>(Rows), default_intrinsic_traits<T>::alignment>, order::ColumnMajor, Rows>;

/*!
 * \brief A hybrid vector with fixed dimensions, in row-major order
 */
template <typename T, std::size_t Rows>
using fast_dyn_vector = fast_matrix_impl<T, cpp::aligned_vector<T, default_intrinsic_traits<T>::alignment>, order::RowMajor, Rows>;

/*!
 * \brief A hybrid matrix with fixed dimensions, in row-major order
 */
template <typename T, std::size_t... Dims>
using fast_dyn_matrix = fast_matrix_impl<T, cpp::aligned_vector<T, default_intrinsic_traits<T>::alignment>, order::RowMajor, Dims...>;

/*!
 * \brief A dynamic matrix, in row-major order, of D dimensions
 */
template <typename T, std::size_t D = 2>
using dyn_matrix                    = dyn_matrix_impl<T, order::RowMajor, D>;

/*!
 * \brief A dynamic matrix, in column-major order, of D dimensions
 */
template <typename T, std::size_t D = 2>
using dyn_matrix_cm                 = dyn_matrix_impl<T, order::ColumnMajor, D>;

/*!
 * \brief A dynamic vector, in row-major order
 */
template <typename T>
using dyn_vector = dyn_matrix_impl<T, order::RowMajor, 1>;

/*!
 * \brief A dynamic matrix, in row-major order, of D dimensions
 */
template <typename T, std::size_t D = 2>
using custom_dyn_matrix                    = custom_dyn_matrix_impl<T, order::RowMajor, D>;

/*!
 * \brief A dynamic matrix, in column-major order, of D dimensions
 */
template <typename T, std::size_t D = 2>
using custom_dyn_matrix_cm                 = custom_dyn_matrix_impl<T, order::ColumnMajor, D>;

/*!
 * \brief A dynamic vector, in row-major order
 */
template <typename T>
using custom_dyn_vector = custom_dyn_matrix_impl<T, order::RowMajor, 1>;

/*!
 * \brief A hybrid vector with fixed dimensions, in row-major order
 */
template <typename T, std::size_t Rows>
using custom_fast_vector = custom_fast_matrix_impl<T, cpp::array_wrapper<T>, order::RowMajor, Rows>;

/*!
 * \brief A hybrid matrix with fixed dimensions, in row-major order
 */
template <typename T, std::size_t... Dims>
using custom_fast_matrix = custom_fast_matrix_impl<T, cpp::array_wrapper<T>, order::RowMajor, Dims...>;

/*!
 * \brief A sparse matrix, of D dimensions
 */
template <typename T, std::size_t D = 2>
using sparse_matrix                 = sparse_matrix_impl<T, sparse_storage::COO, D>;

} //end of namespace etl
