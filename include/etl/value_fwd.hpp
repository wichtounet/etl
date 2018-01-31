//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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

#if __cpp_aligned_new >= 201606
#include "cpp_utils/soft_aligned_array.hpp"
#else
#include "cpp_utils/aligned_array.hpp"
#endif

#include "etl/util/aligned_vector.hpp"

namespace etl {

/*!
 * \brief Compute the real size to allocate for a vector of the
 * given size and type
 * \param size The number of elements of the vector
 * \tparam T The type contained in the vector
 * \return The size to be allocated
 */
template <typename T>
static constexpr size_t alloc_size_vec(size_t size) {
    return padding
        ? size + (size % default_intrinsic_traits<T>::size == 0 ? 0 : (default_intrinsic_traits<T>::size - (size % default_intrinsic_traits<T>::size)))
        : size;
}

#ifdef ETL_ADVANCED_PADDING

/*!
 * \brief Compute the real allocated size for a 2D matrix
 * \tparam T the type of the elements of the matrix
 * \param size The size of the matrix
 * \param last The last dimension of the matrix
 * \return the allocated size for the matrix
 */
template <typename T>
static constexpr size_t alloc_size_mat(size_t size, size_t last) {
    return size == 0 ? 0 :
        (padding
        ? (size / last) * (last + (last % default_intrinsic_traits<T>::size == 0 ? 0 : (default_intrinsic_traits<T>::size - last % default_intrinsic_traits<T>::size)))
        : size);
}

#else

/*!
 * \brief Compute the real allocated size for a 2D matrix
 * \tparam T the type of the elements of the matrix
 * \param size The size of the matrix
 * \param last The last dimension of the matrix
 * \return the allocated size for the matrix
 */
template <typename T>
static constexpr size_t alloc_size_mat(size_t size, size_t last) {
    return (void) last, (size == 0 ? 0 : alloc_size_vec<T>(size));
}

#endif

/*!
 * \brief Compute the real allocated size for a matrix
 * \tparam T the type of the elements of the matrix
 * \tparam Dims The dimensions of the matrix
 * \return the allocated size for the matrix
 */
template <typename T, size_t... Dims>
static constexpr size_t alloc_size_mat() {
    return alloc_size_mat<T>((Dims * ...), nth_size<sizeof...(Dims) - 1, 0, Dims...>);
}

template <typename T, typename ST, order SO, size_t... Dims>
struct fast_matrix_impl;

template <typename T, typename ST, order SO, size_t... Dims>
struct custom_fast_matrix_impl;

template <typename T, order SO, size_t D = 2>
struct dyn_matrix_impl;

template <typename T, order SO, size_t D = 2>
struct gpu_dyn_matrix_impl;

template <typename T, order SO, size_t D = 2>
struct custom_dyn_matrix_impl;

template <typename T, sparse_storage SS, size_t D>
struct sparse_matrix_impl;

template <typename Stream>
struct serializer;

template <typename Stream>
struct deserializer;

/*!
 * \brief Symmetric matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct symmetric_matrix;

/*!
 * \brief Hermitian matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct hermitian_matrix;

/*!
 * \brief Diagonal matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct diagonal_matrix;

/*!
 * \brief Upper triangular matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct upper_matrix;

/*!
 * \brief Strictly upper triangular matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct strictly_upper_matrix;

/*!
 * \brief Uni upper triangular matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct uni_upper_matrix;

/*!
 * \brief Lower triangular matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct lower_matrix;

/*!
 * \brief Strictly lower triangular matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct strictly_lower_matrix;

/*!
 * \brief Uni lower triangular matrix adapter
 * \tparam Matrix The adapted matrix
 */
template <typename Matrix>
struct uni_lower_matrix;

/*
 * In C++17, aligned dynamic allocation of over-aligned type is now supported,
 * so we use the soft_aligned_array.
 *
 * When this is not possible, we use the version with internal padding, but this
 * has a big data overhead.
 */

#if __cpp_aligned_new >= 201606
template <typename T, std::size_t S, std::size_t A>
using aligned_array = cpp::soft_aligned_array<T, S, A>;
#else
template <typename T, std::size_t S, std::size_t A>
using aligned_array = cpp::aligned_array<T, S, A>;
#endif

/*!
 * \brief A static matrix with fixed dimensions, in row-major order
 */
template <typename T, size_t... Dims>
using fast_matrix = fast_matrix_impl<T, aligned_array<T, alloc_size_mat<T, Dims...>(), default_intrinsic_traits<T>::alignment>, order::RowMajor, Dims...>;

/*!
 * \brief A static matrix with fixed dimensions, in column-major order
 */
template <typename T, size_t... Dims>
using fast_matrix_cm = fast_matrix_impl<T, aligned_array<T, alloc_size_mat<T, Dims...>(), default_intrinsic_traits<T>::alignment>, order::ColumnMajor, Dims...>;

/*!
 * \brief A static vector with fixed dimensions, in row-major order
 */
template <typename T, size_t Rows>
using fast_vector = fast_matrix_impl<T, aligned_array<T, alloc_size_vec<T>(Rows), default_intrinsic_traits<T>::alignment>, order::RowMajor, Rows>;

/*!
 * \brief A static vector with fixed dimensions, in column-major order
 */
template <typename T, size_t Rows>
using fast_vector_cm = fast_matrix_impl<T, aligned_array<T, alloc_size_vec<T>(Rows), default_intrinsic_traits<T>::alignment>, order::ColumnMajor, Rows>;

/*!
 * \brief A hybrid vector with fixed dimensions, in row-major order
 */
template <typename T, size_t Rows>
using fast_dyn_vector = fast_matrix_impl<T, etl::aligned_vector<T, default_intrinsic_traits<T>::alignment>, order::RowMajor, Rows>;

/*!
 * \brief A hybrid matrix with fixed dimensions, in row-major order
 */
template <typename T, size_t... Dims>
using fast_dyn_matrix = fast_matrix_impl<T, etl::aligned_vector<T, default_intrinsic_traits<T>::alignment>, order::RowMajor, Dims...>;

/*!
 * \brief A hybrid matrix with fixed dimensions, in specified  order
 */
template <typename T, order SO, size_t... Dims>
using fast_dyn_matrix_o = fast_matrix_impl<T, etl::aligned_vector<T, default_intrinsic_traits<T>::alignment>, SO, Dims...>;

/*!
 * \brief A dynamic matrix, in row-major order, of D dimensions
 */
template <typename T, size_t D = 2>
using dyn_matrix = dyn_matrix_impl<T, order::RowMajor, D>;

/*!
 * \brief A dynamic matrix, in column-major order, of D dimensions
 */
template <typename T, size_t D = 2>
using dyn_matrix_cm = dyn_matrix_impl<T, order::ColumnMajor, D>;

/*!
 * \brief A dynamic matrix, in specific storage order, of D dimensions
 */
template <typename T, order SO, size_t D = 2>
using dyn_matrix_o = dyn_matrix_impl<T, SO, D>;

/*!
 * \brief A dynamic vector, in row-major order
 */
template <typename T>
using dyn_vector = dyn_matrix_impl<T, order::RowMajor, 1>;

/*!
 * \brief A dynamic vector, in column-major order
 */
template <typename T>
using dyn_vector_cm = dyn_matrix_impl<T, order::ColumnMajor, 1>;

/*!
 * \brief A GPU dynamic matrix, in row-major order, of D dimensions
 */
template <typename T, size_t D = 2>
using gpu_dyn_matrix = gpu_dyn_matrix_impl<T, order::RowMajor, D>;

/*!
 * \brief A dynamic matrix, in row-major order, of D dimensions
 */
template <typename T, size_t D = 2>
using custom_dyn_matrix                    = custom_dyn_matrix_impl<T, order::RowMajor, D>;

/*!
 * \brief A dynamic matrix, in column-major order, of D dimensions
 */
template <typename T, size_t D = 2>
using custom_dyn_matrix_cm                 = custom_dyn_matrix_impl<T, order::ColumnMajor, D>;

/*!
 * \brief A dynamic vector, in row-major order
 */
template <typename T>
using custom_dyn_vector = custom_dyn_matrix_impl<T, order::RowMajor, 1>;

/*!
 * \brief A hybrid vector with fixed dimensions, in row-major order
 */
template <typename T, size_t Rows>
using custom_fast_vector = custom_fast_matrix_impl<T, cpp::array_wrapper<T>, order::RowMajor, Rows>;

/*!
 * \brief A hybrid matrix with fixed dimensions, in row-major order
 */
template <typename T, size_t... Dims>
using custom_fast_matrix = custom_fast_matrix_impl<T, cpp::array_wrapper<T>, order::RowMajor, Dims...>;

/*!
 * \brief A sparse matrix, of D dimensions
 */
template <typename T, size_t D = 2>
using sparse_matrix                 = sparse_matrix_impl<T, sparse_storage::COO, D>;

} //end of namespace etl
