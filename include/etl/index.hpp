//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Base class and utilities for dyn matrix implementations
 */

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief Traits to compute the subsize from index I for a matrix.
 *
 * The subsize is used  for row-major index computation.
 *
 * \tparam M The matrix to get sub size from
 * \tparam I The index we need subsize for
 */
template <typename M, std::size_t I, typename Enable = void>
struct matrix_subsize : std::integral_constant<std::size_t, decay_traits<M>::template dim<I + 1>() * matrix_subsize<M, I + 1>::value> {};

/*!
 * \copydoc matrix_subsize
 */
template <typename M, std::size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == decay_traits<M>::dimensions() - 1>> : std::integral_constant<std::size_t, 1> {};

/*!
 * \brief Traits to compute the leading sze from index I for a matrix.
 *
 * The leading sze is used  for column-major index computation.
 *
 * \tparam M The matrix to get sub size from
 * \tparam I The index we need subsize for
 */
template <typename M, std::size_t I, typename Enable = void>
struct matrix_leadingsize : std::integral_constant<std::size_t, decay_traits<M>::template dim<I - 1>() * matrix_leadingsize<M, I - 1>::value> {};

/*!
 * \copydoc matrix_leadingsize
 */
template <typename M>
struct matrix_leadingsize<M, 0> : std::integral_constant<std::size_t, 1> {};

/*!
 * \brief Compute the index inside the row major matrix
 */
template <typename M, std::size_t I>
inline cpp14_constexpr std::size_t rm_compute_index(std::size_t first) noexcept(assert_nothrow) {
    cpp_assert(first < decay_traits<M>::template dim<I>(), "Out of bounds");
    return first;
}

/*!
 * \brief Compute the index inside the row major matrix
 */
template <typename M, std::size_t I, typename... S>
inline cpp14_constexpr std::size_t rm_compute_index(std::size_t first, std::size_t second, S... args) noexcept(assert_nothrow) {
    cpp_assert(first < decay_traits<M>::template dim<I>(), "Out of bounds");
    return matrix_subsize<M, I>::value * first + rm_compute_index<M, I + 1>(second, args...);
}

/*!
 * \brief Compute the index inside the column major matrix
 */
template <typename M, std::size_t I>
inline cpp14_constexpr std::size_t cm_compute_index(std::size_t first) noexcept(assert_nothrow) {
    cpp_assert(first < M::template dim<I>(), "Out of bounds");
    return matrix_leadingsize<M, I>::value * first;
}

/*!
 * \brief Compute the index inside the column major matrix
 */
template <typename M, std::size_t I, typename... S>
inline cpp14_constexpr std::size_t cm_compute_index(std::size_t first, std::size_t second, S... args) noexcept(assert_nothrow) {
    cpp_assert(first < M::template dim<I>(), "Out of bounds");
    return matrix_leadingsize<M, I>::value * first + cm_compute_index<M, I + 1>(second, args...);
}

}

//Note: Version with sizes moved to a std::array and accessed with
//standard loop may be faster, but need some stack space (relevant ?)

// Static index (row major)

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::RowMajor)>
cpp14_constexpr size_t fast_index(size_t i) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 1, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");

    return i;
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::RowMajor)>
cpp14_constexpr size_t fast_index(size_t i, size_t j) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 2, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
    cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");

    return i * decay_traits<T>::template dim<1>() + j;
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::RowMajor)>
cpp14_constexpr size_t fast_index(size_t i, size_t j, size_t k) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 3, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
    cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");
    cpp_assert(k < decay_traits<T>::template dim<2>(), "Out of bounds");

    return i * decay_traits<T>::template dim<1>() * decay_traits<T>::template dim<2>() + j * decay_traits<T>::template dim<2>() + k;
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::RowMajor)>
cpp14_constexpr size_t fast_index(size_t i, size_t j, size_t k, size_t l) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 4, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
    cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");
    cpp_assert(k < decay_traits<T>::template dim<2>(), "Out of bounds");
    cpp_assert(l < decay_traits<T>::template dim<3>(), "Out of bounds");

    return i * decay_traits<T>::template dim<1>() * decay_traits<T>::template dim<2>() * decay_traits<T>::template dim<3>()
        + j * decay_traits<T>::template dim<2>() * decay_traits<T>::template dim<3>() + k * decay_traits<T>::template dim<3>() + l;
}

template <typename T, typename... S, cpp_enable_if((sizeof...(S) > 4 && decay_traits<T>::storage_order == order::RowMajor))>
cpp14_constexpr size_t fast_index(S... sizes) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == sizeof...(S), "Invalid number of dimensions for fast_index");

    return detail::rm_compute_index<T, 0>(sizes...);
}

// Dynamic index (row major)

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::RowMajor)>
size_t dyn_index(const T& expression, size_t i) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 1, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < etl::dim<0>(expression), "Out of bounds");

    return i;
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::RowMajor)>
size_t dyn_index(const T& expression, size_t i, size_t j) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 2, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < etl::dim<0>(expression), "Out of bounds");
    cpp_assert(j < etl::dim<1>(expression), "Out of bounds");

    return i * etl::dim<1>(expression) + j;
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::RowMajor)>
size_t dyn_index(const T& expression, size_t i, size_t j, size_t k) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 3, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < etl::dim<0>(expression), "Out of bounds");
    cpp_assert(j < etl::dim<1>(expression), "Out of bounds");
    cpp_assert(k < etl::dim<2>(expression), "Out of bounds");

    return i * etl::dim<1>(expression) * etl::dim<2>(expression) + j * etl::dim<2>(expression) + k;
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::RowMajor)>
size_t dyn_index(const T& expression, size_t i, size_t j, size_t k, size_t l) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 4, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < etl::dim<0>(expression), "Out of bounds");
    cpp_assert(j < etl::dim<1>(expression), "Out of bounds");
    cpp_assert(k < etl::dim<2>(expression), "Out of bounds");
    cpp_assert(l < etl::dim<3>(expression), "Out of bounds");

    return i * etl::dim<1>(expression) * etl::dim<2>(expression) * etl::dim<3>(expression) + j * etl::dim<2>(expression) * etl::dim<3>(expression) + k * etl::dim<3>(expression) + l;
}

template <typename T, typename... S, cpp_enable_if((sizeof...(S) > 4 && decay_traits<T>::storage_order == order::RowMajor))>
size_t dyn_index(const T& expression, S... sizes) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == sizeof...(S), "Invalid number of dimensions for dyn_index");

    size_t index   = 0;
    size_t subsize = etl::size(expression);
    size_t i       = 0;

    cpp::for_each_in(
        [&subsize, &index, &i, &expression](size_t s) {
            cpp_assert(s < decay_traits<T>::dim(expression, i), "Out of bounds");
            subsize /= decay_traits<T>::dim(expression, i++);
            index += subsize * s;
        },
        sizes...);

    return index;
}

// Fast index (column major)

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::ColumnMajor)>
cpp14_constexpr size_t fast_index(size_t i) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 1, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");

    return i;
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::ColumnMajor)>
cpp14_constexpr size_t fast_index(size_t i, size_t j) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 2, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
    cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");

    return i + j * decay_traits<T>::template dim<0>();
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::ColumnMajor)>
cpp14_constexpr size_t fast_index(size_t i, size_t j, size_t k) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 3, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
    cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");
    cpp_assert(k < decay_traits<T>::template dim<2>(), "Out of bounds");

    return i + j * decay_traits<T>::template dim<0>() + k * decay_traits<T>::template dim<0>() * decay_traits<T>::template dim<1>();
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::ColumnMajor)>
cpp14_constexpr size_t fast_index(size_t i, size_t j, size_t k, size_t l) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 4, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
    cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");
    cpp_assert(k < decay_traits<T>::template dim<2>(), "Out of bounds");
    cpp_assert(l < decay_traits<T>::template dim<3>(), "Out of bounds");

    return i + j * decay_traits<T>::template dim<0>() + k * decay_traits<T>::template dim<0>() * decay_traits<T>::template dim<1>()
        + l * decay_traits<T>::template dim<0>() * decay_traits<T>::template dim<1>() * decay_traits<T>::template dim<2>();
}

template <typename T, typename... S, cpp_enable_if((sizeof...(S) > 4 && decay_traits<T>::storage_order == order::ColumnMajor))>
cpp14_constexpr size_t fast_index(S... sizes) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == sizeof...(S), "Invalid number of dimensions for fast_index");

    return detail::cm_compute_index<T, 0>(sizes...);
}

// Dynamic index (column major)

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::ColumnMajor)>
size_t dyn_index(const T& expression, size_t i) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 1, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < etl::dim<0>(expression), "Out of bounds");

    return i;
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::ColumnMajor)>
size_t dyn_index(const T& expression, size_t i, size_t j) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 2, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < etl::dim<0>(expression), "Out of bounds");
    cpp_assert(j < etl::dim<1>(expression), "Out of bounds");

    return i + j * etl::dim<0>(expression);
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::ColumnMajor)>
size_t dyn_index(const T& expression, size_t i, size_t j, size_t k) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 3, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < etl::dim<0>(expression), "Out of bounds");
    cpp_assert(j < etl::dim<1>(expression), "Out of bounds");
    cpp_assert(k < etl::dim<2>(expression), "Out of bounds");

    return i + j * etl::dim<0>(expression) + k * etl::dim<0>(expression) * etl::dim<1>(expression);
}

template <typename T, cpp_enable_if(decay_traits<T>::storage_order == order::ColumnMajor)>
size_t dyn_index(const T& expression, size_t i, size_t j, size_t k, size_t l) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == 4, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < etl::dim<0>(expression), "Out of bounds");
    cpp_assert(j < etl::dim<1>(expression), "Out of bounds");
    cpp_assert(k < etl::dim<2>(expression), "Out of bounds");
    cpp_assert(l < etl::dim<3>(expression), "Out of bounds");

    return i + j * etl::dim<0>(expression) + k * etl::dim<0>(expression) * etl::dim<1>(expression) + l * etl::dim<0>(expression) * etl::dim<1>(expression) * etl::dim<2>(expression);
}

template <typename T, typename... S, cpp_enable_if((sizeof...(S) > 4 && decay_traits<T>::storage_order == order::ColumnMajor))>
size_t dyn_index(const T& expression, S... sizes) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == sizeof...(S), "Invalid number of dimensions for dyn_index");

    size_t index   = 0;
    size_t subsize = 1;
    size_t i       = 0;

    cpp::for_each_in(
        [&subsize, &index, &i, &expression](size_t s) {
            cpp_assert(s < decay_traits<T>::dim(expression, i), "Out of bounds");
            index += subsize * s;
            subsize *= decay_traits<T>::dim(expression, i++);
        },
        sizes...);

    return index;
}

} //end of namespace etl
