//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
template <typename M, size_t I, typename Enable = void>
struct matrix_subsize : std::integral_constant<size_t, decay_traits<M>::template dim<I + 1>() * matrix_subsize<M, I + 1>::value> {};

/*!
 * \copydoc matrix_subsize
 */
template <typename M, size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == decay_traits<M>::dimensions() - 1>> : std::integral_constant<size_t, 1> {};

template <typename M, size_t I>
inline constexpr size_t matrix_subsize_v = matrix_subsize<M, I>::value;

/*!
 * \brief Traits to compute the leading sze from index I for a matrix.
 *
 * The leading sze is used  for column-major index computation.
 *
 * \tparam M The matrix to get sub size from
 * \tparam I The index we need subsize for
 */
template <typename M, size_t I, typename Enable = void>
struct matrix_leadingsize : std::integral_constant<size_t, decay_traits<M>::template dim<I - 1>() * matrix_leadingsize<M, I - 1>::value> {};

/*!
 * \copydoc matrix_leadingsize
 */
template <typename M>
struct matrix_leadingsize<M, 0> : std::integral_constant<size_t, 1> {};

template <typename M, size_t I>
inline constexpr size_t matrix_leadingsize_v = matrix_leadingsize<M, I>::value;

/*!
 * \brief Compute the index inside the row major matrix
 */
template <typename M, size_t I>
constexpr size_t rm_compute_index(size_t first) noexcept(assert_nothrow) {
    cpp_assert(first < decay_traits<M>::template dim<I>(), "Out of bounds");
    return first;
}

/*!
 * \brief Compute the index inside the row major matrix
 */
template <typename M, size_t I, typename... S>
constexpr size_t rm_compute_index(size_t first, size_t second, S... args) noexcept(assert_nothrow) {
    cpp_assert(first < decay_traits<M>::template dim<I>(), "Out of bounds");
    return matrix_subsize_v<M, I> * first + rm_compute_index<M, I + 1>(second, args...);
}

/*!
 * \brief Compute the index inside the column major matrix
 */
template <typename M, size_t I>
constexpr size_t cm_compute_index(size_t first) noexcept(assert_nothrow) {
    cpp_assert(first < M::template dim<I>(), "Out of bounds");
    return matrix_leadingsize_v<M, I> * first;
}

/*!
 * \brief Compute the index inside the column major matrix
 */
template <typename M, size_t I, typename... S>
constexpr size_t cm_compute_index(size_t first, size_t second, S... args) noexcept(assert_nothrow) {
    cpp_assert(first < M::template dim<I>(), "Out of bounds");
    return matrix_leadingsize_v<M, I> * first + cm_compute_index<M, I + 1>(second, args...);
}

} // end of namespace detail

//Note: Version with sizes moved to a std::array and accessed with
//standard loop may be faster, but need some stack space (relevant ?)

/*!
 * \brief Compute the index for a 1D fast matrix
 * \param i The index to access
 * \return The flat position of (i)
 */
template <typename T>
constexpr size_t fast_index(size_t i) noexcept(assert_nothrow) {
    static_assert(is_1d<T>, "Invalid number of dimensions for fast_index");

    cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");

    return i;
}

/*!
 * \brief Compute the index for a 2D fast matrix
 * \param i The index of the first dimension to access
 * \param j The index of the second dimension to access
 * \return The flat position of (i,j)
 */
template <typename T>
constexpr size_t fast_index(size_t i, size_t j) noexcept(assert_nothrow) {
    static_assert(is_2d<T>, "Invalid number of dimensions for fast_index");

    if constexpr (decay_traits<T>::storage_order == order::RowMajor) {
        cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
        cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");

        return i * decay_traits<T>::template dim<1>() + j;
    } else {
        cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
        cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");

        return i + j * decay_traits<T>::template dim<0>();
    }
}

/*!
 * \brief Compute the index for a 3D fast matrix
 * \param i The index of the first dimension to access
 * \param j The index of the second dimension to access
 * \param k The index of the third dimension to access
 * \return The flat position of (i,j,k)
 */
template <typename T>
constexpr size_t fast_index(size_t i, size_t j, size_t k) noexcept(assert_nothrow) {
    static_assert(is_3d<T>, "Invalid number of dimensions for fast_index");

    if constexpr (decay_traits<T>::storage_order == order::RowMajor) {
        cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
        cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");
        cpp_assert(k < decay_traits<T>::template dim<2>(), "Out of bounds");

        return i * decay_traits<T>::template dim<1>() * decay_traits<T>::template dim<2>() + j * decay_traits<T>::template dim<2>() + k;
    } else {
        cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
        cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");
        cpp_assert(k < decay_traits<T>::template dim<2>(), "Out of bounds");

        return i + j * decay_traits<T>::template dim<0>() + k * decay_traits<T>::template dim<0>() * decay_traits<T>::template dim<1>();
    }
}

/*!
 * \brief Compute the index for a 4D fast matrix
 * \param i The index of the first dimension to access
 * \param j The index of the second dimension to access
 * \param k The index of the third dimension to access
 * \param l The index of the fourth dimension to access
 * \return The flat position of (i,j,k,l)
 */
template <typename T>
constexpr size_t fast_index(size_t i, size_t j, size_t k, size_t l) noexcept(assert_nothrow) {
    static_assert(is_4d<T>, "Invalid number of dimensions for fast_index");

    if constexpr (decay_traits<T>::storage_order == order::RowMajor) {
        cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
        cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");
        cpp_assert(k < decay_traits<T>::template dim<2>(), "Out of bounds");
        cpp_assert(l < decay_traits<T>::template dim<3>(), "Out of bounds");

        return i * decay_traits<T>::template dim<1>() * decay_traits<T>::template dim<2>() * decay_traits<T>::template dim<3>()
               + j * decay_traits<T>::template dim<2>() * decay_traits<T>::template dim<3>() + k * decay_traits<T>::template dim<3>() + l;
    } else {
        cpp_assert(i < decay_traits<T>::template dim<0>(), "Out of bounds");
        cpp_assert(j < decay_traits<T>::template dim<1>(), "Out of bounds");
        cpp_assert(k < decay_traits<T>::template dim<2>(), "Out of bounds");
        cpp_assert(l < decay_traits<T>::template dim<3>(), "Out of bounds");

        return i + j * decay_traits<T>::template dim<0>() + k * decay_traits<T>::template dim<0>() * decay_traits<T>::template dim<1>()
               + l * decay_traits<T>::template dim<0>() * decay_traits<T>::template dim<1>() * decay_traits<T>::template dim<2>();
    }
}

/*!
 * \brief Compute the index for a N-D fast matrix
 * \param sizes The indices to access
 * \return The flat position of (sizes...)
 */
template <typename T, typename... S, cpp_enable_iff(sizeof...(S) > 4)>
constexpr size_t fast_index(S... sizes) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == sizeof...(S), "Invalid number of dimensions for fast_index");

    if constexpr (decay_traits<T>::storage_order == order::RowMajor) {
        return detail::rm_compute_index<T, 0>(sizes...);
    } else {
        return detail::cm_compute_index<T, 0>(sizes...);
    }
}

// Dynamic index (row major)

/*!
 * \brief Compute the index for a 1D dynamic matrix
 * \param expression The matrix reference
 * \param i The index to access
 * \return The flat position of (i)
 */
template <typename T>
size_t dyn_index([[maybe_unused]] const T& expression, size_t i) noexcept(assert_nothrow) {
    static_assert(is_1d<T>, "Invalid number of dimensions for dyn_index");

    cpp_assert(i < decay_traits<T>::dim(expression, 0), "Out of bounds");

    return i;
}

/*!
 * \brief Compute the index for a 2D dynamic matrix
 * \param expression The matrix reference
 * \param i The index of the first dimension to access
 * \param j The index of the second dimension to access
 * \return The flat position of (i,j)
 */
template <typename T>
size_t dyn_index(const T& expression, size_t i, size_t j) noexcept(assert_nothrow) {
    static_assert(is_2d<T>, "Invalid number of dimensions for dyn_index");

    if constexpr (decay_traits<T>::storage_order == order::RowMajor) {
        cpp_assert(i < decay_traits<T>::dim(expression, 0), "Out of bounds");
        cpp_assert(j < decay_traits<T>::dim(expression, 1), "Out of bounds");

        return i * decay_traits<T>::dim(expression, 1) + j;
    } else {
        cpp_assert(i < decay_traits<T>::dim(expression, 0), "Out of bounds");
        cpp_assert(j < decay_traits<T>::dim(expression, 1), "Out of bounds");

        return i + j * decay_traits<T>::dim(expression, 0);
    }
}

/*!
 * \brief Compute the index for a 3D dynamic matrix
 * \param expression The matrix reference
 * \param i The index of the first dimension to access
 * \param j The index of the second dimension to access
 * \param k The index of the third dimension to access
 * \return The flat position of (i,j,k)
 */
template <typename T>
size_t dyn_index(const T& expression, size_t i, size_t j, size_t k) noexcept(assert_nothrow) {
    static_assert(is_3d<T>, "Invalid number of dimensions for dyn_index");

    if constexpr (decay_traits<T>::storage_order == order::RowMajor) {
        cpp_assert(i < decay_traits<T>::dim(expression, 0), "Out of bounds");
        cpp_assert(j < decay_traits<T>::dim(expression, 1), "Out of bounds");
        cpp_assert(k < decay_traits<T>::dim(expression, 2), "Out of bounds");

        return i * decay_traits<T>::dim(expression, 1) * decay_traits<T>::dim(expression, 2) + j * decay_traits<T>::dim(expression, 2) + k;
    } else {
        cpp_assert(i < decay_traits<T>::dim(expression, 0), "Out of bounds");
        cpp_assert(j < decay_traits<T>::dim(expression, 1), "Out of bounds");
        cpp_assert(k < decay_traits<T>::dim(expression, 2), "Out of bounds");

        return i + j * decay_traits<T>::dim(expression, 0) + k * decay_traits<T>::dim(expression, 0) * decay_traits<T>::dim(expression, 1);
    }
}

/*!
 * \brief Compute the index for a 4D dynamic matrix
 * \param expression The matrix reference
 * \param i The index of the first dimension to access
 * \param j The index of the second dimension to access
 * \param k The index of the third dimension to access
 * \param l The index of the fourth dimension to access
 * \return The flat position of (i,j,k,l)
 */
template <typename T>
size_t dyn_index(const T& expression, size_t i, size_t j, size_t k, size_t l) noexcept(assert_nothrow) {
    static_assert(is_4d<T>, "Invalid number of dimensions for dyn_index");

    if constexpr (decay_traits<T>::storage_order == order::RowMajor) {
        cpp_assert(i < decay_traits<T>::dim(expression, 0), "Out of bounds");
        cpp_assert(j < decay_traits<T>::dim(expression, 1), "Out of bounds");
        cpp_assert(k < decay_traits<T>::dim(expression, 2), "Out of bounds");
        cpp_assert(l < decay_traits<T>::dim(expression, 3), "Out of bounds");

        return i * decay_traits<T>::dim(expression, 1) * decay_traits<T>::dim(expression, 2) * decay_traits<T>::dim(expression, 3)
               + j * decay_traits<T>::dim(expression, 2) * decay_traits<T>::dim(expression, 3) + k * decay_traits<T>::dim(expression, 3) + l;
    } else {
        cpp_assert(i < decay_traits<T>::dim(expression, 0), "Out of bounds");
        cpp_assert(j < decay_traits<T>::dim(expression, 1), "Out of bounds");
        cpp_assert(k < decay_traits<T>::dim(expression, 2), "Out of bounds");
        cpp_assert(l < decay_traits<T>::dim(expression, 3), "Out of bounds");

        return i + j * decay_traits<T>::dim(expression, 0) + k * decay_traits<T>::dim(expression, 0) * decay_traits<T>::dim(expression, 1)
               + l * decay_traits<T>::dim(expression, 0) * decay_traits<T>::dim(expression, 1) * decay_traits<T>::dim(expression, 2);
    }
}

/*!
 * \brief Compute the index for a N-D dynamic matrix
 * \param expression The matrix reference
 * \param sizes The indices to access
 * \return The flat position of (sizes...)
 */
template <typename T, typename... S, cpp_enable_iff(sizeof...(S) > 4)>
size_t dyn_index(const T& expression, S... sizes) noexcept(assert_nothrow) {
    static_assert(decay_traits<T>::dimensions() == sizeof...(S), "Invalid number of dimensions for dyn_index");

    if constexpr (decay_traits<T>::storage_order == order::RowMajor) {
        size_t index   = 0;
        size_t subsize = decay_traits<T>::size(expression);
        size_t i       = 0;

        cpp::for_each_in(
            [&subsize, &index, &i, &expression](size_t s) {
                cpp_assert(s < decay_traits<T>::dim(expression, i), "Out of bounds");
                subsize /= decay_traits<T>::dim(expression, i++);
                index += subsize * s;
            },
            sizes...);

        return index;
    } else {
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
}

} //end of namespace etl
