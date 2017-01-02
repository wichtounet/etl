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

//Note: Version with sizes moved to a std::array and accessed with
//standard loop may be faster, but need some stack space (relevant ?)

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

// Dynamic index (row major)

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
