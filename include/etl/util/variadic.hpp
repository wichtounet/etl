//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brie Contains the safe_cast function overloads.
 *
 * This function helps generic code in the BLAS/CUBLAS wrappers to
 * be able to convert types from etl::complex to std::complex
 * easily.
 */

#pragma once

namespace etl {

namespace util {

/*!
 * \brief Returns the size of a matrix given its dimensions
 */
inline size_t size(size_t first) {
    return first;
}

/*!
 * \brief Returns the size of a matrix given its dimensions
 */
template <typename... T>
inline size_t size(size_t first, T... args) {
    return first * size(args...);
}

/*!
 * \brief Returns the size of a matrix given its dimensions
 */
template <size_t... I, typename... T>
inline size_t size(const std::index_sequence<I...>& /*i*/, const T&... args) {
    return size((cpp::nth_value<I>(args...))...);
}

} //end of namespace util

} //end of namespace etl
