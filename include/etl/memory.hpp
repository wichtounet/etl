//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard memory utilities
*/

#pragma once

namespace etl {

/*!
 * \brief Performs a direct memory copy
 * \param first pointer to the first element to copy
 * \param last pointer to the next-to-last element to copy
 * \param target pointer to the first element of the result
 */
template <typename S, typename T>
void direct_copy(const S* first, const S* last, T* target) {
    std::copy(first, last, target);
}

/*!
 * \brief Performs a direct memory copy
 * \param source pointer to the first source element
 * \param target pointer to the first element of the result
 * \param n The number of elements to copy
 */
template <typename S, typename T>
void direct_copy_n(const S* source, T* target, std::size_t n) {
    std::copy_n(source, n, target);
}

} //end of namespace etl
