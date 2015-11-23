//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file globals.hpp
 * \brief Contains some global functions.
*/

#pragma once

namespace etl {

/*!
 * \brief Indicates if the given expression is a square matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a square matrix, false otherwise.
 */
template<typename E>
bool is_square(E&& expr) {
    return decay_traits<E>::dimensions() == 2 && etl::dim<0>(expr) == etl::dim<1>(expr);
}

/*!
 * \brief Indicates if the given expression is a rectangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a rectangular matrix, false otherwise.
 */
template<typename E>
bool is_rectangular(E&& expr) {
    return decay_traits<E>::dimensions() == 2 && etl::dim<0>(expr) != etl::dim<1>(expr);
}

/*!
 * \brief Indicates if the given expression contains sub matrices that are square.
 * \param expr The expression to test
 * \return true if the given expression contains sub matrices that are square, false otherwise.
 */
template<typename E>
bool is_sub_square(E&& expr) {
    return decay_traits<E>::dimensions() == 3 && etl::dim<1>(expr) == etl::dim<2>(expr);
}

/*!
 * \brief Indicates if the given expression contains sub matrices that are rectangular.
 * \param expr The expression to test
 * \return true if the given expression contains sub matrices that are rectangular, false otherwise.
 */
template<typename E>
bool is_sub_rectangular(E&& expr) {
    return decay_traits<E>::dimensions() == 3 && etl::dim<1>(expr) != etl::dim<2>(expr);
}

} //end of namespace etl
