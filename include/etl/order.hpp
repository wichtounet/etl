//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Storage order of a matrix
 */
enum class order {
    RowMajor,   ///< Row-Major storage
    ColumnMajor ///< Column-Major storage
};

/*!
 * \brief Reverse the given storage order.
 * \param o The order to reverse
 * \return the reversed equivalent storage order
 */
constexpr order reverse(order o) {
    return o == order::RowMajor ? order::ColumnMajor : order::RowMajor;
}

} //end of namespace etl
