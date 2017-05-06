//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains all the operators and functions to build expressions.
*/

#pragma once

#include "etl/expression_helpers.hpp"

namespace etl {

/*!
 * \brief Creates an expression representing the Inverse of the given expression
 * \param a The input expression
 * \return an expression representing the 1D FFT of a
 */
template <typename A>
inv_expr<A> inv(A&& a) {
    static_assert(is_etl_expr<A>::value, "Inverse only supported for ETL expressions");

    return inv_expr<A>{a};
}

} //end of namespace etl
