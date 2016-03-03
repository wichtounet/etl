//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the "dot" reduction
 */

#pragma once

namespace etl {

namespace impl {

namespace standard {

/*!
 * \brief Compute the dot product of a and b
 * \param a The lhs expression
 * \param b The rhs expression
 * \return the sum
 */
template <typename A, typename B>
auto dot(const A& a, const B& b) {
    return sum(scale(a, b));
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
