//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the "norm" reduction
 */

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Compute the euclidean norm of a
 * \param a The expression
 * \return the euclidean norm
 */
template <typename A>
value_t<A> norm(const A& a) {
    return std::sqrt(sum(scale(a, a)));
}

} //end of namespace etl::impl::standard
