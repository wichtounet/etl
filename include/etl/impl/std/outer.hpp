//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the outer product
 */

#pragma once

namespace etl {

namespace impl {

namespace standard {

/*!
 * \brief Compute the outer product of a and b and store the result in c
 * \param a The a expression
 * \param b The b expression
 * \param c The c expression
 */
template <typename A, typename B, typename C>
void outer(const A& a, const B& b, C&& c) {
    for (std::size_t i = 0; i < etl::dim<0>(c); ++i) {
        for (std::size_t j = 0; j < etl::dim<1>(c); ++j) {
            c(i, j) = a(i) * b(j);
        }
    }
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
