//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the outer product
 */

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Compute the outer product of a and b and store the result in c
 * \param a The a expression
 * \param b The b expression
 * \param c The c expression
 */
template <typename A, typename B, typename C>
void outer(const A& a, const B& b, C&& c) {
    for (size_t i = 0; i < etl::dim<0>(c); ++i) {
        for (size_t j = 0; j < etl::dim<1>(c); ++j) {
            c(i, j) = a(i) * b(j);
        }
    }
}

/*!
 * \brief Compute the batch outer product of a and b and store the result in c
 * \param lhs The a expression
 * \param rhs The b expression
 * \param c The c expression
 */
template <typename A, typename B, typename C>
void batch_outer(const A& lhs, const B& rhs, C&& c) {
    c = 0;

    for (size_t b = 0; b < etl::dim<0>(lhs); ++b) {
        for (size_t i = 0; i < etl::dim<0>(c); ++i) {
            for (size_t j = 0; j < etl::dim<1>(c); ++j) {
                c(i, j) += lhs(b, i) * rhs(b, j);
            }
        }
    }
}

} //end of namespace etl::impl::standard
