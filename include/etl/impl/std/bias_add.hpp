//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the bias_add computation
 */

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Compute the bias addition of a and b and store the result in c
 * \param lhs The a expression
 * \param rhs The b expression
 * \param c The c expression
 */
template <typename A, typename B, typename C>
void bias_add_4d(const A& lhs, const B& rhs, C&& c) {
    for (size_t i = 0; i < etl::dim<0>(lhs); ++i) {
        for (size_t j = 0; j < etl::dim<1>(lhs); ++j) {
            for (size_t k = 0; k < etl::dim<2>(lhs); ++k) {
                for (size_t l = 0; l < etl::dim<3>(lhs); ++l) {
                    c(i, j, k, l) = lhs(i, j, k, l) + rhs(j);
                }
            }
        }
    }
}

/*!
 * \brief Compute the bias addition of a and b and store the result in c
 * \param lhs The a expression
 * \param rhs The b expression
 * \param c The c expression
 */
template <typename A, typename B, typename C>
void bias_add_2d(const A& lhs, const B& rhs, C&& c) {
    for (size_t i = 0; i < etl::dim<0>(lhs); ++i) {
        for (size_t j = 0; j < etl::dim<1>(lhs); ++j) {
            c(i, j) = lhs(i, j) + rhs(j);
        }
    }
}

} //end of namespace etl::impl::standard
