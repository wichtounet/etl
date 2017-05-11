//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace vec {

/*!
 * \brief Compute the bias addition of b into x and store the result in y
 * \param x The a expression
 * \param b The b expression
 * \param y The c expression
 */
template <typename V, typename L, typename R, typename C>
void bias_add(const L& x, const R& b, C&& y) {
    for (size_t i = 0; i < etl::dim<0>(x); ++i) {
        for (size_t j = 0; j < etl::dim<1>(x); ++j) {
            for (size_t k = 0; k < etl::dim<2>(x); ++k) {
                for (size_t l = 0; l < etl::dim<3>(x); ++l) {
                    y(i, j, k, l) = x(i, j, k, l) + b(j);
                }
            }
        }
    }

    y.invalidate_gpu();
}

/*!
 * \brief Compute the bias addition of b into x and store the result in y
 * \param x The a expression
 * \param b The b expression
 * \param y The c expression
 */
template <typename A, typename B, typename C>
void bias_add(const A& x, const B& b, C&& y) {
    bias_add<default_vec>(x, b, y);
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
