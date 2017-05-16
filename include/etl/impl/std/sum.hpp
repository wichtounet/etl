//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the "sum" reduction
 */

#pragma once

namespace etl {

namespace impl {

namespace standard {

/*!
 * \brief Compute the sum of the input in the given expression
 * \param input The input expression
 * \return the sum
 */
template <typename E>
value_t<E> sum(const E& input) {
    value_t<E> acc(0);

    for (size_t i = 0; i < size(input); ++i) {
        acc += input[i];
    }

    return acc;
}

/*!
 * \brief Compute the sum of the absolute values in the given expression
 * \param input The input expression
 * \return the absolute sum
 */
template <typename E>
value_t<E> asum(const E& input) {
    value_t<E> acc(0);

    for (size_t i = 0; i < size(input); ++i) {
        using std::abs;
        acc += abs(input[i]);
    }

    return acc;
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
