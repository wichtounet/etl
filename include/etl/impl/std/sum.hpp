//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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

template <typename E>
value_t<E> sum(const E& input, std::size_t first, std::size_t last) {
    value_t<E> acc(0);

    for (std::size_t i = first; i < last; ++i) {
        acc += input[i];
    }

    return acc;
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
