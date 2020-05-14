//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/std/inv.hpp"

namespace etl::detail {

/*!
 * \brief Functor for Inverse
 */
struct inv_impl {
    /*!
     * \brief Apply the functor
     * \param a The input sub expression
     * \param c The output sub expression
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        etl::impl::standard::inv(a, c);
    }
};

} //end of namespace etl::detail
