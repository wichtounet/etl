//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file norm.hpp
 * \brief Selector for the euclidean norm operation
 */

#pragma once

//Include the implementations
#include "etl/impl/std/norm.hpp"

namespace etl::detail {

/*!
 * \brief Functor for euclidean norm
 */
struct norm_impl {
    /*!
     * \brief Apply the functor to a
     * \param a the expression
     * \return the euclidean norm of a
     */
    template <typename A>
    static value_t<A> apply(const A& a) {
        return etl::impl::standard::norm(a);
    }
};

} //end of namespace etl::detail
