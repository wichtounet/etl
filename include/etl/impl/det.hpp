//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the determinant implementation
 */

#pragma once

//Include the implementations
#include "etl/impl/std/det.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Functor for determinant
 */
struct det_impl {
    /*!
     * \brief Apply the functor to A
     * \param A The input matrix
     * \return the determinant of the matrix
     */
    template <typename AT>
    static value_t<AT> apply(const AT& A) {
        return etl::impl::standard::det(A);
    }
};

} //end of namespace detail

} //end of namespace etl
