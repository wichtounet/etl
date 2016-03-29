//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the decompositions implementation
 */

#pragma once

//Include the implementations
#include "etl/impl/std/decomposition.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Functor for euclidean norm
 */
struct lu_impl {
    /*!
     * \brief Apply the functor to A, L, U, P
     * \param A The input matrix
     * \param L The L decomposition (output)
     * \param U The U decomposition (output)
     * \param P The P permutation matrix (output)
     */
    template <typename AT, typename LT, typename UT, typename PT>
    static void apply(const AT& A, LT& L, UT& U, PT& P) {
        etl::impl::standard::lu(A, L, U, P);
    }
};

} //end of namespace detail

} //end of namespace etl
