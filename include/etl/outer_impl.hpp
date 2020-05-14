//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for the outer product implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of
 * outer product
 */
enum class outer_impl {
    STD,    ///< Standard implementation
    BLAS,   ///< BLAS implementation
    CUBLAS, ///< CUBLAS implementation
    VEC     ///< VEC implementation
};

} //end of namespace etl
