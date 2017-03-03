//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for the different scalar implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of
 * scalar operations
 */
enum class scalar_impl {
    STD,    ///< Standard implementation
    BLAS,   ///< BLAS implementation
    CUBLAS, ///< CUBLAS implementation
};

} //end of namespace etl
