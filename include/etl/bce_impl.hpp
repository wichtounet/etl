//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for BCE implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of BCE
 */
enum class bce_impl {
    STD,   ///< Standard implementation
    EGBLAS ///< GPU implementation
};

} //end of namespace etl
