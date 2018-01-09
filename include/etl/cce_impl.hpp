//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for CCE implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of CCE
 */
enum class cce_impl {
    STD,   ///< Standard implementation
    EGBLAS ///< GPU implementation
};

} //end of namespace etl
