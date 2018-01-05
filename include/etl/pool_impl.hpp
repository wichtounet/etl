//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for the pooling implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of
 * pooling
 */
enum class pool_impl {
    STD,  ///< Standard implementation
    CUDNN ///< CUDNN (GPU) implementation
};

} //end of namespace etl
