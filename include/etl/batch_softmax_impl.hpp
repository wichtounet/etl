//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for batch_softmax implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of CCE
 */
enum class batch_softmax_impl {
    STD,  ///< Standard implementation
    CUDNN ///< GPU implementation
};

} //end of namespace etl
