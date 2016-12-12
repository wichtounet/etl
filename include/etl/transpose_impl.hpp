//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration of the transpose implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of transpose
 */
enum class transpose_impl {
    SELECT, ///< Select the best implementation
    STD,    ///< Standard implementation
    MKL,    ///< MKL implementation
};

} //end of namespace etl
