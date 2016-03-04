//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration of the different convolution implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different convolution implementations
 */
enum class conv_impl {
    STD, ///< Standard implementation
    SSE, ///< Vectorized SSE implementation
    AVX  ///< Vectorized AVX implementation
};

} //end of namespace etl
