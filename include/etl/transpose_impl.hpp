//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
    STD,    ///< Standard implementation
    VEC,    ///< Vectorized implementation
    MKL,    ///< MKL implementation
    CUBLAS, ///< CUBLAS implementation
};

} //end of namespace etl
