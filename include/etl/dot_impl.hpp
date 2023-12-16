//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration of the dot implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of dot
 */
enum class dot_impl {
    STD,   ///< Standard implementation
    VEC,   ///< Uniform Vectorized implementation
    BLAS,  ///< BLAS implementation
    CUBLAS ///< BLAS implementation
};

} //end of namespace etl
