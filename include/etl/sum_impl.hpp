//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for sum implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of sum
 */
enum class sum_impl {
    STD,   ///< Standard implementation
    VEC,   ///< Vectorized implementation
    BLAS,  ///< BLAS implementation
    CUBLAS ///< BLAS implementation
};

} //end of namespace etl
