//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for the bias_add implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different implementations of
 * bias_add
 */
enum class bias_add_impl {
    STD,    ///< Standard implementation
    VEC,    ///< VEC implementation
    EGBLAS, ///< ETL-GPU-BLAS (GPU) implementation
    CUDNN   ///< CUDNN (GPU) implementation
};

} //end of namespace etl
