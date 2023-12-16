//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration of the different matrix-matrix muliplication implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different matrix-matrix
 * multiplication implementations
 */
enum class gemm_impl {
    STD,   ///< Standard implmentation
    VEC,   ///< Vectorized BLAS implementation
    BLAS,  ///< BLAS implementation
    CUBLAS ///< CUBLAS (GPU) implementation
};

} //end of namespace etl
