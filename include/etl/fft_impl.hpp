//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Enumeration for the fft implementations
 */

#pragma once

namespace etl {

/*!
 * \brief The different FFT implementations
 */
enum class fft_impl {
    STD,  ///< The standard implementation
    MKL,  ///< The Intel MKL implementation
    CUFFT ///< The NVidia CuFFT implementation
};

} //end of namespace etl
