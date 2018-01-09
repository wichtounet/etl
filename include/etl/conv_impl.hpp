//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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
    STD,      ///< Standard implementation
    VEC,      ///< Uniform Vectorized Implementation with locality
    CUDNN,    ///< CUDNN implementation
    FFT_STD,  ///< FFT reduction (with STD impl)
    FFT_MKL,  ///< FFT reduction (with MKL impl)
    FFT_CUFFT ///< FFT reduction (with CUFFT impl)
};

/*!
 * \brief Enumeration describing the different convolution implementations
 */
enum class conv4_impl {
    STD,       ///< Standard implementation
    VEC,       ///< VEC implementation
    CUDNN,     ///< CUDNN implementation
    FFT_STD,   ///< FFT reduction (with STD impl)
    FFT_MKL,   ///< FFT reduction (with MKL impl)
    FFT_CUFFT, ///< FFT reduction (with CUFFT impl)
    BLAS_VEC,  ///< BLAS reduction
    BLAS_MKL   ///< BLAS reduction
};

/*!
 * \brief Enumeration describing the different multiple convolution implementations
 */
enum class conv_multi_impl {
    STD,           ///< Standard implementation
    VEC,           ///< VEC implementation
    VALID_FFT_MKL, ///< Reductiont to FFT (valid)
    FFT_STD,       ///< FFT reduction (with STD impl)
    FFT_MKL,       ///< FFT reduction (with MKL impl)
    FFT_CUFFT,     ///< FFT reduction (with CUFFT impl)
    BLAS_VEC,      ///< Reduction to BLAS (GEMM)
    BLAS_MKL,      ///< Reduction to BLAS (GEMM)
    CUDNN          ///< GPU with CUDNN
};

} //end of namespace etl
