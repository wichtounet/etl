//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cufft.h"

namespace etl {

namespace impl {

namespace cufft {

/*!
 * \brief RAII wrapper for CUFFT context
 */
struct cufft_handle {
    cufftHandle handle; ///< The CUFFT context handle

    /*!
     * \brief Returns the CUFFT context
     * \return the CUFFT context
     */
    cufftHandle& get() {
        return handle;
    }

    /*!
     * \brief Destroy the handle and release the CUFFT context
     */
    ~cufft_handle() {
        cufftDestroy(handle);
    }
};

/*!
 * \brief Obtain an handle to the CUFFT context
 * \return An handle to the CUFFT context
 */
inline cufft_handle start_cufft() {
    return cufft_handle();
}

/*!
 * \brief cast a std::complex to its CUFFT equivalent
 * \param ptr Pointer the memory to cast
 * \return Pointer to the same memory but casted to CUFFT equivalent
 */
inline cufftComplex* complex_cast(std::complex<float>* ptr) {
    return reinterpret_cast<cufftComplex*>(ptr);
}

/*!
 * \brief cast a std::complex to its CUFFT equivalent
 * \param ptr Pointer the memory to cast
 * \return Pointer to the same memory but casted to CUFFT equivalent
 */
inline cufftDoubleComplex* complex_cast(std::complex<double>* ptr) {
    return reinterpret_cast<cufftDoubleComplex*>(ptr);
}

} //end of namespace cufft

} //end of namespace impl

} //end of namespace etl
