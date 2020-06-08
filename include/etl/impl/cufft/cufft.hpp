//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cufft.h"

namespace etl::impl::cufft {

/*!
 * \brief Returns the string representation of the given
 * CUFFT status code.
 *
 * \param code The CUFFT status code
 *
 * \return the string representation of the given status code
 */
inline const char* cufft_str(cufftResult code) {
    switch (code) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
        default:
            return "unknown CUFFT error";
    }
}

#define cufft_check(call)                                                                                    \
    if (auto status = call; status != CUFFT_SUCCESS) {                                                       \
        std::cerr << "CUDA error: " << etl::impl::cufft::cufft_str(status) << " from " << #call << std::endl \
                  << "from " << __FILE__ << ":" << __LINE__ << std::endl;                                    \
    }

/*!
 * \brief RAII wrapper for CUFFT context
 */
struct cufft_handle {
    cufftHandle handle; ///< The CUFFT context handle

    /*!
     * \brief Create a new cufft_handle.
     */
    cufft_handle() {
        cufft_check(cufftCreate(&handle));
    }

    cufft_handle(const cufft_handle& rhs) = delete;
    cufft_handle& operator=(const cufft_handle& rhs) = delete;

    cufft_handle(cufft_handle&& rhs) = default;
    cufft_handle& operator=(cufft_handle&& rhs) = default;

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
        cufft_check(cufftDestroy(handle));
    }
};

/*!
 * \brief Obtain an handle to the CUFFT context
 * \return An handle to the CUFFT context
 */
inline cufft_handle start_cufft() {
    return {};
}

/*!
 * \brief cast a etl::complex to its CUFFT equivalent
 * \param ptr Pointer the memory to cast
 * \return Pointer to the same memory but casted to CUFFT equivalent
 */
inline cufftComplex* complex_cast(etl::complex<float>* ptr) {
    return reinterpret_cast<cufftComplex*>(ptr);
}

/*!
 * \brief cast a etl::complex to its CUFFT equivalent
 * \param ptr Pointer the memory to cast
 * \return Pointer to the same memory but casted to CUFFT equivalent
 */
inline cufftDoubleComplex* complex_cast(etl::complex<double>* ptr) {
    return reinterpret_cast<cufftDoubleComplex*>(ptr);
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

} //end of namespace etl::impl::cufft
