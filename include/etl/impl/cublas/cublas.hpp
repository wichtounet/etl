//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Utility functions for cublas
 */

#pragma once

#include "cublas_v2.h"

namespace etl::impl::cublas {

/*!
 * \brief Returns the string representation of the given
 * CUBLAS status code.
 *
 * \param code The  CUBLAS status code
 *
 * \return the string representation of the given status code
 */
inline const char* cublas_str(cublasStatus_t code) {
    switch (code) {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:
            return "unknown CUBLAS error";
    }
}

#define cublas_check(call)                                                                                         \
    {                                                                                                              \
        auto status = call;                                                                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                     \
            std::cerr << "CUDA error: " << etl::impl::cublas::cublas_str(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                                      \
        }                                                                                                          \
    }

/*!
 * \brief RTTI helper to manage CUBLAS handle
 */
struct cublas_handle {
    cublasHandle_t handle; ///< The raw cublas handle

    /*!
     * \brief Construct the helper and create the handle directly
     */
    cublas_handle() {
        cublasCreate(&handle);
    }

    /*!
     * \brief Construct the helper from the raw handle
     * \param handle The raw cublas handle
     */
    explicit cublas_handle(cublasHandle_t handle) : handle(handle) {}

    cublas_handle(const cublas_handle& rhs) = delete;
    cublas_handle& operator=(const cublas_handle& rhs) = delete;

    cublas_handle(cublas_handle&& rhs) noexcept = default;
    cublas_handle& operator=(cublas_handle&& rhs) noexcept = default;

    /*!
     * \brief Get the cublas handle
     * \return the raw cublas handle
     */
    cublasHandle_t get() {
        return handle;
    }

    /*!
     * \brief Destruct the helper and release the raw cublas handle
     */
    ~cublas_handle() {
        cublasDestroy(handle);
    }
};

#ifndef ETL_CUBLAS_LOCAL_HANDLE

/*!
 * \brief Start cublas and return a RTTI helper over a raw cublas handle
 * \return RTTI helper over a raw cublas handle
 */
inline cublas_handle& start_cublas() {
    static cublas_handle handle;
    return handle;
}

#else

/*!
 * \brief Start cublas and return a RTTI helper over a raw cublas handle
 * \return RTTI helper over a raw cublas handle
 */
inline cublas_handle start_cublas() {
    return {};
}

#endif

} //end of namespace etl::impl::cublas
