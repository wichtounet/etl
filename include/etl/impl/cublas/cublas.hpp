//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

namespace etl {

namespace impl {

namespace cublas {

/*!
 * \brief RTTI helper to manage CUBLAS handle
 */
struct cublas_handle {
    cublasHandle_t handle; ///< The raw cublas handle

    /*!
     * \brief Construct the helper and create the handle directly
     */
    cublas_handle(){
        cublasCreate(&handle);
    }

    /*!
     * \brief Construct the helper from the raw handle
     * \param handle The raw cublas handle
     */
    cublas_handle(cublasHandle_t handle)
            : handle(handle) {}

    cublas_handle(const cublas_handle& rhs) = delete;
    cublas_handle& operator=(const cublas_handle& rhs) = delete;

    cublas_handle(cublas_handle&& rhs) = default;
    cublas_handle& operator=(cublas_handle&& rhs) = default;

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

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
