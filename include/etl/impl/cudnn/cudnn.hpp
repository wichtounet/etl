//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Utility functions for cudnn
 */

#pragma once

#include "cudnn.h"

#define cudnn_check(call)                                                                                 \
    {                                                                                                     \
        cudnnStatus_t status = call;                                                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                                                             \
            std::cerr << "CUDNN error: " << cudnnGetErrorString(status) << " from " << #call << std::endl \
                      << "from " << __FILE__ << ":" << __LINE__ << std::endl;                             \
        }                                                                                                 \
    }

namespace etl {

namespace impl {

namespace cudnn {

/*!
 * \brief RTTI helper to manage CUDNN handle
 */
struct cudnn_handle {
    cudnnHandle_t handle; ///< The raw cudnn handle

    /*!
     * \brief Construct the helper and create the handle directly
     */
    cudnn_handle(){
        cudnn_check(cudnnCreate(&handle));
    }

    /*!
     * \brief Construct the helper from the raw handle
     * \param handle The raw cudnn handle
     */
    cudnn_handle(cudnnHandle_t handle) : handle(handle) {}

    cudnn_handle(const cudnn_handle& rhs) = delete;
    cudnn_handle& operator=(const cudnn_handle& rhs) = delete;

    cudnn_handle(cudnn_handle&& rhs) = default;
    cudnn_handle& operator=(cudnn_handle&& rhs) = default;

    /*!
     * \brief Get the cudnn handle
     * \return the raw cudnn handle
     */
    cudnnHandle_t get() {
        return handle;
    }

    /*!
     * \brief Destruct the helper and release the raw cudnn handle
     */
    ~cudnn_handle() {
        cudnn_check(cudnnDestroy(handle));
    }
};

#ifndef ETL_CUDNN_LOCAL_HANDLE

/*!
 * \brief Start cudnn and return a RTTI helper over a raw cudnn handle
 * \return RTTI helper over a raw cudnn handle
 */
inline cudnn_handle& start_cudnn() {
    static cudnn_handle handle;
    return handle;
}

#else

/*!
 * \brief Start cudnn and return a RTTI helper over a raw cudnn handle
 * \return RTTI helper over a raw cudnn handle
 */
inline cudnn_handle start_cudnn() {
    return {};
}

#endif

} //end of namespace cudnn

} //end of namespace impl

} //end of namespace etl
