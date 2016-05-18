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

namespace etl {

namespace impl {

namespace cudnn {

/*!
 * \brief RTTI helper to manage CUDNN handle
 */
struct cudnn_handle {
    cudnnHandle_t handle; ///< The raw cudnn handle

    /*!
     * \brief Construct the helper from the raw handle
     * \param handle The raw cudnn handle
     */
    cudnn_handle(cudnnHandle_t handle)
            : handle(handle) {}

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
        cudnnDestroy(handle);
    }
};

/*!
 * \brief Start cudnn and return a RTTI helper over a raw cudnn handle
 * \return RTTI helper over a raw cudnn handle
 */
inline cudnn_handle start_cudnn() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);
    return {handle};
}

} //end of namespace cudnn

} //end of namespace impl

} //end of namespace etl
