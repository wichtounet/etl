//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cublas_v2.h"

namespace etl {

namespace impl {

namespace cublas {

struct cublas_handle {
    cublasHandle_t handle;

    cublas_handle(cublasHandle_t handle)
            : handle(handle) {}

    cublasHandle_t get() {
        return handle;
    }
    ~cublas_handle() {
        cublasDestroy(handle);
    }
};

inline cublas_handle start_cublas() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    return {handle};
}

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
