//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cufft.h"

namespace etl {

namespace impl {

namespace cufft {

struct cufft_handle {
    cufftHandle handle;

    cufftHandle& get() {
        return handle;
    }

    ~cufft_handle() {
        cufftDestroy(handle);
    }
};

inline cufft_handle start_cufft() {
    return cufft_handle();
}

inline cufftComplex* complex_cast(std::complex<float>* ptr) {
    return reinterpret_cast<cufftComplex*>(ptr);
}

inline cufftDoubleComplex* complex_cast(std::complex<double>* ptr) {
    return reinterpret_cast<cufftDoubleComplex*>(ptr);
}

} //end of namespace cufft

} //end of namespace impl

} //end of namespace etl
