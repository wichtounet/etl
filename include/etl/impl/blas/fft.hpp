//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_IMPL_BLAS_FFT_HPP
#define ETL_IMPL_BLAS_FFT_HPP

#include "../../config.hpp"

#ifdef ETL_MKL_MODE
#include "mkl_dfti.h"
#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_MKL_MODE

template<typename A, typename C>
void sfft1(A&& a, C&& c){
    auto a_complex = allocate<std::complex<float>>(a.size());

    std::copy(a.begin(), a.end(), a_complex.get());

    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    status = DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, a.size()); //Specify size and precision
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);                //Out of place FFT
    status = DftiCommitDescriptor(descriptor);                                          //Finalize the descriptor
    status = DftiComputeForward(descriptor, a_complex.get(), c.memory_start());         //Compute the Forward FFT
    status = DftiFreeDescriptor(&descriptor);                                           //Free the descriptor
};

template<typename A, typename C>
void dfft1(A&& a, C&& c){
    auto a_complex = allocate<std::complex<double>>(a.size());

    std::copy(a.begin(), a.end(), a_complex.get());

    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, a.size()); //Specify size and precision
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);                //Out of place FFT
    status = DftiCommitDescriptor(descriptor);                                          //Finalize the descriptor
    status = DftiComputeForward(descriptor, a_complex.get(), c.memory_start());         //Compute the Forward FFT
    status = DftiFreeDescriptor(&descriptor);                                           //Free the descriptor
};

template<typename A, typename C>
void cfft1(A&& a, C&& c){
    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    auto* a_ptr = const_cast<void*>(static_cast<const void*>(a.memory_start()));

    status = DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, a.size()); //Specify size and precision
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);                //Out of place FFT
    status = DftiCommitDescriptor(descriptor);                                          //Finalize the descriptor
    status = DftiComputeForward(descriptor, a_ptr, c.memory_start());                   //Compute the Forward FFT
    status = DftiFreeDescriptor(&descriptor);                                           //Free the descriptor
};

template<typename A, typename C>
void zfft1(A&& a, C&& c){
    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    auto* a_ptr = const_cast<void*>(static_cast<const void*>(a.memory_start()));

    status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, a.size()); //Specify size and precision
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);                //Out of place FFT
    status = DftiCommitDescriptor(descriptor);                                          //Finalize the descriptor
    status = DftiComputeForward(descriptor, a_ptr, c.memory_start());                   //Compute the Forward FFT
    status = DftiFreeDescriptor(&descriptor);                                           //Free the descriptor
};

template<typename A, typename C>
void cifft1(A&& a, C&& c){
    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    auto* a_ptr = const_cast<void*>(static_cast<const void*>(a.memory_start()));

    status = DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, a.size()); //Specify size and precision
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);                //Out of place FFT
    status = DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / a.size());           //Scale down the output
    status = DftiCommitDescriptor(descriptor);                                          //Finalize the descriptor
    status = DftiComputeBackward(descriptor, a_ptr, c.memory_start());                  //Compute the Forward FFT
    status = DftiFreeDescriptor(&descriptor);                                           //Free the descriptor
};

template<typename A, typename C>
void zifft1(A&& a, C&& c){
    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    auto* a_ptr = const_cast<void*>(static_cast<const void*>(a.memory_start()));

    status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, a.size()); //Specify size and precision
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);                //Out of place FFT
    status = DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / a.size());           //Scale down the output
    status = DftiCommitDescriptor(descriptor);                                          //Finalize the descriptor
    status = DftiComputeBackward(descriptor, a_ptr, c.memory_start());                  //Compute the Forward FFT
    status = DftiFreeDescriptor(&descriptor);                                           //Free the descriptor
};

template<typename A, typename C>
void cifft1_real(A&& a, C&& c){
    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    auto c_complex = allocate<std::complex<float>>(a.size());

    auto* a_ptr = const_cast<void*>(static_cast<const void*>(a.memory_start()));

    status = DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, a.size()); //Specify size and precision
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);                //Out of place FFT
    status = DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / a.size());           //Scale down the output
    status = DftiCommitDescriptor(descriptor);                                          //Finalize the descriptor
    status = DftiComputeBackward(descriptor, a_ptr, c_complex.get());                   //Compute the Forward FFT
    status = DftiFreeDescriptor(&descriptor);                                           //Free the descriptor

    for(std::size_t i = 0; i < a.size(); ++i){
        c[i] = c_complex[i].real();
    }
};

template<typename A, typename C>
void zifft1_real(A&& a, C&& c){
    DFTI_DESCRIPTOR_HANDLE descriptor;
    MKL_LONG status;

    auto c_complex = allocate<std::complex<double>>(a.size());

    auto* a_ptr = const_cast<void*>(static_cast<const void*>(a.memory_start()));

    status = DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, a.size()); //Specify size and precision
    status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);                //Out of place FFT
    status = DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / a.size());            //Scale down the output
    status = DftiCommitDescriptor(descriptor);                                          //Finalize the descriptor
    status = DftiComputeBackward(descriptor, a_ptr, c_complex.get());                   //Compute the Forward FFT
    status = DftiFreeDescriptor(&descriptor);                                           //Free the descriptor

    for(std::size_t i = 0; i < a.size(); ++i){
        c[i] = c_complex[i].real();
    }
};

#else

template<typename A, typename C>
void sfft1(A&&, C&&);

template<typename A, typename C>
void dfft1(A&&, C&&);

template<typename A, typename C>
void cfft1(A&&, C&&);

template<typename A, typename C>
void zfft1(A&&, C&&);

template<typename A, typename C>
void cifft1(A&&, C&&);

template<typename A, typename C>
void zifft1(A&&, C&&);

template<typename A, typename C>
void cifft1_real(A&&, C&&);

template<typename A, typename C>
void zifft1_real(A&&, C&&);

#endif

} //end of namespace blas

} //end of namespace impl

} //end of namespace etl

#endif
