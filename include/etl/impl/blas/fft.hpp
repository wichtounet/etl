//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/allocator.hpp"

#ifdef ETL_MKL_MODE
#include "mkl_dfti.h"
#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_MKL_MODE

namespace detail {

inline void cfft_kernel(const std::complex<float>* in, std::size_t s, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                        //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void zfft_kernel(const std::complex<double>* in, std::size_t s, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                        //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void cfft_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t n, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, n); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);         //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, n);                   //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, n);                  //Output stride
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                        //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void zfft_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t n, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, n); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);         //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, n);                   //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, n);                  //Output stride
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                        //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void inplace_cfft_kernel(std::complex<float>* in, std::size_t s) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = static_cast<void*>(in);

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr);                             //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void inplace_zfft_kernel(std::complex<double>* in, std::size_t s) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = static_cast<void*>(in);

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, in);                         //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void cifft_kernel(const std::complex<float>* in, std::size_t s, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / s);            //Scale down the output
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                       //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void zifft_kernel(const std::complex<double>* in, std::size_t s, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / s);             //Scale down the output
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                       //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void cifft_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t s, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / s);            //Scale down the output
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);         //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, s);                   //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, s);                  //Output stride
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                       //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void zifft_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t s, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / s);             //Scale down the output
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);         //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, s);                   //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, s);                  //Output stride
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                       //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void inplace_cifft_kernel(std::complex<float>* in, std::size_t s) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = static_cast<void*>(in);

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / s);            //Scale down the output
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                            //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void inplace_zifft_kernel(std::complex<double>* in, std::size_t s) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = static_cast<void*>(in);

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / s);             //Scale down the output
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                            //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

inline void cfft2_kernel(const std::complex<float>* in, std::size_t d1, std::size_t d2, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                          //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void zfft2_kernel(const std::complex<double>* in, std::size_t d1, std::size_t d2, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                          //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void cfft2_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t d1, std::size_t d2, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);           //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, d1 * d2);               //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, d1 * d2);              //Output stride
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                          //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void zfft2_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t d1, std::size_t d2, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);           //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, d1 * d2);               //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, d1 * d2);              //Output stride
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                          //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void inplace_cfft2_kernel(std::complex<float>* in, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr);                               //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void inplace_zfft2_kernel(std::complex<double>* in, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr);                               //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void cifft2_kernel(const std::complex<float>* in, std::size_t d1, std::size_t d2, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / (d1 * d2));      //Scale down the output
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                         //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void zifft2_kernel(const std::complex<double>* in, std::size_t d1, std::size_t d2, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / (d1 * d2));       //Scale down the output
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                         //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void cifft2_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t d1, std::size_t d2, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / (d1 * d2));      //Scale down the output
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);           //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, d1 * d2);               //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, d1 * d2);              //Output stride
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                         //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void zifft2_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t d1, std::size_t d2, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / (d1 * d2));       //Scale down the output
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);           //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, d1 * d2);               //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, d1 * d2);              //Output stride
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                         //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void inplace_cifft2_kernel(std::complex<float>* in, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / (d1 * d2));      //Scale down the output
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                              //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

inline void inplace_zifft2_kernel(std::complex<double>* in, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / (d1 * d2));       //Scale down the output
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                              //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

} //End of namespace detail

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1(A&& a, C&& c) {
    auto a_complex = allocate<std::complex<float>>(etl::size(a));

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    detail::cfft_kernel(a_complex.get(), etl::size(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1(A&& a, C&& c) {
    auto a_complex = allocate<std::complex<double>>(etl::size(a));

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    detail::zfft_kernel(a_complex.get(), etl::size(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft1(A&& a, C&& c) {
    detail::cfft_kernel(a.memory_start(), etl::size(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft1(A&& a, C&& c) {
    detail::zfft_kernel(a.memory_start(), etl::size(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1(A&& a, C&& c) {
    detail::cifft_kernel(a.memory_start(), etl::size(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1(A&& a, C&& c) {
    detail::zifft_kernel(a.memory_start(), etl::size(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    auto c_complex = allocate<std::complex<float>>(etl::size(a));

    detail::cifft_kernel(a.memory_start(), etl::size(a), c_complex.get());

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = c_complex[i].real();
    }
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    auto c_complex = allocate<std::complex<double>>(etl::size(a));

    detail::zifft_kernel(a.memory_start(), etl::size(a), c_complex.get());

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = c_complex[i].real();
    }
}

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    auto a_complex = allocate<std::complex<float>>(etl::size(a));

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    detail::cfft_many_kernel(a_complex.get(), batch, n, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    auto a_complex = allocate<std::complex<double>>(etl::size(a));

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    detail::zfft_many_kernel(a_complex.get(), batch, n, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    detail::cfft_many_kernel(a.memory_start(), batch, n, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    detail::zfft_many_kernel(a.memory_start(), batch, n, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    detail::cifft_many_kernel(a.memory_start(), batch, n, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    detail::zifft_many_kernel(a.memory_start(), batch, n, c.memory_start());
}

template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft1_convolve(A&& a, B&& b, C&& c) {
    const std::size_t m    = etl::size(a);
    const std::size_t n    = etl::size(b);
    const std::size_t size = m + n - 1;

    auto a_padded = allocate<std::complex<float>>(size);
    auto b_padded = allocate<std::complex<float>>(size);

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_padded.get());
    standard_evaluator::direct_copy(b.memory_start(), b.memory_end(), b_padded.get());

    detail::inplace_cfft_kernel(a_padded.get(), size);
    detail::inplace_cfft_kernel(b_padded.get(), size);

    for (std::size_t i = 0; i < size; ++i) {
        a_padded[i] *= b_padded[i];
    }

    detail::inplace_cifft_kernel(a_padded.get(), size);

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real();
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft1_convolve(A&& a, B&& b, C&& c) {
    const std::size_t m    = etl::size(a);
    const std::size_t n    = etl::size(b);
    const std::size_t size = m + n - 1;

    auto a_padded = allocate<std::complex<double>>(size);
    auto b_padded = allocate<std::complex<double>>(size);

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_padded.get());
    standard_evaluator::direct_copy(b.memory_start(), b.memory_end(), b_padded.get());

    detail::inplace_zfft_kernel(a_padded.get(), size);
    detail::inplace_zfft_kernel(b_padded.get(), size);

    for (std::size_t i = 0; i < size; ++i) {
        a_padded[i] *= b_padded[i];
    }

    detail::inplace_zifft_kernel(a_padded.get(), size);

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real();
    }
}

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2(A&& a, C&& c) {
    auto a_complex = allocate<std::complex<float>>(etl::size(a));

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    detail::cfft2_kernel(a_complex.get(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    auto a_complex = allocate<std::complex<double>>(etl::size(a));

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    detail::zfft2_kernel(a_complex.get(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft2(A&& a, C&& c) {
    detail::cfft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    detail::zfft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    auto a_complex = allocate<std::complex<float>>(etl::size(a));

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    detail::cfft2_many_kernel(a_complex.get(), batch, n1, n2, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    auto a_complex = allocate<std::complex<double>>(etl::size(a));

    standard_evaluator::direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    detail::zfft2_many_kernel(a_complex.get(), batch, n1, n2, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    detail::cfft2_many_kernel(a.memory_start(), batch, n1, n2, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    detail::zfft2_many_kernel(a.memory_start(), batch, n1, n2, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    detail::cifft2_many_kernel(a.memory_start(), batch, n1, n2, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2_many(A&& a, C&& c) {
    static constexpr const std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    detail::zifft2_many_kernel(a.memory_start(), batch, n1, n2, c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2(A&& a, C&& c) {
    detail::cifft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2(A&& a, C&& c) {
    detail::zifft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());
}

template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft2_real(A&& a, C&& c) {
    auto c_complex = allocate<std::complex<float>>(etl::size(a));

    detail::cifft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c_complex.get());

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = c_complex[i].real();
    }
}

template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft2_real(A&& a, C&& c) {
    auto c_complex = allocate<std::complex<double>>(etl::size(a));

    detail::zifft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c_complex.get());

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = c_complex[i].real();
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2_convolve(A&& a, B&& b, C&& c) {
    const std::size_t m1 = etl::dim<0>(a);
    const std::size_t n1 = etl::dim<0>(b);
    const std::size_t s1 = m1 + n1 - 1;

    const std::size_t m2 = etl::dim<1>(a);
    const std::size_t n2 = etl::dim<1>(b);
    const std::size_t s2 = m2 + n2 - 1;

    //Note: use of value_t to make the type dependent!
    dyn_vector<etl::complex<value_t<A>>> a_padded(etl::size(c));
    dyn_vector<etl::complex<value_t<A>>> b_padded(etl::size(c));

    for (std::size_t i = 0; i < m1; ++i) {
        for (std::size_t j = 0; j < m2; ++j) {
            a_padded[i * s2 + j] = a(i, j);
        }
    }

    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < n2; ++j) {
            b_padded[i * s2 + j] = b(i, j);
        }
    }

    detail::inplace_cfft2_kernel(reinterpret_cast<std::complex<float>*>(a_padded.memory_start()), s1, s2);
    detail::inplace_cfft2_kernel(reinterpret_cast<std::complex<float>*>(b_padded.memory_start()), s1, s2);

    a_padded *= b_padded;

    detail::inplace_cifft2_kernel(reinterpret_cast<std::complex<float>*>(a_padded.memory_start()), s1, s2);

    for (std::size_t i = 0; i < etl::size(c); ++i) {
        c[i] = a_padded[i].real;
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2_convolve(A&& a, B&& b, C&& c) {
    const std::size_t m1 = etl::dim<0>(a);
    const std::size_t n1 = etl::dim<0>(b);
    const std::size_t s1 = m1 + n1 - 1;

    const std::size_t m2 = etl::dim<1>(a);
    const std::size_t n2 = etl::dim<1>(b);
    const std::size_t s2 = m2 + n2 - 1;

    //Note: use of value_t to make the type dependent!
    dyn_vector<etl::complex<value_t<A>>> a_padded(etl::size(c));
    dyn_vector<etl::complex<value_t<A>>> b_padded(etl::size(c));

    for (std::size_t i = 0; i < m1; ++i) {
        for (std::size_t j = 0; j < m2; ++j) {
            a_padded[i * s2 + j] = a(i, j);
        }
    }

    for (std::size_t i = 0; i < n1; ++i) {
        for (std::size_t j = 0; j < n2; ++j) {
            b_padded[i * s2 + j] = b(i, j);
        }
    }

    detail::inplace_zfft2_kernel(reinterpret_cast<std::complex<float>*>(a_padded.memory_start()), s1, s2);
    detail::inplace_zfft2_kernel(reinterpret_cast<std::complex<float>*>(b_padded.memory_start()), s1, s2);

    a_padded *= b_padded;

    detail::inplace_zifft2_kernel(reinterpret_cast<std::complex<float>*>(a_padded.memory_start()), s1, s2);

    for (std::size_t i = 0; i < etl::size(c); ++i) {
        c[i] = a_padded[i].real;
    }
}

#else

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft1(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1_real(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void fft1_many(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform many 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void ifft1_many(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void fft2(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2_real(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void fft2_many(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform many 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C>
void ifft2_many(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 1D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void fft1_convolve(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void fft2_convolve(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

#endif

} //end of namespace blas

} //end of namespace impl

} //end of namespace etl
