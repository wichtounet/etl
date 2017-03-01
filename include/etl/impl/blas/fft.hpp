//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_MKL_MODE
#include "mkl_dfti.h"
#include "etl/util/safe_cast.hpp"
#include "etl/impl/common/conv.hpp"
#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_MKL_MODE

namespace mkl_detail {

/*!
 * \brief FFT kernel, single precision
 * \param in The input vector
 * \param s The size of the vector
 * \param out The output vector
 */
inline void fft_kernel(const std::complex<float>* in, std::size_t s, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                        //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

/*!
 * \brief FFT kernel, double precision
 * \param in The input vector
 * \param s The size of the vector
 * \param out The output vector
 */
inline void fft_kernel(const std::complex<double>* in, std::size_t s, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                        //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

/*!
 * \brief Many FFT kernel, single precision
 * \param in The input vectors
 * \param batch The number of batches
 * \param n The size of the inner vector
 * \param out The output vectors
 */
inline void fft_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t n, std::complex<float>* out) {
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

/*!
 * \brief Many FFT kernel, double precision
 * \param in The input vectors
 * \param batch The number of batches
 * \param n The size of the inner vector
 * \param out The output vectors
 */
inline void fft_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t n, std::complex<double>* out) {
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

/*!
 * \brief Inplace FFT kernel, single precision
 * \param in The input vector
 * \param s The size of the vector
 */
inline void inplace_fft_kernel(std::complex<float>* in, std::size_t s) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = static_cast<void*>(in);

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr);                             //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

/*!
 * \brief Inplace FFT kernel, double precision
 * \param in The input vector
 * \param s The size of the vector
 */
inline void inplace_fft_kernel(std::complex<double>* in, std::size_t s) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = static_cast<void*>(in);

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, in);                         //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

/*!
 * \brief Inverse FFT kernel, single precision
 * \param in The input vector
 * \param s The size of the vector
 * \param out The output vector
 */
inline void ifft_kernel(const std::complex<float>* in, std::size_t s, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / s);            //Scale down the output
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                       //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

/*!
 * \brief Inverse FFT kernel, double precision
 * \param in The input vector
 * \param s The size of the vector
 * \param out The output vector
 */
inline void ifft_kernel(const std::complex<double>* in, std::size_t s, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);         //Out of place FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / s);             //Scale down the output
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr, out);                       //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

/*!
 * \brief Many Inverse FFT kernel, single precision
 * \param in The input vectors
 * \param batch The number of batches
 * \param s The size of the vector
 * \param out The output vectors
 */
inline void ifft_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t s, std::complex<float>* out) {
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

/*!
 * \brief Many Inverse FFT kernel, double precision
 * \param in The input vectors
 * \param batch The number of batches
 * \param s The size of the vector
 * \param out The output vectors
 */
inline void ifft_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t s, std::complex<double>* out) {
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

/*!
 * \brief Inplace Inverse FFT kernel, single precision
 * \param in The input vector
 * \param s The size of the vector
 */
inline void inplace_ifft_kernel(std::complex<float>* in, std::size_t s) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = static_cast<void*>(in);

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / s);            //Scale down the output
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                            //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

/*!
 * \brief Inplace Inverse FFT kernel, double precision
 * \param in The input vector
 * \param s The size of the vector
 */
inline void inplace_ifft_kernel(std::complex<double>* in, std::size_t s) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    void* in_ptr = static_cast<void*>(in);

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 1, s); //Specify size and precision
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / s);             //Scale down the output
    DftiCommitDescriptor(descriptor);                                   //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                            //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                    //Free the descriptor
}

/*!
 * \brief 2D FFT kernel, single precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void fft2_kernel(const std::complex<float>* in, std::size_t d1, std::size_t d2, std::complex<float>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                          //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief 2D FFT kernel, double precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void fft2_kernel(const std::complex<double>* in, std::size_t d1, std::size_t d2, std::complex<double>* out) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE);           //Out of place FFT
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr, out);                          //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief 2D FFT kernel, single precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void fft2_kernel(const etl::complex<float>* in, std::size_t d1, std::size_t d2, etl::complex<float>* out) {
    fft2_kernel(reinterpret_cast<const std::complex<float>*>(in), d1, d2, reinterpret_cast<std::complex<float>*>(out));
}

/*!
 * \brief 2D FFT kernel, single precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void fft2_kernel(const etl::complex<double>* in, std::size_t d1, std::size_t d2, etl::complex<double>* out) {
    fft2_kernel(reinterpret_cast<const std::complex<double>*>(in), d1, d2, reinterpret_cast<std::complex<double>*>(out));
}

/*!
 * \brief Many 2D FFT kernel, single precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 */
inline void inplace_fft2_many_kernel(std::complex<float>* in, std::size_t batch, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_INPLACE);               //Inpllace FFT
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);           //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, d1 * d2);               //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, d1 * d2);              //Output stride
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr);                               //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief Many 2D FFT kernel, double precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 */
inline void inplace_fft2_many_kernel(std::complex<double>* in, std::size_t batch, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_INPLACE);               //Inplace FFT
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);           //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, d1 * d2);               //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, d1 * d2);              //Output stride
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr);                               //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief Many 2D FFT kernel, single precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void fft2_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t d1, std::size_t d2, std::complex<float>* out) {
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

/*!
 * \brief Many 2D FFT kernel, double precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void fft2_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t d1, std::size_t d2, std::complex<double>* out) {
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

/*!
 * \brief Many 2D FFT kernel, single precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void fft2_many_kernel(const etl::complex<float>* in, std::size_t batch, std::size_t d1, std::size_t d2, etl::complex<float>* out) {
    fft2_many_kernel(reinterpret_cast<const std::complex<float>*>(in), batch, d1, d2, reinterpret_cast<std::complex<float>*>(out));
}

/*!
 * \brief Many 2D FFT kernel, double precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void fft2_many_kernel(const etl::complex<double>* in, std::size_t batch, std::size_t d1, std::size_t d2, etl::complex<double>* out) {
    fft2_many_kernel(reinterpret_cast<const std::complex<double>*>(in), batch, d1, d2, reinterpret_cast<std::complex<double>*>(out));
}

/*!
 * \brief Inplace 2D FFT kernel, single precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 */
inline void inplace_fft2_kernel(std::complex<float>* in, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr);                               //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief Inplace 2D FFT kernel, double precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 */
inline void inplace_fft2_kernel(std::complex<double>* in, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeForward(descriptor, in_ptr);                               //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief Inverse 2D FFT kernel, single precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 */
inline void ifft2_kernel(const std::complex<float>* in, std::size_t d1, std::size_t d2, std::complex<float>* out) {
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

/*!
 * \brief Inverse 2D FFT kernel, double precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 */
inline void ifft2_kernel(const std::complex<double>* in, std::size_t d1, std::size_t d2, std::complex<double>* out) {
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

/*!
 * \brief Many Inverse 2D FFT kernel, single precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void inplace_ifft2_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_INPLACE);               //Inplace FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / (d1 * d2));      //Scale down the output
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);           //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, d1 * d2);               //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, d1 * d2);              //Output stride
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                              //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief Many Inverse 2D FFT kernel, double precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void inplace_ifft2_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_INPLACE);               //Inplace FFT
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / (d1 * d2));       //Scale down the output
    DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS, batch);           //Number of transforms
    DftiSetValue(descriptor, DFTI_INPUT_DISTANCE, d1 * d2);               //Input stride
    DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE, d1 * d2);              //Output stride
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                              //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief Many Inverse 2D FFT kernel, single precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void ifft2_many_kernel(const std::complex<float>* in, std::size_t batch, std::size_t d1, std::size_t d2, std::complex<float>* out) {
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

/*!
 * \brief Many Inverse 2D FFT kernel, double precision
 * \param in The input matrix
 * \param batch The number of batches
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 * \param out The output matrix
 */
inline void ifft2_many_kernel(const std::complex<double>* in, std::size_t batch, std::size_t d1, std::size_t d2, std::complex<double>* out) {
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

/*!
 * \brief Inplace Inverse 2D FFT kernel, single precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 */
inline void inplace_ifft2_kernel(std::complex<float>* in, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0f / (d1 * d2));      //Scale down the output
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                              //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief Inplace Inverse 2D FFT kernel, double precision
 * \param in The input matrix
 * \param d1 The first dimension of the matrix
 * \param d2 The second dimension of the matrix
 */
inline void inplace_ifft2_kernel(std::complex<double>* in, std::size_t d1, std::size_t d2) {
    DFTI_DESCRIPTOR_HANDLE descriptor;

    MKL_LONG dim[]{static_cast<long>(d1), static_cast<long>(d2)};

    void* in_ptr = const_cast<void*>(static_cast<const void*>(in));

    DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_COMPLEX, 2, dim); //Specify size and precision
    DftiSetValue(descriptor, DFTI_BACKWARD_SCALE, 1.0 / (d1 * d2));       //Scale down the output
    DftiCommitDescriptor(descriptor);                                     //Finalize the descriptor
    DftiComputeBackward(descriptor, in_ptr);                              //Compute the Forward FFT
    DftiFreeDescriptor(&descriptor);                                      //Free the descriptor
}

/*!
 * \brief 2D full convolution kernel with FFT
 * \param a The input matrix
 * \param m1 The first dimension of the input matrix
 * \param m2 The second dimension of the input matrix
 * \param b The kernel matrix
 * \param n1 The first dimension of the kernel matrix
 * \param n2 The second dimension of the kernel matrix
 * \param c The output matrix
 * \param beta The multiplier for the previous value of c
 */
template <typename T>
void conv2_full_kernel(const T* a, std::size_t m1, std::size_t m2, const T* b, std::size_t n1, std::size_t n2, T* c, T beta) {
    const std::size_t s1 = m1 + n1 - 1;
    const std::size_t s2 = m2 + n2 - 1;
    const std::size_t size = s1 * s2;

    dyn_vector<etl::complex<T>> a_padded(size);
    dyn_vector<etl::complex<T>> b_padded(size);

    for (std::size_t i = 0; i < m1; ++i) {
        direct_copy_n(a + i * m2, a_padded.memory_start() + i * s2, m2);
    }

    for (std::size_t i = 0; i < n1; ++i) {
        direct_copy_n(b + i * n2, b_padded.memory_start() + i * s2, n2);
    }

    inplace_fft2_kernel(safe_cast(a_padded.memory_start()), s1, s2);
    inplace_fft2_kernel(safe_cast(b_padded.memory_start()), s1, s2);

    a_padded *= b_padded;

    inplace_ifft2_kernel(safe_cast(a_padded.memory_start()), s1, s2);

    if (beta == T(0.0)) {
        for (std::size_t i = 0; i < size; ++i) {
            c[i] = a_padded[i].real;
        }
    } else {
        for (std::size_t i = 0; i < size; ++i) {
            c[i] = beta * c[i] + a_padded[i].real;
        }
    }
}

} //End of namespace mkl_detail

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template<typename A, typename C, cpp_enable_if(all_single_precision<A>::value && all_complex_single_precision<C>::value)>
void fft1(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    auto a_complex = allocate<std::complex<float>>(etl::size(a));

    direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    mkl_detail::fft_kernel(a_complex.get(), a.size(), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template<typename A, typename C, cpp_enable_if(all_double_precision<A>::value && all_complex_double_precision<C>::value)>
void fft1(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    auto a_complex = allocate<std::complex<double>>(a.size());

    direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    mkl_detail::fft_kernel(a_complex.get(), a.size(), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template<typename A, typename C, cpp_enable_if(all_complex<A,C>::value)>
void fft1(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    mkl_detail::fft_kernel(a.memory_start(), etl::size(a), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft1(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    mkl_detail::ifft_kernel(a.memory_start(), etl::size(a), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_single_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    auto c_complex = allocate<std::complex<float>>(etl::size(a));

    mkl_detail::ifft_kernel(a.memory_start(), etl::size(a), c_complex.get());

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = c_complex[i].real();
    }

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 1D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex_double_precision<A>::value)>
void ifft1_real(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    auto c_complex = allocate<std::complex<double>>(etl::size(a));

    mkl_detail::ifft_kernel(a.memory_start(), etl::size(a), c_complex.get());

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = c_complex[i].real();
    }

    c.invalidate_gpu();
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template<typename A, typename C, cpp_enable_if(!all_complex<A>::value && all_complex<C>::value)>
void fft1_many(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    static constexpr std::size_t N = etl::dimensions<C>();

    std::size_t n     = a.template dim<N - 1>(); //Size of the transform
    std::size_t batch = etl::size(a) / n;            //Number of batch

    auto a_complex = allocate<value_t<C>>(etl::size(a));

    direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    mkl_detail::fft_many_kernel(a_complex.get(), batch, n, c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform many 1D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template<typename A, typename C, cpp_enable_if(all_complex<A, C>::value)>
void fft1_many(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    static constexpr std::size_t N = etl::dimensions<C>();

    std::size_t n     = a.template dim<N - 1>(); //Size of the transform
    std::size_t batch = etl::size(a) / n;        //Number of batch

    mkl_detail::fft_many_kernel(a.memory_start(), batch, n, c.memory_start());

    c.invalidate_gpu();
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
    a.ensure_cpu_up_to_date();

    static constexpr std::size_t N = etl::dimensions<A>();

    std::size_t n     = etl::dim<N - 1>(a); //Size of the transform
    std::size_t batch = etl::size(a) / n;   //Number of batch

    mkl_detail::ifft_many_kernel(a.memory_start(), batch, n, c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 1D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv1_full(A&& a, B&& b, C&& c) {
    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    using type = value_t<A>;

    const std::size_t m    = etl::size(a);
    const std::size_t n    = etl::size(b);
    const std::size_t size = m + n - 1;

    //Note: use of value_t to make the type dependent!
    dyn_vector<etl::complex<type>> a_padded(etl::size(c));
    dyn_vector<etl::complex<type>> b_padded(etl::size(c));

    direct_copy(a.memory_start(), a.memory_end(), a_padded.memory_start());
    direct_copy(b.memory_start(), b.memory_end(), b_padded.memory_start());

    mkl_detail::inplace_fft_kernel(reinterpret_cast<std::complex<type>*>(a_padded.memory_start()), size);
    mkl_detail::inplace_fft_kernel(reinterpret_cast<std::complex<type>*>(b_padded.memory_start()), size);

    a_padded *= b_padded;

    mkl_detail::inplace_ifft_kernel(reinterpret_cast<std::complex<type>*>(a_padded.memory_start()), size);

    for (std::size_t i = 0; i < size; ++i) {
        c[i] = a_padded[i].real;
    }

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    auto a_complex = allocate<std::complex<float>>(etl::size(a));

    direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    mkl_detail::fft2_kernel(a_complex.get(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    auto a_complex = allocate<std::complex<double>>(etl::size(a));

    direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    mkl_detail::fft2_kernel(a_complex.get(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex<A>::value)>
void fft2(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    mkl_detail::fft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_single_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    static constexpr std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    auto a_complex = allocate<std::complex<float>>(etl::size(a));

    direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    mkl_detail::fft2_many_kernel(a_complex.get(), batch, n1, n2, c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_double_precision<A>::value)>
void fft2_many(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    static constexpr std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    auto a_complex = allocate<std::complex<double>>(etl::size(a));

    direct_copy(a.memory_start(), a.memory_end(), a_complex.get());

    mkl_detail::fft2_many_kernel(a_complex.get(), batch, n1, n2, c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform many 2D FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 *
 * The first dimension of a and c are considered batch dimensions
 */
template <typename A, typename C, cpp_enable_if(all_complex<A>::value)>
void fft2_many(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    static constexpr std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    mkl_detail::fft2_many_kernel(a.memory_start(), batch, n1, n2, c.memory_start());

    c.invalidate_gpu();
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
    a.ensure_cpu_up_to_date();

    static constexpr std::size_t N = decay_traits<A>::dimensions();

    std::size_t n1    = etl::dim<N - 2>(a);       //Size of the transform
    std::size_t n2    = etl::dim<N - 1>(a);       //Size of the transform
    std::size_t batch = etl::size(a) / (n1 * n2); //Number of batch

    mkl_detail::ifft2_many_kernel(safe_cast(a.memory_start()), batch, n1, n2, safe_cast(c.memory_start()));

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C, cpp_enable_if(all_complex<A>::value)>
void ifft2(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    mkl_detail::ifft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c.memory_start());

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D Inverse FFT on a and store the real part of the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void ifft2_real(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    auto c_complex = allocate<std::complex<value_t<C>>>(etl::size(a));

    mkl_detail::ifft2_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), c_complex.get());

    for (std::size_t i = 0; i < etl::size(a); ++i) {
        c[i] = c_complex[i].real();
    }

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full(I&& a, K&& b, C&& c) {
    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    mkl_detail::conv2_full_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), b.memory_start(), etl::dim<0>(b), etl::dim<1>(b), c.memory_start(), value_t<I>(0.0));

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c,
 * with the flipped kernels of b.
 *
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_flipped(I&& a, K&& b, C&& c) {
    a.ensure_cpu_up_to_date();
    b.ensure_cpu_up_to_date();

    etl::dyn_matrix<value_t<I>, 2> prepared_b(etl::dim<0>(b), etl::dim<1>(b));

    std::copy(b.memory_start(), b.memory_end(), prepared_b.memory_start());

    prepared_b.fflip_inplace();

    mkl_detail::conv2_full_kernel(a.memory_start(), etl::dim<0>(a), etl::dim<1>(a), prepared_b.memory_start(), etl::dim<0>(b), etl::dim<1>(b), c.memory_start(), value_t<I>(0.0));

    c.invalidate_gpu();
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi(I&& input, K&& kernel, C&& conv) {
    using T = value_t<I>;

    const auto KK = etl::dim<0>(kernel);

    if (KK) {
        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        const auto k_s = etl::dim<1>(kernel) * etl::dim<2>(kernel);
        const auto c_s = etl::dim<1>(conv) * etl::dim<2>(conv);

        const auto m1 = etl::dim<0>(input);
        const auto m2 = etl::dim<1>(input);

        const auto n1 = etl::dim<1>(kernel);
        const auto n2 = etl::dim<2>(kernel);

        const std::size_t s1   = m1 + n1 - 1;
        const std::size_t s2   = m2 + n2 - 1;
        const std::size_t size = s1 * s2;

        dyn_vector<etl::complex<T>> a_padded(size);

        for (std::size_t i = 0; i < m1; ++i) {
            direct_copy_n(input.memory_start() + i * m2, a_padded.memory_start() + i * s2, m2);
        }

        mkl_detail::inplace_fft2_kernel(safe_cast(a_padded.memory_start()), s1, s2);

        auto batch_fun_k = [&](const size_t first, const size_t last) {
            SERIAL_SECTION {
                for (std::size_t k = first; k < last; ++k) {
                    const T* b = kernel.memory_start() + k * k_s;
                    T* c       = conv.memory_start() + k * c_s;

                    dyn_vector<etl::complex<T>> b_padded(size);

                    for (std::size_t i = 0; i < n1; ++i) {
                        direct_copy_n(b + i * n2, b_padded.memory_start() + i * s2, n2);
                    }

                    mkl_detail::inplace_fft2_kernel(safe_cast(b_padded.memory_start()), s1, s2);

                    b_padded >>= a_padded;

                    mkl_detail::inplace_ifft2_kernel(reinterpret_cast<std::complex<T>*>(b_padded.memory_start()), s1, s2);

                    for (std::size_t i = 0; i < size; ++i) {
                        c[i] = b_padded[i].real;
                    }
                }
            }
        };

        if (etl::is_parallel) {
            dispatch_1d_any(select_parallel(KK, 2), batch_fun_k, 0, KK);
        } else {
            batch_fun_k(0, KK);
        }

        conv.invalidate_gpu();
    }
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi_flipped(I&& input, K&& kernel, C&& conv) {
    using T = value_t<I>;

    kernel.ensure_cpu_up_to_date();

    etl::dyn_matrix<T, 3> prepared_k(etl::dim<0>(kernel), etl::dim<1>(kernel), etl::dim<2>(kernel));

    std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

    prepared_k.deep_fflip_inplace();

    conv2_full_multi(input, prepared_k, conv);
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_full(I&& input, KK&& kernel, CC&& conv) {
    using T = value_t<I>;

    if (etl::dim<1>(kernel) > 0) {
        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        auto conv_i_inc = etl::dim<1>(conv) * etl::dim<2>(conv) * etl::dim<3>(conv);
        auto conv_c_inc = etl::dim<2>(conv) * etl::dim<3>(conv);

        auto kernel_k_inc = etl::dim<1>(kernel) * etl::dim<2>(kernel) * etl::dim<3>(kernel);
        auto kernel_c_inc = etl::dim<2>(kernel) * etl::dim<3>(kernel);

        auto input_i_inc = etl::dim<1>(input) * etl::dim<2>(input) * etl::dim<3>(input);
        auto input_k_inc = etl::dim<2>(input) * etl::dim<3>(input);

        const auto N = etl::dim<0>(input);
        const auto K = etl::dim<0>(kernel);
        const auto C = etl::dim<1>(kernel);

        const auto m1 = etl::dim<2>(input);
        const auto m2 = etl::dim<3>(input);

        const auto n1 = etl::dim<2>(kernel);
        const auto n2 = etl::dim<3>(kernel);

        const auto s1   = m1 + n1 - 1;
        const auto s2   = m2 + n2 - 1;
        const auto size = s1 * s2;

        std::fill(conv.memory_start(), conv.memory_end(), 0);

        dyn_matrix<etl::complex<T>, 3> b_padded(K, C, size);

        auto batch_fun_kc = [&](const size_t first, const size_t last) {
            for (std::size_t kc = first; kc < last; ++kc) {
                size_t k = kc / C;
                size_t c = kc % C;

                const T* b = kernel.memory_start() + k * kernel_k_inc + c * kernel_c_inc; // kernel(k)(c)

                b_padded(k)(c) = 0;
                for (std::size_t i = 0; i < n1; ++i) {
                    direct_copy_n(b + i * n2, b_padded(k)(c).memory_start() + i * s2, n2);
                }

                mkl_detail::inplace_fft2_kernel(safe_cast(b_padded(k)(c).memory_start()), s1, s2);
            }
        };

        auto batch_fun_n = [&](const size_t first, const size_t last) {
            if (last - first) {
                SERIAL_SECTION {
                    for (std::size_t i = first; i < last; ++i) {
                        for (std::size_t k = 0; k < K; ++k) {
                            const T* a = input.memory_start() + i * input_i_inc + k * input_k_inc; // input(i)(k)

                            dyn_vector<etl::complex<T>> a_padded(size);
                            dyn_vector<etl::complex<T>> tmp(size);

                            a_padded = 0;

                            for (std::size_t i = 0; i < m1; ++i) {
                                direct_copy_n(a + i * m2, a_padded.memory_start() + i * s2, m2);
                            }

                            mkl_detail::inplace_fft2_kernel(safe_cast(a_padded.memory_start()), s1, s2);

                            for (std::size_t c = 0; c < C; ++c) {
                                T* cc      = conv.memory_start() + i * conv_i_inc + c * conv_c_inc;       // conv(i)(c)

                                tmp = a_padded >> b_padded(k)(c);

                                mkl_detail::inplace_ifft2_kernel(safe_cast(tmp.memory_start()), s1, s2);

                                for (std::size_t i = 0; i < size; ++i) {
                                    cc[i] += tmp[i].real;
                                }
                            }
                        }
                    }
                }
            }
        };

        if (etl::is_parallel) {
            dispatch_1d_any(select_parallel(K * C, 2), batch_fun_kc, 0, K * C);
            dispatch_1d_any(select_parallel(N, 2), batch_fun_n, 0, N);
        } else {
            batch_fun_kc(0, K * C);
            batch_fun_n(0, N);
        }

        conv.invalidate_gpu();
    }
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_full_flipped(I&& input, K&& kernel, C&& conv) {
    if (etl::dim<1>(kernel) > 0) {
        kernel.ensure_cpu_up_to_date(); // Need for flipping

        etl::dyn_matrix<value_t<I>, 4> prepared_k(etl::dim<0>(kernel), etl::dim<1>(kernel), etl::dim<2>(kernel), etl::dim<3>(kernel));

        std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

        prepared_k.deep_fflip_inplace();

        conv4_full(input, prepared_k, conv);
    }
}

/*!
 * \brief FFT implementation of a 2D 'valid' convolution C = I * K, with multiple kernels.
 *
 * This works by doing a full convolution by FFT and then extracting
 * only the valid part of the convolution.
 *
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const size_t K = etl::dim<0>(kernels);
    const size_t i1 = etl::dim<0>(input);
    const size_t i2 = etl::dim<1>(input);
    const size_t k1 = etl::dim<1>(kernels);
    const size_t k2 = etl::dim<2>(kernels);

    // Dimensions of the final valid convolution (stride,padding)
    const size_t c1 = (i1 - k1 + 2 * p1) / s1 + 1;
    const size_t c2 = (i2 - k2 + 2 * p2) / s1 + 1;

    //Dimensions of the valid convolution (unit strided)
    const size_t v1 = (i1 - k1 + 2 * p1) + 1;
    const size_t v2 = (i2 - k2 + 2 * p2) + 1;

    // Dimensions of the full convolution
    const size_t t1 = (i1 + k1 + 2 * p1) - 1;
    const size_t t2 = (i2 + k2 + 2 * p2) - 1;

    // Dimensions of the 'full' borders
    const size_t b1 = (t1 - v1) / 2;
    const size_t b2 = (t2 - v2) / 2;

    input.ensure_cpu_up_to_date();
    kernels.ensure_cpu_up_to_date();

    etl::dyn_matrix<etl::complex<value_t<I>>> input_padded(t1, t2);
    etl::dyn_matrix<etl::complex<value_t<I>>, 3> kernels_padded(K, t1, t2);

    impl::common::pad_2d_input(input, input_padded, p1, p2);
    impl::common::complex_pad_3d(kernels, kernels_padded);

    mkl_detail::inplace_fft2_kernel(safe_cast(input_padded.memory_start()), t1, t2);
    mkl_detail::inplace_fft2_many_kernel(safe_cast(kernels_padded.memory_start()), K, t1, t2);

    for (size_t k = 0; k < K; ++k) {
        kernels_padded(k) >>= input_padded;
    }

    mkl_detail::inplace_ifft2_many_kernel(safe_cast(kernels_padded.memory_start()), K, t1, t2);

    for (size_t k = 0; k < K; ++k) {
        for (size_t i = 0; i < c1; ++i) {
            for (size_t j = 0; j < c2; ++j) {
                conv(k, i, j) = kernels_padded(k, i * s1 + b1, j * s2 + b2).real;
            }
        }
    }

    conv.invalidate_gpu();
}

/*!
 * \brief MKL FFT implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi_flipped(I&& input, K_T&& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    auto kernels_f = etl::force_temporary(kernels);

    kernels_f.deep_fflip_inplace();

    // TODO It would be faster to do the flip while padding
    fft_conv2_valid_multi(input, kernels_f, conv, s1, s2, p1, p2);
}

#else

//COVERAGE_EXCLUDE_BEGIN

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
void conv1_full(A&& a, B&& b, C&& c) {
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
void conv2_full(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full_multi(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 2D full convolution of a with multiple kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full_multi_flipped(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 2D full convolution of a with b and store the result in c,
 * with the flipped kernels of b.
 *
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv2_full_flipped(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 4D full convolution of a with b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv4_full(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief Perform the 4D full convolution of a with the flipped kernels of b and store the result in c
 * \param a The input matrix
 * \param b The kernel matrix
 * \param c The output matrix
 */
template <typename A, typename B, typename C>
void conv4_full_flipped(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief FFT implementation of a 2D 'valid' convolution C = I * K, with multiple kernels.
 *
 * This works by doing a full convolution by FFT and then extracting
 * only the valid part of the convolution.
 *
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi(const I& a, const K_T& b, C&& c, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

/*!
 * \brief MKL FFT implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi_flipped(I&& a, K_T&& b, C&& c, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unused(s1);
    cpp_unused(s2);
    cpp_unused(p1);
    cpp_unused(p2);
    cpp_unreachable("Unsupported feature called: mkl fft");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace blas

} //end of namespace impl

} //end of namespace etl
