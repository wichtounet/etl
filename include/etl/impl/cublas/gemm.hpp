//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "../../config.hpp"

#ifdef ETL_CUBLAS_MODE

#include "cuda.hpp"
#include "cublas.hpp"

#endif

namespace etl {

namespace impl {

namespace cublas {

#ifdef ETL_CUBLAS_MODE

template<typename A, typename B, typename C>
void sgemm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    float alpha = 1.0;
    float beta = 0.0;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    // Do the actual multiplication

    if(row_major){
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_T,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            gpu_a.get(), etl::columns(a),
            gpu_b.get(), etl::columns(b),
            &beta,
            gpu_c.get(), etl::rows(c));
    } else {
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            gpu_a.get(), etl::rows(a),
            gpu_b.get(), etl::rows(b),
            &beta,
            gpu_c.get(), etl::rows(c));
    }

    //Copy the result from GPU to CPU

    if(decay_traits<C>::storage_order == order::RowMajor){
        auto gpu_d = cuda_allocate(c);

        cublasSgeam(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c),
            &alpha,
            gpu_c.get(), etl::columns(c),
            &beta,
            gpu_d.get(), etl::columns(c),
            gpu_d.get(), etl::columns(c));

        cudaMemcpy(c.memory_start(), gpu_d.get(), etl::size(c) * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

template<typename A, typename B, typename C>
void dgemm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    double alpha = 1.0;
    double beta = 0.0;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    // Do the actual multiplication

    if(row_major){
        cublasDgemm(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_T,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            gpu_a.get(), etl::columns(a),
            gpu_b.get(), etl::columns(b),
            &beta,
            gpu_c.get(), etl::rows(c));
    } else {
        cublasDgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            gpu_a.get(), etl::rows(a),
            gpu_b.get(), etl::rows(b),
            &beta,
            gpu_c.get(), etl::rows(c));
    }

    //Copy the result from GPU to CPU

    if(decay_traits<C>::storage_order == order::RowMajor){
        auto gpu_d = cuda_allocate(c);

        cublasDgeam(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c),
            &alpha,
            gpu_c.get(), etl::columns(c),
            &beta,
            gpu_d.get(), etl::columns(c),
            gpu_d.get(), etl::columns(c));

        cudaMemcpy(c.memory_start(), gpu_d.get(), etl::size(c) * sizeof(double), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(double), cudaMemcpyDeviceToHost);
    }
}

template<typename A, typename B, typename C>
void cgemm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    cuComplex alpha = make_cuComplex(1.0, 0.0);
    cuComplex beta = make_cuComplex(0.0, 0.0);

    // Do the actual multiplication

    if(row_major){
        cublasCgemm(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_T,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            reinterpret_cast<cuComplex*>(gpu_a.get()), etl::columns(a),
            reinterpret_cast<cuComplex*>(gpu_b.get()), etl::columns(b),
            &beta,
            reinterpret_cast<cuComplex*>(gpu_c.get()), etl::rows(c));
    } else {
        cublasCgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            reinterpret_cast<cuComplex*>(gpu_a.get()), etl::rows(a),
            reinterpret_cast<cuComplex*>(gpu_b.get()), etl::rows(b),
            &beta,
            reinterpret_cast<cuComplex*>(gpu_c.get()), etl::rows(c));
    }

    //Copy the result from GPU to CPU

    if(decay_traits<C>::storage_order == order::RowMajor){
        auto gpu_d = cuda_allocate(c);

        cublasCgeam(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c),
            &alpha,
            reinterpret_cast<cuComplex*>(gpu_c.get()), etl::columns(c),
            &beta,
            reinterpret_cast<cuComplex*>(gpu_d.get()), etl::columns(c),
            reinterpret_cast<cuComplex*>(gpu_d.get()), etl::columns(c));

        cudaMemcpy(c.memory_start(), gpu_d.get(), etl::size(c) * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    }
}

template<typename A, typename B, typename C>
void zgemm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    // Do the actual multiplication

    if(row_major){
        cublasZgemm(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_T,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(gpu_a.get()), etl::columns(a),
            reinterpret_cast<cuDoubleComplex*>(gpu_b.get()), etl::columns(b),
            &beta,
            reinterpret_cast<cuDoubleComplex*>(gpu_c.get()), etl::rows(c));
    } else {
        cublasZgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(gpu_a.get()), etl::rows(a),
            reinterpret_cast<cuDoubleComplex*>(gpu_b.get()), etl::rows(b),
            &beta,
            reinterpret_cast<cuDoubleComplex*>(gpu_c.get()), etl::rows(c));
    }

    //Copy the result from GPU to CPU

    if(decay_traits<C>::storage_order == order::RowMajor){
        auto gpu_d = cuda_allocate(c);

        cublasZgeam(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(gpu_c.get()), etl::columns(c),
            &beta,
            reinterpret_cast<cuDoubleComplex*>(gpu_d.get()), etl::columns(c),
            reinterpret_cast<cuDoubleComplex*>(gpu_d.get()), etl::columns(c));

        cudaMemcpy(c.memory_start(), gpu_d.get(), etl::size(c) * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
    }
}

#else

template<typename A, typename B, typename C>
void sgemm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void dgemm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void cgemm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void zgemm(A&& a, B&& b, C&& c);

#endif

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
