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

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);
    auto gpu_d = cuda_allocate(c);

    float alpha = 1.0;
    float beta = 0.0;

    // Do the actual multiplication
    cublasSgemm(
        handle.get(),
        CUBLAS_OP_T, CUBLAS_OP_T,
        etl::rows(c), etl::columns(c), etl::columns(a),
        &alpha,
        gpu_a.get(), etl::columns(a),
        gpu_b.get(), etl::columns(b),
        &beta,
        gpu_c.get(), etl::rows(c));

    //gpu_d = gpu_c'
    cublasSgeam(
        handle.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        etl::rows(c), etl::columns(c),
        &alpha,
        gpu_c.get(), etl::columns(c),
        &beta,
        gpu_d.get(), etl::columns(c),
        gpu_d.get(), etl::columns(c));

    //C = gpu_d
    cudaMemcpy(c.memory_start(), gpu_d.get(), etl::size(c) * sizeof(float), cudaMemcpyDeviceToHost);
}

template<typename A, typename B, typename C>
void dgemm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);
    auto gpu_d = cuda_allocate(c);

    double alpha = 1.0;
    double beta = 0.0;

    // Do the actual multiplication
    cublasDgemm(
        handle.get(),
        CUBLAS_OP_T, CUBLAS_OP_T,
        etl::rows(c), etl::columns(c), etl::columns(a),
        &alpha,
        gpu_a.get(), etl::columns(a),
        gpu_b.get(), etl::columns(b),
        &beta,
        gpu_c.get(), etl::rows(c));

    //gpu_d = gpu_c'
    cublasDgeam(
        handle.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        etl::rows(c), etl::columns(c),
        &alpha,
        gpu_c.get(), etl::columns(c),
        &beta,
        gpu_d.get(), etl::columns(c),
        gpu_d.get(), etl::columns(c));

    //C = gpu_d
    cudaMemcpy(c.memory_start(), gpu_d.get(), etl::size(c) * sizeof(double), cudaMemcpyDeviceToHost);
}

#else

template<typename A, typename B, typename C>
void sgemm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void dgemm(A&& a, B&& b, C&& c);

#endif

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
