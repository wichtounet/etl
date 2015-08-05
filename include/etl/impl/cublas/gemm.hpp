//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/config.hpp"

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"

#endif

namespace etl {

namespace impl {

namespace cublas {

#ifdef ETL_CUBLAS_MODE

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

inline void cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
        const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc){
    cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline void cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
        const double *alpha, const double *A, int lda, const double *beta, const double *B, int ldb, double *C, int ldc){
    cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline void cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
        const cfloat *alpha, const cfloat *A, int lda, const cfloat *beta, const cfloat *B, int ldb, cfloat *C, int ldc){
    cublasCgeam(handle, transa, transb, m, n, reinterpret_cast<const cuComplex*>(alpha),
        reinterpret_cast<const cuComplex*>(A), lda,
        reinterpret_cast<const cuComplex*>(beta), reinterpret_cast<const cuComplex*>(B), ldb,
        reinterpret_cast<cuComplex*>(C), ldc);
}

inline void cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
        const cdouble *alpha, const cdouble *A, int lda, const cdouble *beta,
        const cdouble *B, int ldb, cdouble *C, int ldc){
    cublasZgeam(handle, transa, transb, m, n, reinterpret_cast<const cuDoubleComplex*>(alpha),
        reinterpret_cast<const cuDoubleComplex*>(A), lda,
        reinterpret_cast<const cuDoubleComplex*>(beta), reinterpret_cast<const cuDoubleComplex*>(B), ldb,
        reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

template<typename C, typename T>
void copy_matrix(cublas_handle& handle, cuda_memory<T>& gpu, C&& c){
    if(decay_traits<C>::storage_order == order::RowMajor){
        auto gpu_d = cuda_allocate(c);

        T alpha = 1.0;
        T beta = 0.0;

        cublas_geam(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c),
            &alpha,
            gpu.get(), etl::columns(c),
            &beta,
            gpu_d.get(), etl::columns(c),
            gpu_d.get(), etl::columns(c));

        cudaMemcpy(c.memory_start(), gpu_d.get(), etl::size(c) * sizeof(T), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(c.memory_start(), gpu.get(), etl::size(c) * sizeof(T), cudaMemcpyDeviceToHost);
    }
}

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

    copy_matrix(handle, gpu_c, c);
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

    copy_matrix(handle, gpu_c, c);
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

    copy_matrix(handle, gpu_c, c);
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

    copy_matrix(handle, gpu_c, c);
}

template<typename A, typename B, typename C>
void sgemv(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    float alpha = 1.0;
    float beta = 0.0;

    //Perform the actual multiplication

    if(row_major){
        cublasSgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(a), etl::columns(a),
            &alpha,
            gpu_a.get(), major_stride(a),
            gpu_b.get(), 1,
            &beta,
            gpu_c.get(), 1
        );
    } else {
        cublasSgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(a), etl::columns(a),
            &alpha,
            gpu_a.get(), major_stride(a),
            gpu_b.get(), 1,
            &beta,
            gpu_c.get(), 1
        );
    }

    //Copy the result from GPU to CPU

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(float), cudaMemcpyDeviceToHost);
}

template<typename A, typename B, typename C>
void dgemv(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    double alpha = 1.0;
    double beta = 0.0;

    //Perform the actual multiplication

    if(row_major){
        cublasDgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(a), etl::columns(a),
            &alpha,
            gpu_a.get(), major_stride(a),
            gpu_b.get(), 1,
            &beta,
            gpu_c.get(), 1
        );
    } else {
        cublasDgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(a), etl::columns(a),
            &alpha,
            gpu_a.get(), major_stride(a),
            gpu_b.get(), 1,
            &beta,
            gpu_c.get(), 1
        );
    }

    //Copy the result from GPU to CPU

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(double), cudaMemcpyDeviceToHost);
}

template<typename A, typename B, typename C>
void cgemv(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    cuComplex alpha = make_cuComplex(1.0, 0.0);
    cuComplex beta = make_cuComplex(0.0, 0.0);

    //Perform the actual multiplication

    if(row_major){
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(a), etl::columns(a),
            &alpha,
            reinterpret_cast<cuComplex*>(gpu_a.get()), major_stride(a),
            reinterpret_cast<cuComplex*>(gpu_b.get()), 1,
            &beta,
            reinterpret_cast<cuComplex*>(gpu_c.get()), 1
        );
    } else {
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(a), etl::columns(a),
            &alpha,
            reinterpret_cast<cuComplex*>(gpu_a.get()), major_stride(a),
            reinterpret_cast<cuComplex*>(gpu_b.get()), 1,
            &beta,
            reinterpret_cast<cuComplex*>(gpu_c.get()), 1
        );
    }

    //Copy the result from GPU to CPU

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
}

template<typename A, typename B, typename C>
void zgemv(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    //Perform the actual multiplication

    if(row_major){
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(a), etl::columns(a),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(gpu_a.get()), major_stride(a),
            reinterpret_cast<cuDoubleComplex*>(gpu_b.get()), 1,
            &beta,
            reinterpret_cast<cuDoubleComplex*>(gpu_c.get()), 1
        );
    } else {
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(a), etl::columns(a),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(gpu_a.get()), major_stride(a),
            reinterpret_cast<cuDoubleComplex*>(gpu_b.get()), 1,
            &beta,
            reinterpret_cast<cuDoubleComplex*>(gpu_c.get()), 1
        );
    }

    //Copy the result from GPU to CPU

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
}

template<typename A, typename B, typename C>
void sgevm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    float alpha = 1.0;
    float beta = 0.0;

    //Perform the actual multiplication

    if(row_major){
        cublasSgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(b), etl::columns(b),
            &alpha,
            gpu_b.get(), major_stride(b),
            gpu_a.get(), 1,
            &beta,
            gpu_c.get(), 1
        );
    } else {
        cublasSgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(b), etl::columns(b),
            &alpha,
            gpu_b.get(), major_stride(b),
            gpu_a.get(), 1,
            &beta,
            gpu_c.get(), 1
        );
    }

    //Copy the result from GPU to CPU

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(float), cudaMemcpyDeviceToHost);
}

template<typename A, typename B, typename C>
void dgevm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    double alpha = 1.0;
    double beta = 0.0;

    //Perform the actual multiplication

    if(row_major){
        cublasDgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(b), etl::columns(b),
            &alpha,
            gpu_b.get(), major_stride(b),
            gpu_a.get(), 1,
            &beta,
            gpu_c.get(), 1
        );
    } else {
        cublasDgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(b), etl::columns(b),
            &alpha,
            gpu_b.get(), major_stride(b),
            gpu_a.get(), 1,
            &beta,
            gpu_c.get(), 1
        );
    }

    //Copy the result from GPU to CPU

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(double), cudaMemcpyDeviceToHost);
}

template<typename A, typename B, typename C>
void cgevm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    cuComplex alpha = make_cuComplex(1.0, 0.0);
    cuComplex beta = make_cuComplex(0.0, 0.0);

    //Perform the actual multiplication

    if(row_major){
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(b), etl::columns(b),
            &alpha,
            reinterpret_cast<cuComplex*>(gpu_b.get()), major_stride(b),
            reinterpret_cast<cuComplex*>(gpu_a.get()), 1,
            &beta,
            reinterpret_cast<cuComplex*>(gpu_c.get()), 1
        );
    } else {
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(b), etl::columns(b),
            &alpha,
            reinterpret_cast<cuComplex*>(gpu_b.get()), major_stride(b),
            reinterpret_cast<cuComplex*>(gpu_a.get()), 1,
            &beta,
            reinterpret_cast<cuComplex*>(gpu_c.get()), 1
        );
    }

    //Copy the result from GPU to CPU

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
}

template<typename A, typename B, typename C>
void zgevm(A&& a, B&& b, C&& c){
    auto handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    auto gpu_a = cuda_allocate_copy(a);
    auto gpu_b = cuda_allocate_copy(b);
    auto gpu_c = cuda_allocate(c);

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

    //Perform the actual multiplication

    if(row_major){
        cublasZgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(b), etl::columns(b),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(gpu_b.get()), major_stride(b),
            reinterpret_cast<cuDoubleComplex*>(gpu_a.get()), 1,
            &beta,
            reinterpret_cast<cuDoubleComplex*>(gpu_c.get()), 1
        );
    } else {
        cublasZgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(b), etl::columns(b),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(gpu_b.get()), major_stride(b),
            reinterpret_cast<cuDoubleComplex*>(gpu_a.get()), 1,
            &beta,
            reinterpret_cast<cuDoubleComplex*>(gpu_c.get()), 1
        );
    }

    //Copy the result from GPU to CPU

    cudaMemcpy(c.memory_start(), gpu_c.get(), etl::size(c) * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
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

template<typename A, typename B, typename C>
void sgemv(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void dgemv(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void cgemv(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void zgemv(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void sgevm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void dgevm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void cgevm(A&& a, B&& b, C&& c);

template<typename A, typename B, typename C>
void zgevm(A&& a, B&& b, C&& c);

#endif

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
