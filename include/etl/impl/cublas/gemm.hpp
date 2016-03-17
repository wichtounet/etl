//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"

#endif

namespace etl {

namespace impl {

namespace cublas {

#ifdef ETL_CUBLAS_MODE

using cfloat  = std::complex<float>;
using cdouble = std::complex<double>;

inline void cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                        const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc) {
    cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline void cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                        const double* alpha, const double* A, int lda, const double* beta, const double* B, int ldb, double* C, int ldc) {
    cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline void cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                        const cfloat* alpha, const cfloat* A, int lda, const cfloat* beta, const cfloat* B, int ldb, cfloat* C, int ldc) {
    cublasCgeam(handle, transa, transb, m, n, reinterpret_cast<const cuComplex*>(alpha),
                reinterpret_cast<const cuComplex*>(A), lda,
                reinterpret_cast<const cuComplex*>(beta), reinterpret_cast<const cuComplex*>(B), ldb,
                reinterpret_cast<cuComplex*>(C), ldc);
}

inline void cublas_geam(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                        const cdouble* alpha, const cdouble* A, int lda, const cdouble* beta,
                        const cdouble* B, int ldb, cdouble* C, int ldc) {
    cublasZgeam(handle, transa, transb, m, n, reinterpret_cast<const cuDoubleComplex*>(alpha),
                reinterpret_cast<const cuDoubleComplex*>(A), lda,
                reinterpret_cast<const cuDoubleComplex*>(beta), reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_single_precision<A, B, C>::value)>
void gemm(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    float alpha = 1.0;
    float beta  = 0.0;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    // Do the actual multiplication

    if (row_major) {
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::columns(b), etl::rows(a), etl::columns(a),
            &alpha,
            b.gpu_memory(), etl::columns(b),
            a.gpu_memory(), etl::columns(a),
            &beta,
            c.gpu_memory(), etl::columns(b));
    } else {
        cublasSgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            a.gpu_memory(), etl::rows(a),
            b.gpu_memory(), etl::rows(b),
            &beta,
            c.gpu_memory(), etl::rows(c));
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_double_precision<A, B, C>::value)>
void gemm(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    double alpha = 1.0;
    double beta  = 0.0;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    // Do the actual multiplication

    if (row_major) {
        cublasDgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::columns(b), etl::rows(a), etl::columns(a),
            &alpha,
            b.gpu_memory(), etl::columns(b),
            a.gpu_memory(), etl::columns(a),
            &beta,
            c.gpu_memory(), etl::columns(b));
    } else {
        cublasDgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            a.gpu_memory(), etl::rows(a),
            b.gpu_memory(), etl::rows(b),
            &beta,
            c.gpu_memory(), etl::rows(c));
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_complex_single_precision<A, B, C>::value)>
void gemm(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    cuComplex alpha = make_cuComplex(1.0, 0.0);
    cuComplex beta  = make_cuComplex(0.0, 0.0);

    // Do the actual multiplication

    if (row_major) {
        cublasCgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::columns(b), etl::rows(a), etl::columns(a),
            &alpha,
            reinterpret_cast<cuComplex*>(b.gpu_memory()), etl::columns(b),
            reinterpret_cast<cuComplex*>(a.gpu_memory()), etl::columns(a),
            &beta,
            reinterpret_cast<cuComplex*>(c.gpu_memory()), etl::columns(b));
    } else {
        cublasCgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            reinterpret_cast<cuComplex*>(a.gpu_memory()), etl::rows(a),
            reinterpret_cast<cuComplex*>(b.gpu_memory()), etl::rows(b),
            &beta,
            reinterpret_cast<cuComplex*>(c.gpu_memory()), etl::rows(c));
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_complex_double_precision<A, B, C>::value)>
void gemm(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);

    // Do the actual multiplication

    if (row_major) {
        cublasZgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::columns(b), etl::rows(a), etl::columns(a),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(b.gpu_memory()), etl::columns(b),
            reinterpret_cast<cuDoubleComplex*>(a.gpu_memory()), etl::columns(a),
            &beta,
            reinterpret_cast<cuDoubleComplex*>(c.gpu_memory()), etl::columns(b));
    } else {
        cublasZgemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(a.gpu_memory()), etl::rows(a),
            reinterpret_cast<cuDoubleComplex*>(b.gpu_memory()), etl::rows(b),
            &beta,
            reinterpret_cast<cuDoubleComplex*>(c.gpu_memory()), etl::rows(c));
    }
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_single_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    float alpha = 1.0;
    float beta  = 0.0;

    //Perform the actual multiplication

    if (row_major) {
        cublasSgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::columns(a), etl::rows(a),
            &alpha,
            a.gpu_memory(), major_stride(a),
            b.gpu_memory(), 1,
            &beta,
            c.gpu_memory(), 1);
    } else {
        cublasSgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(a), etl::columns(a),
            &alpha,
            a.gpu_memory(), major_stride(a),
            b.gpu_memory(), 1,
            &beta,
            c.gpu_memory(), 1);
    }

    //Copy the result from GPU to CPU

    c.gpu_copy_from();
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_double_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    double alpha = 1.0;
    double beta  = 0.0;

    //Perform the actual multiplication

    if (row_major) {
        cublasDgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::columns(a), etl::rows(a),
            &alpha,
            a.gpu_memory(), major_stride(a),
            b.gpu_memory(), 1,
            &beta,
            c.gpu_memory(), 1);
    } else {
        cublasDgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(a), etl::columns(a),
            &alpha,
            a.gpu_memory(), major_stride(a),
            b.gpu_memory(), 1,
            &beta,
            c.gpu_memory(), 1);
    }

    //Copy the result from GPU to CPU

    c.gpu_copy_from();
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_complex_single_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    cuComplex alpha = make_cuComplex(1.0, 0.0);
    cuComplex beta  = make_cuComplex(0.0, 0.0);

    //Perform the actual multiplication

    if (row_major) {
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::columns(a), etl::rows(a),
            &alpha,
            reinterpret_cast<cuComplex*>(a.gpu_memory()), major_stride(a),
            reinterpret_cast<cuComplex*>(b.gpu_memory()), 1,
            &beta,
            reinterpret_cast<cuComplex*>(c.gpu_memory()), 1);
    } else {
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(a), etl::columns(a),
            &alpha,
            reinterpret_cast<cuComplex*>(a.gpu_memory()), major_stride(a),
            reinterpret_cast<cuComplex*>(b.gpu_memory()), 1,
            &beta,
            reinterpret_cast<cuComplex*>(c.gpu_memory()), 1);
    }

    //Copy the result from GPU to CPU

    c.gpu_copy_from();
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_complex_double_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);

    //Perform the actual multiplication

    if (row_major) {
        cublasZgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::columns(a), etl::rows(a),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(a.gpu_memory()), major_stride(a),
            reinterpret_cast<cuDoubleComplex*>(b.gpu_memory()), 1,
            &beta,
            reinterpret_cast<cuDoubleComplex*>(c.gpu_memory()), 1);
    } else {
        cublasZgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::rows(a), etl::columns(a),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(a.gpu_memory()), major_stride(a),
            reinterpret_cast<cuDoubleComplex*>(b.gpu_memory()), 1,
            &beta,
            reinterpret_cast<cuDoubleComplex*>(c.gpu_memory()), 1);
    }

    //Copy the result from GPU to CPU

    c.gpu_copy_from();
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_single_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    float alpha = 1.0;
    float beta  = 0.0;

    //Perform the actual multiplication

    if (row_major) {
        cublasSgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::columns(b), etl::rows(b),
            &alpha,
            b.gpu_memory(), major_stride(b),
            a.gpu_memory(), 1,
            &beta,
            c.gpu_memory(), 1);
    } else {
        cublasSgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(b), etl::columns(b),
            &alpha,
            b.gpu_memory(), major_stride(b),
            a.gpu_memory(), 1,
            &beta,
            c.gpu_memory(), 1);
    }

    //Copy the result from GPU to CPU

    c.gpu_copy_from();
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_double_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    double alpha = 1.0;
    double beta  = 0.0;

    //Perform the actual multiplication

    if (row_major) {
        cublasDgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::columns(b), etl::rows(b),
            &alpha,
            b.gpu_memory(), major_stride(b),
            a.gpu_memory(), 1,
            &beta,
            c.gpu_memory(), 1);
    } else {
        cublasDgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(b), etl::columns(b),
            &alpha,
            b.gpu_memory(), major_stride(b),
            a.gpu_memory(), 1,
            &beta,
            c.gpu_memory(), 1);
    }

    //Copy the result from GPU to CPU

    c.gpu_copy_from();
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_complex_single_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    cuComplex alpha = make_cuComplex(1.0, 0.0);
    cuComplex beta  = make_cuComplex(0.0, 0.0);

    //Perform the actual multiplication

    if (row_major) {
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::columns(b), etl::rows(b),
            &alpha,
            reinterpret_cast<cuComplex*>(b.gpu_memory()), major_stride(b),
            reinterpret_cast<cuComplex*>(a.gpu_memory()), 1,
            &beta,
            reinterpret_cast<cuComplex*>(c.gpu_memory()), 1);
    } else {
        cublasCgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(b), etl::columns(b),
            &alpha,
            reinterpret_cast<cuComplex*>(b.gpu_memory()), major_stride(b),
            reinterpret_cast<cuComplex*>(a.gpu_memory()), 1,
            &beta,
            reinterpret_cast<cuComplex*>(c.gpu_memory()), 1);
    }

    //Copy the result from GPU to CPU

    c.gpu_copy_from();
}

template <typename A, typename B, typename C, cpp_enable_if(all_dma<A, B, C>::value, all_complex_double_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    cublas_handle handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.gpu_allocate_copy_if_necessary();
    b.gpu_allocate_copy_if_necessary();
    c.gpu_allocate_if_necessary();

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(0.0, 0.0);

    //Perform the actual multiplication

    if (row_major) {
        cublasZgemv(
            handle.get(),
            CUBLAS_OP_N,
            etl::columns(b), etl::rows(b),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(b.gpu_memory()), major_stride(b),
            reinterpret_cast<cuDoubleComplex*>(a.gpu_memory()), 1,
            &beta,
            reinterpret_cast<cuDoubleComplex*>(c.gpu_memory()), 1);
    } else {
        cublasZgemv(
            handle.get(),
            CUBLAS_OP_T,
            etl::rows(b), etl::columns(b),
            &alpha,
            reinterpret_cast<cuDoubleComplex*>(b.gpu_memory()), major_stride(b),
            reinterpret_cast<cuDoubleComplex*>(a.gpu_memory()), 1,
            &beta,
            reinterpret_cast<cuDoubleComplex*>(c.gpu_memory()), 1);
    }

    //Copy the result from GPU to CPU

    c.gpu_copy_from();
}

template <typename A, typename B, typename C, cpp_enable_if(!all_dma<A, B, C>::value)>
void gemm(A&&, B&& b, C&&);

template <typename A, typename B, typename C, cpp_enable_if(!all_dma<A, B, C>::value)>
void gemv(A&&, B&& b, C&&);

template <typename A, typename B, typename C, cpp_enable_if(!all_dma<A, B, C>::value)>
void gevm(A&&, B&& b, C&&);

#else

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm(A&&, B&&, C&&) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemv(A&&, B&&, C&&) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gevm(A&&, B&&, C&&) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

#endif

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
