//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

template <typename T>
using cublas_type = std::conditional_t<
    is_complex_single_t<T>::value,
    cuComplex,
    std::conditional_t<
        is_complex_double_t<T>::value,
        cuDoubleComplex,
        T>>;

/*!
 * \brief Create the default multiplication of the given type
 * \param value The value to get in return
 */
template<typename T>
T make_default(double value);

/*!
 * \copydoc make_default
 */
template<>
inline float make_default<float>(double value){
    return value;
}

/*!
 * \copydoc make_default
 */
template<>
inline double make_default<double>(double value){
    return value;
}

/*!
 * \copydoc make_default
 */
template<>
inline cuComplex make_default<cuComplex>(double value){
    return {float(value), 0.0f};
}

/*!
 * \copydoc make_default
 */
template<>
inline cuDoubleComplex make_default<cuDoubleComplex>(double value){
    return {value, 0.0};
}

inline void cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                 size_t m, size_t n, size_t k,
                 const float* alpha,
                 const float* A, size_t lda,
                 const float* B, size_t ldb,
                 const float* beta,
                 float* C, size_t ldc){
    cublas_check(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

inline void cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                 size_t m, size_t n, size_t k,
                 const double* alpha,
                 const double* A, size_t lda,
                 const double* B, size_t ldb,
                 const double* beta,
                 double* C, size_t ldc){
    cublas_check(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

inline void cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                 size_t m, size_t n, size_t k,
                 const cuComplex* alpha,
                 const std::complex<float>* A, size_t lda,
                 const std::complex<float>* B, size_t ldb,
                 cuComplex* beta,
                 std::complex<float>* C, size_t ldc){
    cublas_check(cublasCgemm(handle, transa, transb, m, n, k, alpha,
        reinterpret_cast<const cuComplex*>(A), lda,
        reinterpret_cast<const cuComplex*>(B), ldb, beta,
        reinterpret_cast<cuComplex*>(C), ldc));
}

inline void cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                 size_t m, size_t n, size_t k,
                 const cuDoubleComplex* alpha,
                 const std::complex<double>* A, size_t lda,
                 const std::complex<double>* B, size_t ldb,
                 cuDoubleComplex* beta,
                 std::complex<double>* C, size_t ldc){
    cublas_check(cublasZgemm(handle, transa, transb, m, n, k, alpha,
        reinterpret_cast<const cuDoubleComplex*>(A), lda,
        reinterpret_cast<const cuDoubleComplex*>(B), ldb, beta,
        reinterpret_cast<cuDoubleComplex*>(C), ldc));
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    using VT = value_t<A>;
    using T = cublas_type<VT>;

    auto alpha = make_default<T>(1.0);
    auto beta  = make_default<T>(0.0);

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

    // Do the actual multiplication

    if (row_major) {
        cublas_gemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::columns(c), etl::rows(c), etl::columns(a),
            &alpha,
            b.gpu_memory(), etl::major_stride(b),
            a.gpu_memory(), etl::major_stride(a),
            &beta,
            c.gpu_memory(), etl::major_stride(c));
    } else {
        cublas_gemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            a.gpu_memory(), etl::major_stride(a),
            b.gpu_memory(), etl::major_stride(b),
            &beta,
            c.gpu_memory(), etl::major_stride(c));
    }

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm_nt(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    using VT = value_t<A>;
    using T = cublas_type<VT>;

    auto alpha = make_default<T>(1.0);
    auto beta  = make_default<T>(0.0);

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

    // Do the actual multiplication

    if (row_major) {
        cublas_gemm(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            etl::columns(c), etl::rows(c), etl::columns(a),
            &alpha,
            b.gpu_memory(), etl::major_stride(b),
            a.gpu_memory(), etl::major_stride(a),
            &beta,
            c.gpu_memory(), etl::major_stride(c));
    } else {
        cublas_gemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            etl::rows(c), etl::columns(c), etl::columns(a),
            &alpha,
            a.gpu_memory(), etl::major_stride(a),
            b.gpu_memory(), etl::major_stride(b),
            &beta,
            c.gpu_memory(), etl::major_stride(c));
    }

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm_tn(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    using VT = value_t<A>;
    using T = cublas_type<VT>;

    auto alpha = make_default<T>(1.0);
    auto beta  = make_default<T>(0.0);

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

    // Do the actual multiplication

    if (row_major) {
        cublas_gemm(
            handle.get(),
            CUBLAS_OP_N, CUBLAS_OP_T,
            etl::columns(c), etl::rows(c), etl::rows(a),
            &alpha,
            b.gpu_memory(), etl::major_stride(b),
            a.gpu_memory(), etl::major_stride(a),
            &beta,
            c.gpu_memory(), etl::major_stride(c));
    } else {
        cublas_gemm(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            etl::rows(c), etl::columns(c), etl::rows(a),
            &alpha,
            a.gpu_memory(), etl::major_stride(a),
            b.gpu_memory(), etl::major_stride(b),
            &beta,
            c.gpu_memory(), etl::major_stride(c));
    }

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm_tt(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

    using VT = value_t<A>;
    using T = cublas_type<VT>;

    auto alpha = make_default<T>(1.0);
    auto beta  = make_default<T>(0.0);

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

    // Do the actual multiplication

    if (row_major) {
        cublas_gemm(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_T,
            etl::columns(c), etl::rows(c), etl::rows(a),
            &alpha,
            b.gpu_memory(), etl::major_stride(b),
            a.gpu_memory(), etl::major_stride(a),
            &beta,
            c.gpu_memory(), etl::major_stride(c));
    } else {
        cublas_gemm(
            handle.get(),
            CUBLAS_OP_T, CUBLAS_OP_T,
            etl::rows(c), etl::columns(c), etl::rows(a),
            &alpha,
            a.gpu_memory(), etl::major_stride(a),
            b.gpu_memory(), etl::major_stride(b),
            &beta,
            c.gpu_memory(), etl::major_stride(c));
    }

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

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

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

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

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_single_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

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

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_double_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

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

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

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

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

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

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_single_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

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

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_double_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    decltype(auto) handle = start_cublas();

    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

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

    c.validate_gpu();
    c.invalidate_cpu();
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm_nt(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm_tn(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm_tt(A&& a, B&& b, C&& c) {
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
void gemv(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gevm(A&& a, B&& b, C&& c) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace cublas

} //end of namespace impl

} //end of namespace etl
