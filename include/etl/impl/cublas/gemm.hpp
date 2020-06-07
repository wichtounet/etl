//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/util/safe_cast.hpp"
#include "etl/impl/cublas/cublas.hpp"

#endif

namespace etl::impl::cublas {

#ifdef ETL_CUBLAS_MODE

/*!
 * \brief Helper to get the CUBLAS type from a floating point type
 */
template <typename T>
using cublas_type = std::conditional_t<is_complex_single_t<T>, cuComplex, std::conditional_t<is_complex_double_t<T>, cuDoubleComplex, T>>;

/*!
 * \brief Create the default multiplication of the given type
 * \param value The value to get in return
 */
template <typename T>
T make_default(double value);

/*!
 * \copydoc make_default
 */
template <>
inline float make_default<float>(double value) {
    return value;
}

/*!
 * \copydoc make_default
 */
template <>
inline double make_default<double>(double value) {
    return value;
}

/*!
 * \copydoc make_default
 */
template <>
inline cuComplex make_default<cuComplex>(double value) {
    return {float(value), 0.0f};
}

/*!
 * \copydoc make_default
 */
template <>
inline cuDoubleComplex make_default<cuDoubleComplex>(double value) {
    return {value, 0.0};
}

/*!
 * \brief Create the default multiplication of the given type
 * \param value The value to get in return
 */
template <typename CT, typename T>
inline CT cublas_convert(T value) {
    return value;
}

template <>
inline cuComplex cublas_convert(std::complex<float> value) {
    return {value.real(), value.imag()};
}

template <>
inline cuDoubleComplex cublas_convert(std::complex<double> value) {
    return {value.real(), value.imag()};
}

template <>
inline cuComplex cublas_convert(etl::complex<float> value) {
    return {value.real, value.imag};
}

template <>
inline cuDoubleComplex cublas_convert(etl::complex<double> value) {
    return {value.real, value.imag};
}

/*!
 * \brief Perform a GEMM operation with CUBLAS, overloaded version for single-precision
 * \param handle The handle to the CUBLAS library context
 * \param transa The operation op(A)
 * \param transb The operation op(B)
 * \param m The number of rows of matrix op(A) and C
 * \param n The number of columns of matrix op(B) and C
 * \param k The number of columsn of matrix op(a) and rows of op(B)
 * \param alpha Scalar used for multiplication of op(A)
 * \param A Pointer to the memory of matrix A
 * \param lda Leading dimension of the matrix A
 * \param B Pointer to the memory of matrix B
 * \param ldb Leading dimension of the matrix B
 * \param beta Scalar used for multiplication of C
 * \param C Pointer to the memory of matrix C
 * \param ldc Leading dimension of the matrix C
 */
inline void cublas_gemm(cublasHandle_t handle,
                        cublasOperation_t transa,
                        cublasOperation_t transb,
                        size_t m,
                        size_t n,
                        size_t k,
                        const float* alpha,
                        const float* A,
                        size_t lda,
                        const float* B,
                        size_t ldb,
                        const float* beta,
                        float* C,
                        size_t ldc) {
    cublas_check(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

/*!
 * \brief Perform a GEMM operation with CUBLAS, overloaded version for double-precision
 * \param handle The handle to the CUBLAS library context
 * \param transa The operation op(A)
 * \param transb The operation op(B)
 * \param m The number of rows of matrix op(A) and C
 * \param n The number of columns of matrix op(B) and C
 * \param k The number of columsn of matrix op(a) and rows of op(B)
 * \param alpha Scalar used for multiplication of op(A)
 * \param A Pointer to the memory of matrix A
 * \param lda Leading dimension of the matrix A
 * \param B Pointer to the memory of matrix B
 * \param ldb Leading dimension of the matrix B
 * \param beta Scalar used for multiplication of C
 * \param C Pointer to the memory of matrix C
 * \param ldc Leading dimension of the matrix C
 */
inline void cublas_gemm(cublasHandle_t handle,
                        cublasOperation_t transa,
                        cublasOperation_t transb,
                        size_t m,
                        size_t n,
                        size_t k,
                        const double* alpha,
                        const double* A,
                        size_t lda,
                        const double* B,
                        size_t ldb,
                        const double* beta,
                        double* C,
                        size_t ldc) {
    cublas_check(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

/*!
 * \brief Perform a GEMM operation with CUBLAS, overloaded version for complex single-precision
 * \param handle The handle to the CUBLAS library context
 * \param transa The operation op(A)
 * \param transb The operation op(B)
 * \param m The number of rows of matrix op(A) and C
 * \param n The number of columns of matrix op(B) and C
 * \param k The number of columsn of matrix op(a) and rows of op(B)
 * \param alpha Scalar used for multiplication of op(A)
 * \param A Pointer to the memory of matrix A
 * \param lda Leading dimension of the matrix A
 * \param B Pointer to the memory of matrix B
 * \param ldb Leading dimension of the matrix B
 * \param beta Scalar used for multiplication of C
 * \param C Pointer to the memory of matrix C
 * \param ldc Leading dimension of the matrix C
 */
inline void cublas_gemm(cublasHandle_t handle,
                        cublasOperation_t transa,
                        cublasOperation_t transb,
                        size_t m,
                        size_t n,
                        size_t k,
                        const cuComplex* alpha,
                        const std::complex<float>* A,
                        size_t lda,
                        const std::complex<float>* B,
                        size_t ldb,
                        cuComplex* beta,
                        std::complex<float>* C,
                        size_t ldc) {
    cublas_check(cublasCgemm(handle, transa, transb, m, n, k, alpha, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb,
                             beta, reinterpret_cast<cuComplex*>(C), ldc));
}

/*!
 * \brief Perform a GEMM operation with CUBLAS, overloaded version for complex double-precision
 * \param handle The handle to the CUBLAS library context
 * \param transa The operation op(A)
 * \param transb The operation op(B)
 * \param m The number of rows of matrix op(A) and C
 * \param n The number of columns of matrix op(B) and C
 * \param k The number of columsn of matrix op(a) and rows of op(B)
 * \param alpha Scalar used for multiplication of op(A)
 * \param A Pointer to the memory of matrix A
 * \param lda Leading dimension of the matrix A
 * \param B Pointer to the memory of matrix B
 * \param ldb Leading dimension of the matrix B
 * \param beta Scalar used for multiplication of C
 * \param C Pointer to the memory of matrix C
 * \param ldc Leading dimension of the matrix C
 */
inline void cublas_gemm(cublasHandle_t handle,
                        cublasOperation_t transa,
                        cublasOperation_t transb,
                        size_t m,
                        size_t n,
                        size_t k,
                        const cuDoubleComplex* alpha,
                        const std::complex<double>* A,
                        size_t lda,
                        const std::complex<double>* B,
                        size_t ldb,
                        cuDoubleComplex* beta,
                        std::complex<double>* C,
                        size_t ldc) {
    cublas_check(cublasZgemm(handle, transa, transb, m, n, k, alpha, reinterpret_cast<const cuDoubleComplex*>(A), lda,
                             reinterpret_cast<const cuDoubleComplex*>(B), ldb, beta, reinterpret_cast<cuDoubleComplex*>(C), ldc));
}

/*!
 * \brief Perform a GEMV operation with CUBLAS, overloaded version for single-precision
 * \param handle The handle to the CUBLAS library context
 * \param transa The operation op(A)
 * \param m The number of rows of matrix A
 * \param n The number of columns of matrix A
 * \param alpha Scalar used for multiplication of op(A)
 * \param A Pointer to the memory of matrix A
 * \param lda Leading dimension of the matrix A
 * \param B Pointer to the memory of vector B
 * \param ldb Stride of the vector B
 * \param beta Scalar used for multiplication of C
 * \param C Pointer to the memory of vector C
 * \param ldc Leading dimension of the vector C
 */
inline void cublas_gemv(cublasHandle_t handle,
                        cublasOperation_t trans,
                        size_t m,
                        size_t n,
                        const float* alpha,
                        const float* A,
                        size_t lda,
                        const float* B,
                        size_t ldb,
                        const float* beta,
                        float* C,
                        size_t ldc) {
    cublas_check(cublasSgemv(handle, trans, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
}

/*!
 * \brief Perform a GEMV operation with CUBLAS, overloaded version for double-precision
 * \param handle The handle to the CUBLAS library context
 * \param transa The operation op(A)
 * \param m The number of rows of matrix A
 * \param n The number of columns of matrix A
 * \param alpha Scalar used for multiplication of op(A)
 * \param A Pointer to the memory of matrix A
 * \param lda Leading dimension of the matrix A
 * \param B Pointer to the memory of vector B
 * \param ldb Stride of the vector B
 * \param beta Scalar used for multiplication of C
 * \param C Pointer to the memory of vector C
 * \param ldc Leading dimension of the vector C
 */
inline void cublas_gemv(cublasHandle_t handle,
                        cublasOperation_t trans,
                        size_t m,
                        size_t n,
                        const double* alpha,
                        const double* A,
                        size_t lda,
                        const double* B,
                        size_t ldb,
                        const double* beta,
                        double* C,
                        size_t ldc) {
    cublas_check(cublasDgemv(handle, trans, m, n, alpha, A, lda, B, ldb, beta, C, ldc));
}

/*!
 * \brief Perform a GEMV operation with CUBLAS, overloaded version for complex single-precision
 * \param handle The handle to the CUBLAS library context
 * \param transa The operation op(A)
 * \param m The number of rows of matrix A
 * \param n The number of columns of matrix A
 * \param alpha Scalar used for multiplication of op(A)
 * \param A Pointer to the memory of matrix A
 * \param lda Leading dimension of the matrix A
 * \param B Pointer to the memory of vector B
 * \param ldb Stride of the vector B
 * \param beta Scalar used for multiplication of C
 * \param C Pointer to the memory of vector C
 * \param ldc Leading dimension of the vector C
 */
inline void cublas_gemv(cublasHandle_t handle,
                        cublasOperation_t trans,
                        size_t m,
                        size_t n,
                        cuComplex* alpha,
                        const std::complex<float>* A,
                        size_t lda,
                        const std::complex<float>* B,
                        size_t ldb,
                        cuComplex* beta,
                        std::complex<float>* C,
                        size_t ldc) {
    cublas_check(cublasCgemv(handle, trans, m, n, alpha, reinterpret_cast<const cuComplex*>(A), lda, reinterpret_cast<const cuComplex*>(B), ldb, beta,
                             reinterpret_cast<cuComplex*>(C), ldc));
}

/*!
 * \brief Perform a GEMV operation with CUBLAS, overloaded version for complex double-precision
 * \param handle The handle to the CUBLAS library context
 * \param transa The operation op(A)
 * \param m The number of rows of matrix A
 * \param n The number of columns of matrix A
 * \param alpha Scalar used for multiplication of op(A)
 * \param A Pointer to the memory of matrix A
 * \param lda Leading dimension of the matrix A
 * \param B Pointer to the memory of vector B
 * \param ldb Stride of the vector B
 * \param beta Scalar used for multiplication of C
 * \param C Pointer to the memory of vector C
 * \param ldc Leading dimension of the vector C
 */
inline void cublas_gemv(cublasHandle_t handle,
                        cublasOperation_t trans,
                        size_t m,
                        size_t n,
                        cuDoubleComplex* alpha,
                        const std::complex<double>* A,
                        size_t lda,
                        const std::complex<double>* B,
                        size_t ldb,
                        cuDoubleComplex* beta,
                        std::complex<double>* C,
                        size_t ldc) {
    cublas_check(cublasZgemv(handle, trans, m, n, alpha, reinterpret_cast<const cuDoubleComplex*>(A), lda, reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                             beta, reinterpret_cast<cuDoubleComplex*>(C), ldc));
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, typename T>
void gemm([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c, [[maybe_unused]] T alpha) {
    if constexpr ((all_row_major<A, B, C> || all_column_major<A, B, C>) &&all_homogeneous<A, B, C>) {
        decltype(auto) handle = start_cublas();

        constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

        using VT = value_t<A>;
        using CT  = cublas_type<VT>;

        auto calpha = cublas_convert<CT>(alpha);
        auto beta   = make_default<CT>(0.0);

        a.ensure_gpu_up_to_date();
        b.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        // Do the actual multiplication

        if (row_major) {
            cublas_gemm(handle.get(),
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        etl::columns(c),
                        etl::rows(c),
                        etl::columns(a),
                        &calpha,
                        b.gpu_memory(),
                        etl::major_stride(b),
                        a.gpu_memory(),
                        etl::major_stride(a),
                        &beta,
                        c.gpu_memory(),
                        etl::major_stride(c));
        } else {
            cublas_gemm(handle.get(),
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        etl::rows(c),
                        etl::columns(c),
                        etl::columns(a),
                        &calpha,
                        a.gpu_memory(),
                        etl::major_stride(a),
                        b.gpu_memory(),
                        etl::major_stride(b),
                        &beta,
                        c.gpu_memory(),
                        etl::major_stride(c));
        }

        c.validate_gpu();
        c.invalidate_cpu();
    } else if constexpr (all_row_major<B, C> && is_column_major<A> && all_homogeneous<A, B, C>) {
        gemm(force_temporary_opp(a), b, c, alpha);
    } else if constexpr (all_row_major<A, C> && is_column_major<B> && all_homogeneous<A, B, C>) {
        gemm(a, force_temporary_opp(b), c, alpha);
    } else if constexpr (is_row_major<C> && all_column_major<A, B> && all_homogeneous<A, B, C>) {
        gemm(force_temporary_opp(a), force_temporary_opp(b), c, alpha);
    } else if constexpr (is_row_major<A> && all_column_major<B, C> && all_homogeneous<A, B, C>) {
        gemm(force_temporary_opp(a), b, c, alpha);
    } else if constexpr (is_row_major<B> && all_column_major<A, C> && all_homogeneous<A, B, C>) {
        gemm(a, force_temporary_opp(b), c, alpha);
    } else if constexpr (all_row_major<A, B> && is_column_major<C> && all_homogeneous<A, B, C>) {
        gemm(force_temporary_opp(a), force_temporary_opp(b), c, alpha);
    } else {
        cpp_unreachable("Unhandled condition in cublas:gemm");
    }
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, typename T>
void gemm_nt([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c, [[maybe_unused]] T alpha) {
    if constexpr (all_homogeneous<A, B, C>) {
        decltype(auto) handle = start_cublas();

        constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

        static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

        using VT = value_t<A>;
        using CT  = cublas_type<VT>;

        auto calpha = cublas_convert<CT>(alpha);
        auto beta   = make_default<CT>(0.0);

        a.ensure_gpu_up_to_date();
        b.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        // Do the actual multiplication

        if (row_major) {
            cublas_gemm(handle.get(),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        etl::columns(c),
                        etl::rows(c),
                        etl::columns(a),
                        &calpha,
                        b.gpu_memory(),
                        etl::major_stride(b),
                        a.gpu_memory(),
                        etl::major_stride(a),
                        &beta,
                        c.gpu_memory(),
                        etl::major_stride(c));
        } else {
            cublas_gemm(handle.get(),
                        CUBLAS_OP_N,
                        CUBLAS_OP_T,
                        etl::rows(c),
                        etl::columns(c),
                        etl::columns(a),
                        &calpha,
                        a.gpu_memory(),
                        etl::major_stride(a),
                        b.gpu_memory(),
                        etl::major_stride(b),
                        &beta,
                        c.gpu_memory(),
                        etl::major_stride(c));
        }

        c.validate_gpu();
        c.invalidate_cpu();
    } else {
        cpp_unreachable("Invalid operation called cublas::gemm_nt with heterogeneous types");
    }
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, typename T>
void gemm_tn([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c, [[maybe_unused]] T alpha) {
    if constexpr (all_homogeneous<A, B, C>) {
        decltype(auto) handle = start_cublas();

        constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

        static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

        using VT = value_t<A>;
        using CT  = cublas_type<VT>;

        auto calpha = cublas_convert<CT>(alpha);
        auto beta   = make_default<CT>(0.0);

        a.ensure_gpu_up_to_date();
        b.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        // Do the actual multiplication

        if (row_major) {
            cublas_gemm(handle.get(),
                        CUBLAS_OP_N,
                        CUBLAS_OP_T,
                        etl::columns(c),
                        etl::rows(c),
                        etl::rows(a),
                        &calpha,
                        b.gpu_memory(),
                        etl::major_stride(b),
                        a.gpu_memory(),
                        etl::major_stride(a),
                        &beta,
                        c.gpu_memory(),
                        etl::major_stride(c));
        } else {
            cublas_gemm(handle.get(),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        etl::rows(c),
                        etl::columns(c),
                        etl::rows(a),
                        &calpha,
                        a.gpu_memory(),
                        etl::major_stride(a),
                        b.gpu_memory(),
                        etl::major_stride(b),
                        &beta,
                        c.gpu_memory(),
                        etl::major_stride(c));
        }

        c.validate_gpu();
        c.invalidate_cpu();
    } else {
        cpp_unreachable("Invalid operation called cublas::gemm_tn with heterogeneous types");
    }
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, typename T>
void gemm_tt([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c, [[maybe_unused]] T alpha) {
    if constexpr (all_homogeneous<A, B, C>) {
        decltype(auto) handle = start_cublas();

        constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

        static_assert(decay_traits<A>::storage_order == decay_traits<B>::storage_order, "gemm only for same A/B storage order");

        using VT = value_t<A>;
        using CT  = cublas_type<VT>;

        auto calpha = cublas_convert<CT>(alpha);
        auto beta = make_default<CT>(0.0);

        a.ensure_gpu_up_to_date();
        b.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        // Do the actual multiplication

        if (row_major) {
            cublas_gemm(handle.get(),
                        CUBLAS_OP_T,
                        CUBLAS_OP_T,
                        etl::columns(c),
                        etl::rows(c),
                        etl::rows(a),
                        &calpha,
                        b.gpu_memory(),
                        etl::major_stride(b),
                        a.gpu_memory(),
                        etl::major_stride(a),
                        &beta,
                        c.gpu_memory(),
                        etl::major_stride(c));
        } else {
            cublas_gemm(handle.get(),
                        CUBLAS_OP_T,
                        CUBLAS_OP_T,
                        etl::rows(c),
                        etl::columns(c),
                        etl::rows(a),
                        &alpha,
                        a.gpu_memory(),
                        etl::major_stride(a),
                        b.gpu_memory(),
                        etl::major_stride(b),
                        &beta,
                        c.gpu_memory(),
                        etl::major_stride(c));
        }

        c.validate_gpu();
        c.invalidate_cpu();
    } else {
        cpp_unreachable("Invalid operation called cublas::gemm_tt with heterogeneous types");
    }
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemv([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    if constexpr (all_homogeneous<A, B, C>) {
        decltype(auto) handle = start_cublas();

        constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

        a.ensure_gpu_up_to_date();
        b.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        using VT = value_t<A>;
        using CT  = cublas_type<VT>;

        auto alpha = make_default<CT>(1.0);
        auto beta  = make_default<CT>(0.0);

        // Perform the actual multiplication

        if (row_major) {
            cublas_gemv(handle.get(),
                        CUBLAS_OP_T,
                        etl::columns(a),
                        etl::rows(a),
                        &alpha,
                        safe_cast(a.gpu_memory()),
                        major_stride(a),
                        safe_cast(b.gpu_memory()),
                        1,
                        &beta,
                        safe_cast(c.gpu_memory()),
                        1);
        } else {
            cublas_gemv(handle.get(),
                        CUBLAS_OP_N,
                        etl::rows(a),
                        etl::columns(a),
                        &alpha,
                        safe_cast(a.gpu_memory()),
                        major_stride(a),
                        safe_cast(b.gpu_memory()),
                        1,
                        &beta,
                        safe_cast(c.gpu_memory()),
                        1);
        }

        // Copy the result from GPU to CPU

        c.validate_gpu();
        c.invalidate_cpu();
    } else {
        cpp_unreachable("Invalid operation called blas::gemv with heterogeneous types");
    }
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemv_t([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    if constexpr (all_homogeneous<A, B, C>) {
        decltype(auto) handle = start_cublas();

        constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

        a.ensure_gpu_up_to_date();
        b.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        using VT = value_t<A>;
        using CT  = cublas_type<VT>;

        auto alpha = make_default<CT>(1.0);
        auto beta = make_default<CT>(0.0);

        // Perform the actual multiplication

        if (row_major) {
            cublas_gemv(handle.get(),
                        CUBLAS_OP_N,
                        etl::columns(a),
                        etl::rows(a),
                        &alpha,
                        safe_cast(a.gpu_memory()),
                        major_stride(a),
                        safe_cast(b.gpu_memory()),
                        1,
                        &beta,
                        safe_cast(c.gpu_memory()),
                        1);
        } else {
            cublas_gemv(handle.get(),
                        CUBLAS_OP_T,
                        etl::rows(a),
                        etl::columns(a),
                        &alpha,
                        safe_cast(a.gpu_memory()),
                        major_stride(a),
                        safe_cast(b.gpu_memory()),
                        1,
                        &beta,
                        safe_cast(c.gpu_memory()),
                        1);
        }

        // Copy the result from GPU to CPU

        c.validate_gpu();
        c.invalidate_cpu();
    } else {
        cpp_unreachable("Invalid operation called blas::gemv_t with heterogeneous types");
    }
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gevm([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    if constexpr (all_homogeneous<A, B, C>) {
        decltype(auto) handle = start_cublas();

        constexpr bool row_major = decay_traits<B>::storage_order == order::RowMajor;

        a.ensure_gpu_up_to_date();
        b.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        using VT = value_t<A>;
        using CT  = cublas_type<VT>;

        auto alpha = make_default<CT>(1.0);
        auto beta = make_default<CT>(0.0);

        // Perform the actual multiplication

        if (row_major) {
            cublas_gemv(handle.get(),
                        CUBLAS_OP_N,
                        etl::columns(b),
                        etl::rows(b),
                        &alpha,
                        safe_cast(b.gpu_memory()),
                        major_stride(b),
                        safe_cast(a.gpu_memory()),
                        1,
                        &beta,
                        safe_cast(c.gpu_memory()),
                        1);
        } else {
            cublas_gemv(handle.get(),
                        CUBLAS_OP_T,
                        etl::rows(b),
                        etl::columns(b),
                        &alpha,
                        safe_cast(b.gpu_memory()),
                        major_stride(b),
                        safe_cast(a.gpu_memory()),
                        1,
                        &beta,
                        safe_cast(c.gpu_memory()),
                        1);
        }

        // Copy the result from GPU to CPU

        c.validate_gpu();
        c.invalidate_cpu();
    } else {
        cpp_unreachable("Invalid operation called blas::gevm with heterogeneous types");
    }
}

/*!
 * \brief Compute the vector-matrix mutplication of a and trans(B) and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gevm_t([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    if constexpr (all_homogeneous<A, B, C>) {
        decltype(auto) handle = start_cublas();

        constexpr bool row_major = decay_traits<B>::storage_order == order::RowMajor;

        a.ensure_gpu_up_to_date();
        b.ensure_gpu_up_to_date();
        c.ensure_gpu_allocated();

        using VT = value_t<A>;
        using CT  = cublas_type<VT>;

        auto alpha = make_default<CT>(1.0);
        auto beta = make_default<CT>(0.0);

        // Perform the actual multiplication

        if (row_major) {
            cublas_gemv(handle.get(),
                        CUBLAS_OP_T,
                        etl::columns(b),
                        etl::rows(b),
                        &alpha,
                        safe_cast(b.gpu_memory()),
                        major_stride(b),
                        safe_cast(a.gpu_memory()),
                        1,
                        &beta,
                        safe_cast(c.gpu_memory()),
                        1);
        } else {
            cublas_gemv(handle.get(),
                        CUBLAS_OP_N,
                        etl::rows(b),
                        etl::columns(b),
                        &alpha,
                        safe_cast(b.gpu_memory()),
                        major_stride(b),
                        safe_cast(a.gpu_memory()),
                        1,
                        &beta,
                        safe_cast(c.gpu_memory()),
                        1);
        }

        // Copy the result from GPU to CPU

        c.validate_gpu();
        c.invalidate_cpu();
    } else {
        cpp_unreachable("Invalid operation called blas::gevm_t with heterogeneous types");
    }
}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, typename T>
void gemm([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c, [[maybe_unused]] T alpha) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, typename T>
void gemm_nt([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c, [[maybe_unused]] T alpha) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, typename T>
void gemm_tn([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c, [[maybe_unused]] T alpha) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, typename T>
void gemm_tt([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c, [[maybe_unused]] T alpha) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemv([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemv_t([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gevm([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

/*!
 * \brief Compute the vector-matrix mutplication of a and trans(B) and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gevm_t([[maybe_unused]] A&& a, [[maybe_unused]] B&& b, [[maybe_unused]] C&& c) {
    cpp_unreachable("Unsupported feature called: cublas gemm");
}

    //COVERAGE_EXCLUDE_END

#endif

} //end of namespace etl::impl::cublas
