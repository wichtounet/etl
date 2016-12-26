//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#ifdef ETL_BLAS_MODE

extern "C" {
#include "cblas.h"
}

#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_BLAS_MODE

// GEMM overloads

inline void cblas_gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K,
        const float alpha, const float* A, size_t lda, const float* B, size_t ldb, const float beta, float* C, size_t ldc) {
    cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K,
        const double alpha, const double* A, size_t lda, const double* B, size_t ldb, const double beta, double* C, size_t ldc) {
    cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

inline void cblas_gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K,
        const etl::complex<float> alpha, const etl::complex<float>* A, size_t lda, const etl::complex<float>* B, size_t ldb, const etl::complex<float> beta, etl::complex<float>* C, size_t ldc) {
    cblas_cgemm(Layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K,
        const etl::complex<double> alpha, const etl::complex<double>* A, size_t lda, const etl::complex<double>* B, size_t ldb, const etl::complex<double> beta, etl::complex<double>* C, size_t ldc) {
    cblas_zgemm(Layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K,
        const std::complex<float> alpha, const std::complex<float>* A, size_t lda, const std::complex<float>* B, size_t ldb, const std::complex<float> beta, std::complex<float>* C, size_t ldc) {
    cblas_cgemm(Layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

inline void cblas_gemm(CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, size_t M, size_t N, size_t K,
        const std::complex<double> alpha, const std::complex<double>* A, size_t lda, const std::complex<double>* B, size_t ldb, const std::complex<double> beta, std::complex<double>* C, size_t ldc) {
    cblas_zgemm(Layout, TransA, TransB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

// GEMV overloads

inline void cblas_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, size_t M, size_t N, const float alpha, const float *A, size_t lda,
        const float *X, size_t incX, const float beta, float *Y, size_t incY){
    cblas_sgemv(Layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void cblas_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, size_t M, size_t N, const double alpha, const double *A, size_t lda,
        const double *X, size_t incX, const double beta, double *Y, size_t incY){
    cblas_dgemv(Layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

inline void cblas_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, size_t M, size_t N, const std::complex<float> alpha, const std::complex<float> *A, size_t lda,
        const std::complex<float> *X, size_t incX, const std::complex<float> beta, std::complex<float> *Y, size_t incY){
    cblas_cgemv(Layout, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void cblas_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, size_t M, size_t N, const std::complex<double> alpha, const std::complex<double> *A, size_t lda,
        const std::complex<double> *X, size_t incX, const std::complex<double> beta, std::complex<double> *Y, size_t incY){
    cblas_zgemv(Layout, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void cblas_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, size_t M, size_t N, const etl::complex<float> alpha, const etl::complex<float> *A, size_t lda,
        const etl::complex<float> *X, size_t incX, const etl::complex<float> beta, etl::complex<float> *Y, size_t incY){
    cblas_cgemv(Layout, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
}

inline void cblas_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, size_t M, size_t N, const etl::complex<double> alpha, const etl::complex<double> *A, size_t lda,
        const etl::complex<double> *X, size_t incX, const etl::complex<double> beta, etl::complex<double> *Y, size_t incY){
    cblas_zgemv(Layout, TransA, M, N, &alpha, A, lda, X, incX, &beta, Y, incY);
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemm(A&& a, B&& b, C&& c) {
    using T = value_t<A>;

    static constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    T alpha(1.0);
    T beta(0.0);

    cblas_gemm(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans, CblasNoTrans,
        etl::rows(a), etl::columns(b), etl::columns(a),
        alpha,
        a.memory_start(), major_stride(a),
        b.memory_start(), major_stride(b),
        beta,
        c.memory_start(), major_stride(c));
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gemv(A&& a, B&& b, C&& c) {
    using T = value_t<A>;

    static constexpr bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    T alpha(1.0);
    T beta(0.0);

    cblas_gemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans,
        etl::rows(a), etl::columns(a),
        alpha,
        a.memory_start(), major_stride(a),
        b.memory_start(), 1,
        beta,
        c.memory_start(), 1);
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C>
void gevm(A&& a, B&& b, C&& c) {
    using T = value_t<A>;

    static constexpr bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    T alpha(1.0);
    T beta(0.0);

    cblas_gemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasTrans,
        etl::rows(b), etl::columns(b),
        alpha,
        b.memory_start(), major_stride(b),
        a.memory_start(), 1,
        beta,
        c.memory_start(), 1);
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
    cpp_unreachable("Unsupported feature called: blas gemm");
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
    cpp_unreachable("Unsupported feature called: blas gemm");
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
    cpp_unreachable("Unsupported feature called: blas gemm");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace blas

} //end of namespace impl

} //end of namespace etl
