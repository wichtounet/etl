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

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A, B, C>::value)>
void gemm(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    cblas_sgemm(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans, CblasNoTrans,
        etl::rows(a), etl::columns(b), etl::columns(a),
        1.0f,
        a.memory_start(), major_stride(a),
        b.memory_start(), major_stride(b),
        0.0f,
        c.memory_start(), major_stride(c));
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A, B, C>::value)>
void gemm(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    cblas_dgemm(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans, CblasNoTrans,
        etl::rows(a), etl::columns(b), etl::columns(a),
        1.0,
        a.memory_start(), major_stride(a),
        b.memory_start(), major_stride(b),
        0.0,
        c.memory_start(), major_stride(c));
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_single_precision<A, B, C>::value)>
void gemm(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    std::complex<float> alpha(1.0, 0.0);
    std::complex<float> beta(0.0, 0.0);

    cblas_cgemm(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans, CblasNoTrans,
        etl::rows(a), etl::columns(b), etl::columns(a),
        &alpha,
        a.memory_start(), major_stride(a),
        b.memory_start(), major_stride(b),
        &beta,
        c.memory_start(), major_stride(c));
}

/*!
 * \brief Compute the matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_double_precision<A, B, C>::value)>
void gemm(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

    cblas_zgemm(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans, CblasNoTrans,
        etl::rows(a), etl::columns(b), etl::columns(a),
        &alpha,
        a.memory_start(), major_stride(a),
        b.memory_start(), major_stride(b),
        &beta,
        c.memory_start(), major_stride(c));
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    cblas_dgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans,
        etl::rows(a), etl::columns(a),
        1.0,
        a.memory_start(), major_stride(a),
        b.memory_start(), 1,
        0.0,
        c.memory_start(), 1);
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    cblas_sgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans,
        etl::rows(a), etl::columns(a),
        1.0,
        a.memory_start(), major_stride(a),
        b.memory_start(), 1,
        0.0,
        c.memory_start(), 1);
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_single_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    std::complex<float> alpha(1.0, 0.0);
    std::complex<float> beta(0.0, 0.0);

    cblas_cgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans,
        etl::rows(a), etl::columns(a),
        &alpha,
        a.memory_start(), major_stride(a),
        b.memory_start(), 1,
        &beta,
        c.memory_start(), 1);
}

/*!
 * \brief Compute the matrix-vector mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_double_precision<A, B, C>::value)>
void gemv(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

    cblas_zgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasNoTrans,
        etl::rows(a), etl::columns(a),
        &alpha,
        a.memory_start(), major_stride(a),
        b.memory_start(), 1,
        &beta,
        c.memory_start(), 1);
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_double_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    cblas_dgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasTrans,
        etl::rows(b), etl::columns(b),
        1.0,
        b.memory_start(), major_stride(b),
        a.memory_start(), 1,
        0.0,
        c.memory_start(), 1);
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_single_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    cblas_sgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasTrans,
        etl::rows(b), etl::columns(b),
        1.0,
        b.memory_start(), major_stride(b),
        a.memory_start(), 1,
        0.0,
        c.memory_start(), 1);
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_single_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    std::complex<float> alpha(1.0, 0.0);
    std::complex<float> beta(0.0, 0.0);

    cblas_cgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasTrans,
        etl::rows(b), etl::columns(b),
        &alpha,
        b.memory_start(), major_stride(b),
        a.memory_start(), 1,
        &beta,
        c.memory_start(), 1);
}

/*!
 * \brief Compute the vector-matrix mutplication of a and b and store the result in c
 * param a The lhs of the multiplication
 * param b The rhs of the multiplication
 * param c The result
 */
template <typename A, typename B, typename C, cpp_enable_if(all_complex_double_precision<A, B, C>::value)>
void gevm(A&& a, B&& b, C&& c) {
    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    std::complex<double> alpha(1.0, 0.0);
    std::complex<double> beta(0.0, 0.0);

    cblas_zgemv(
        row_major ? CblasRowMajor : CblasColMajor,
        CblasTrans,
        etl::rows(b), etl::columns(b),
        &alpha,
        b.memory_start(), major_stride(b),
        a.memory_start(), 1,
        &beta,
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
