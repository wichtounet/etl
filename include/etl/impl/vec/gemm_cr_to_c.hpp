//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Kernels for colum-major matrix - row-major matrix multiplication and
 * assignment to a column-major matrix
 */

#pragma once

namespace etl {

namespace impl {

namespace vec {

/*!
 * \brief Optimized version of GEMM for assignment of a small
 * Column-Major Matrix - Row Major Matrix to a Column Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_small_kernel_cr_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unused(M);
    cpp_unused(N);
    cpp_unused(K);
}

/*!
 * \brief Optimized version of GEMM for assignment of a large
 * Column-Major Matrix - Row Major Matrix to a Column Major Matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 */
template <typename V, typename T>
void gemm_large_kernel_cr_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unused(c);
    cpp_unused(M);
    cpp_unused(N);
    cpp_unused(K);
}

/*!
 * \brief Vectorized implementation of column-major matrix - row-major matrix
 * multiplication and assignment into a column-major matrix.
 *
 * \param a The lhs matrix
 * \param b The rhs matrix
 * \param c The result matrix
 *
 * \param M The number of rows of the matrix A and rows of the matrix C
 * \param N The number of columns of the matrix B and columns of the matrix C
 * \param K The number of columns of the matrix A and rows of the matrix B
 */
template <typename T>
void gemm_cr_to_c(const T* a, const T* b, T* c, size_t M, size_t N, size_t K) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    if (M * N <= gemm_rr_small_threshold) {
        gemm_small_kernel_cr_to_c<default_vec>(a, b, c, M, N, K);
    } else {
        direct_fill_n(c, M * N, T(0));
        gemm_large_kernel_cr_to_c<default_vec>(a, b, c, M, N, K);
    }
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
