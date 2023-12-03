//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief BLAS implementation of the outer product
 */

#pragma once

#ifdef ETL_CUBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"
#include "etl/impl/cublas/cublas.hpp"

#endif

namespace etl::impl::cublas {

#ifdef ETL_CUBLAS_MODE

/*!
 * \brief Compute the batch_outer product of a and b and store the result in c
 * \param a The lhs expression
 * \param b The rhs expression
 * \param c The output expression
 */
template <etl_single_precision A, etl_single_precision B, etl_single_precision C>
void batch_outer(const A& a, const B& b, C&& c) {
    decltype(auto) handle = start_cublas();

    float alpha = 1.0;
    float beta  = 0.0;

    // This is brain-killing :s
    // CUBLAS need matrices in column-major order. By switching both
    // matrices, this is achieved. However, since one of the matrix
    // needs to be transposed, it must be changed again

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

    cublas_check(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_T, etl::columns(c), etl::rows(c), etl::rows(b), &alpha, b.gpu_memory(), etl::columns(b),
                             a.gpu_memory(), etl::columns(a), &beta, c.gpu_memory(), etl::columns(b)));

    c.validate_gpu();
    c.invalidate_cpu();
}

/*!
 * \copydoc batch_outer
 */
template <etl_double_precision A, etl_double_precision B, etl_double_precision C>
void batch_outer(const A& a, const B& b, C&& c) {
    decltype(auto) handle = start_cublas();

    double alpha = 1.0;
    double beta  = 0.0;

    a.ensure_gpu_up_to_date();
    b.ensure_gpu_up_to_date();
    c.ensure_gpu_allocated();

    cublas_check(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_T, etl::columns(c), etl::rows(c), etl::rows(b), &alpha, b.gpu_memory(), etl::columns(b),
                             a.gpu_memory(), etl::columns(a), &beta, c.gpu_memory(), etl::columns(b)));

    c.validate_gpu();
    c.invalidate_cpu();
}

#else

/*!
 * \copydoc batch_outer
 */
template <typename A, typename B, typename C>
void batch_outer(const A& /*a*/, const B& /*b*/, C&& /*c*/) {
    cpp_unreachable("CUBLAS not enabled/available");
}

#endif

} //end of namespace etl::impl::cublas
