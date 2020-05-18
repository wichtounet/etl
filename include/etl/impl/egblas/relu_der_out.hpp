//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the relu_der_out operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision relu_der_out.
 */
#ifdef EGBLAS_HAS_SRELU_DER_OUT
static constexpr bool has_srelu_der_out = true;
#else
static constexpr bool has_srelu_der_out = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas relu_der_out operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void relu_der_out([[maybe_unused]] size_t n,
                         [[maybe_unused]] float alpha,
                         [[maybe_unused]] float* A,
                         [[maybe_unused]] size_t lda,
                         [[maybe_unused]] float* B,
                         [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SRELU_DER_OUT
    inc_counter("egblas");
    egblas_srelu_der_out(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::relu_der_out");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision relu_der_out.
 */
#ifdef EGBLAS_HAS_DRELU_DER_OUT
static constexpr bool has_drelu_der_out = true;
#else
static constexpr bool has_drelu_der_out = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas relu_der_out operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void relu_der_out([[maybe_unused]] size_t n,
                         [[maybe_unused]] double alpha,
                         [[maybe_unused]] double* A,
                         [[maybe_unused]] size_t lda,
                         [[maybe_unused]] double* B,
                         [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_DRELU_DER_OUT
    inc_counter("egblas");
    egblas_drelu_der_out(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::relu_der_out");
#endif
}

} //end of namespace etl::impl::egblas
