//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

namespace etl {

namespace impl {

namespace egblas {

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
inline void relu_der_out(size_t n, float alpha, float* A , size_t lda, float* B , size_t ldb){
#ifdef EGBLAS_HAS_SRELU_DER_OUT
    egblas_srelu_der_out(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

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
inline void relu_der_out(size_t n, double alpha, double* A , size_t lda, double* B , size_t ldb){
#ifdef EGBLAS_HAS_DRELU_DER_OUT
    egblas_drelu_der_out(n, alpha, A, lda, B, ldb);
#else
    cpp_unused(n);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);

    cpp_unreachable("Invalid call to egblas::relu_der_out");
#endif
}

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
