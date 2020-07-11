//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the transpose_front operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_STRANSPOSE_FRONT
static constexpr bool has_stranspose_front = true;
#else
static constexpr bool has_stranspose_front = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas transpose_front operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void transpose_front([[maybe_unused]] size_t m,
                            [[maybe_unused]] size_t n,
                            [[maybe_unused]] size_t k,
                            [[maybe_unused]] float* A,
                            [[maybe_unused]] float* B) {
#ifdef EGBLAS_HAS_STRANSPOSE_FRONT
    inc_counter("egblas");
    egblas_stranspose_front(m, n, k, A, B);
#else
    cpp_unreachable("Invalid call to egblas::transpose_front");
#endif
}

#ifdef EGBLAS_HAS_DTRANSPOSE_FRONT
static constexpr bool has_dtranspose_front = true;
#else
static constexpr bool has_dtranspose_front = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas transpose_front operation
 * \param b The batch dimension of the matrix
 * \param n The size of the output vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void transpose_front([[maybe_unused]] size_t m,
                            [[maybe_unused]] size_t n,
                            [[maybe_unused]] size_t k,
                            [[maybe_unused]] double* A,
                            [[maybe_unused]] double* B) {
#ifdef EGBLAS_HAS_DTRANSPOSE_FRONT
    inc_counter("egblas");
    egblas_dtranspose_front(m, n, k, A, B);
#else
    cpp_unreachable("Invalid call to egblas::transpose_front");
#endif
}

} //end of namespace etl::impl::egblas
