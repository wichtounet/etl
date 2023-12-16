//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the xor operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision xor.
 */
#ifdef EGBLAS_HAS_BXOR
static constexpr bool has_bxor = true;
#else
static constexpr bool has_bxor = false;
#endif

/*!
 * \brief Wrappers for or operation
 * \param n The size of the vector
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void logical_xor([[maybe_unused]] size_t n,
                        [[maybe_unused]] const bool* A,
                        [[maybe_unused]] size_t lda,
                        [[maybe_unused]] const bool* B,
                        [[maybe_unused]] size_t ldb,
                        [[maybe_unused]] bool* C,
                        [[maybe_unused]] size_t ldc) {
#ifdef EGBLAS_HAS_BXOR
    inc_counter("egblas");
    egblas_bxor(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::logical_xor");
#endif
}

} //end of namespace etl::impl::egblas
