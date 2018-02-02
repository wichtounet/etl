//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the or operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

/*!
 * \brief Indicates if EGBLAS has single-precision or.
 */
#ifdef EGBLAS_HAS_BOR
static constexpr bool has_bor = true;
#else
static constexpr bool has_bor = false;
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
inline void logical_or(size_t n, const bool* A, size_t lda, const bool* B, size_t ldb, bool* C, size_t ldc) {
#ifdef EGBLAS_HAS_BOR
    inc_counter("egblas");
    egblas_bor(n, A, lda, B, ldb, C, ldc);
#else
    cpp_unused(n);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(B);
    cpp_unused(ldb);
    cpp_unused(C);
    cpp_unused(ldc);

    cpp_unreachable("Invalid call to egblas::or");
#endif
}

} //end of namespace etl::impl::egblas
