//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the sigmoid operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

#ifdef EGBLAS_HAS_SSIGMOID
static constexpr bool has_ssigmoid = true;
#else
static constexpr bool has_ssigmoid = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas sigmoid operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sigmoid([[maybe_unused]] size_t n,
                    [[maybe_unused]] float  alpha,
                    [[maybe_unused]] float* A,
                    [[maybe_unused]] size_t lda,
                    [[maybe_unused]] float* B,
                    [[maybe_unused]] size_t ldb) {
#ifdef EGBLAS_HAS_SSIGMOID
    inc_counter("egblas");
    egblas_ssigmoid(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sigmoid");
#endif
}

#ifdef EGBLAS_HAS_DSIGMOID
static constexpr bool has_dsigmoid = true;
#else
static constexpr bool has_dsigmoid = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas sigmoid operation
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param B The memory of the vector b
 * \param ldb The leading dimension of b
 */
inline void sigmoid([[maybe_unused]] size_t  n,
                    [[maybe_unused]] double  alpha,
                    [[maybe_unused]] double* A,
                    [[maybe_unused]] size_t  lda,
                    [[maybe_unused]] double* B,
                    [[maybe_unused]] size_t  ldb) {
#ifdef EGBLAS_HAS_DSIGMOID
    inc_counter("egblas");
    egblas_dsigmoid(n, alpha, A, lda, B, ldb);
#else
    cpp_unreachable("Invalid call to egblas::sigmoid");
#endif
}

} //end of namespace etl::impl::egblas
