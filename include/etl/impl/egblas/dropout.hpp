//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the dropout operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl {

namespace impl {

namespace egblas {

// dropout

/*!
 * \brief Indicates if EGBLAS has single-precision dropout.
 */
#ifdef EGBLAS_HAS_SDROPOUT
static constexpr bool has_sdropout = true;
#else
static constexpr bool has_sdropout = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void dropout(size_t n, float p, float* alpha, float* A, size_t lda) {
#ifdef EGBLAS_HAS_SDROPOUT
    egblas_sdropout(n, p, *alpha, A, lda);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);

    cpp_unreachable("Invalid call to egblas::dropout");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision dropout.
 */
#ifdef EGBLAS_HAS_DDROPOUT
static constexpr bool has_ddropout = true;
#else
static constexpr bool has_ddropout = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void dropout(size_t n, double p, double* alpha, double* A, size_t lda) {
#ifdef EGBLAS_HAS_DDROPOUT
    egblas_ddropout(n, p, *alpha, A, lda);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);

    cpp_unreachable("Invalid call to egblas::dropout");
#endif
}

// dropout_seed

/*!
 * \brief Indicates if EGBLAS has single-precision dropout.
 */
#ifdef EGBLAS_HAS_SDROPOUT_SEED
static constexpr bool has_sdropout_seed = true;
#else
static constexpr bool has_sdropout_seed = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void dropout_seed(size_t n, float p, float* alpha, float* A, size_t lda, size_t seed) {
#ifdef EGBLAS_HAS_SDROPOUT_SEED
    egblas_sdropout_seed(n, p, *alpha, A, lda, seed);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(seed);

    cpp_unreachable("Invalid call to egblas::dropout");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision dropout.
 */
#ifdef EGBLAS_HAS_DDROPOUT_SEED
static constexpr bool has_ddropout_seed = true;
#else
static constexpr bool has_ddropout_seed = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void dropout_seed(size_t n, double p, double* alpha, double* A, size_t lda, size_t seed) {
#ifdef EGBLAS_HAS_DDROPOUT_SEED
    egblas_ddropout_seed(n, p, *alpha, A, lda, seed);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(seed);

    cpp_unreachable("Invalid call to egblas::dropout");
#endif
}

// inverted dropout

/*!
 * \brief Indicates if EGBLAS has single-precision dropout.
 */
#ifdef EGBLAS_HAS_SINV_DROPOUT
static constexpr bool has_sinv_dropout = true;
#else
static constexpr bool has_sinv_dropout = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void inv_dropout(size_t n, float p, float* alpha, float* A, size_t lda) {
#ifdef EGBLAS_HAS_SINV_DROPOUT
    egblas_sinv_dropout(n, p, *alpha, A, lda);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);

    cpp_unreachable("Invalid call to egblas::inv_dropout");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision dropout.
 */
#ifdef EGBLAS_HAS_DINV_DROPOUT
static constexpr bool has_dinv_dropout = true;
#else
static constexpr bool has_dinv_dropout = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void inv_dropout(size_t n, double p, double* alpha, double* A, size_t lda) {
#ifdef EGBLAS_HAS_DINV_DROPOUT
    egblas_dinv_dropout(n, p, *alpha, A, lda);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);

    cpp_unreachable("Invalid call to egblas::inv_dropout");
#endif
}

// dropout_seed

/*!
 * \brief Indicates if EGBLAS has single-precision dropout.
 */
#ifdef EGBLAS_HAS_SINV_DROPOUT_SEED
static constexpr bool has_sinv_dropout_seed = true;
#else
static constexpr bool has_sinv_dropout_seed = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void inv_dropout_seed(size_t n, float p, float* alpha, float* A, size_t lda, size_t seed) {
#ifdef EGBLAS_HAS_SINV_DROPOUT_SEED
    egblas_sinv_dropout_seed(n, p, *alpha, A, lda, seed);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(seed);

    cpp_unreachable("Invalid call to egblas::inv_dropout");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision dropout.
 */
#ifdef EGBLAS_HAS_DINV_DROPOUT_SEED
static constexpr bool has_dinv_dropout_seed = true;
#else
static constexpr bool has_dinv_dropout_seed = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void inv_dropout_seed(size_t n, double p, double* alpha, double* A, size_t lda, size_t seed) {
#ifdef EGBLAS_HAS_DINV_DROPOUT_SEED
    egblas_dinv_dropout_seed(n, p, *alpha, A, lda, seed);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(seed);

    cpp_unreachable("Invalid call to egblas::inv_dropout");
#endif
}

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
