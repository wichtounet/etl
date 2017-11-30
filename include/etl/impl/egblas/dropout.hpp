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

// dropout prepare

/*!
 * \brief Indicates if EGBLAS has dropout_prepare
 */
#ifdef EGBLAS_HAS_DROPOUT_PREPARE
static constexpr bool has_dropout_prepare = true;
#else
static constexpr bool has_dropout_prepare = false;
#endif

/*!
 * \brief Prepare random states for dropout
 * \return random states for dropout
 */
inline void* dropout_prepare() {
#ifdef EGBLAS_HAS_DROPOUT_PREPARE
    inc_counter("egblas");
    return egblas_dropout_prepare();
#else
    cpp_unreachable("Invalid call to egblas::dropout_prepare");
    return nullptr;
#endif
}

/*!
 * \brief Indicates if EGBLAS has dropout_prepare
 */
#ifdef EGBLAS_HAS_DROPOUT_PREPARE_SEED
static constexpr bool has_dropout_prepare_seed = true;
#else
static constexpr bool has_dropout_prepare_seed = false;
#endif

/*!
 * \brief Prepare random states for dropout with the given seed
 * \param seed The seed
 * \return random states for dropout
 */
inline void* dropout_prepare_seed(size_t seed) {
#ifdef EGBLAS_HAS_DROPOUT_PREPARE_SEED
    inc_counter("egblas");
    return egblas_dropout_prepare_seed(seed);
#else
    cpp_unused(seed);

    cpp_unreachable("Invalid call to egblas::dropout_prepare_seed");
    return nullptr;
#endif
}

/*!
 * \brief Indicates if EGBLAS has dropout_prepare
 */
#ifdef EGBLAS_HAS_DROPOUT_RELEASE
static constexpr bool has_dropout_release = true;
#else
static constexpr bool has_dropout_release = false;
#endif

/*!
 * \brief Prepare random states for dropout with the given seed
 * \param seed The seed
 * \return random states for dropout
 */
inline void dropout_release(void* state) {
#ifdef EGBLAS_HAS_DROPOUT_RELEASE
    inc_counter("egblas");
    egblas_dropout_release(state);
#else
    cpp_unused(state);

    cpp_unreachable("Invalid call to egblas::dropout_release");
#endif
}

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
inline void dropout(size_t n, float p, float alpha, float* A, size_t lda) {
#ifdef EGBLAS_HAS_SDROPOUT
    inc_counter("egblas");
    egblas_sdropout(n, p, alpha, A, lda);
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
inline void dropout(size_t n, double p, double alpha, double* A, size_t lda) {
#ifdef EGBLAS_HAS_DDROPOUT
    inc_counter("egblas");
    egblas_ddropout(n, p, alpha, A, lda);
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
inline void dropout_seed(size_t n, float p, float alpha, float* A, size_t lda, size_t seed) {
#ifdef EGBLAS_HAS_SDROPOUT_SEED
    inc_counter("egblas");
    egblas_sdropout_seed(n, p, alpha, A, lda, seed);
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
inline void dropout_seed(size_t n, double p, double alpha, double* A, size_t lda, size_t seed) {
#ifdef EGBLAS_HAS_DDROPOUT_SEED
    inc_counter("egblas");
    egblas_ddropout_seed(n, p, alpha, A, lda, seed);
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

// dropout_states

/*!
 * \brief Indicates if EGBLAS has single-precision dropout.
 */
#ifdef EGBLAS_HAS_SDROPOUT_STATES
static constexpr bool has_sdropout_states = true;
#else
static constexpr bool has_sdropout_states = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void dropout_states(size_t n, float p, float alpha, float* A, size_t lda, void* states) {
#ifdef EGBLAS_HAS_SDROPOUT_STATES
    inc_counter("egblas");
    egblas_sdropout_states(n, p, alpha, A, lda, states);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(states);

    cpp_unreachable("Invalid call to egblas::dropout_states");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision dropout.
 */
#ifdef EGBLAS_HAS_DDROPOUT_STATES
static constexpr bool has_ddropout_states = true;
#else
static constexpr bool has_ddropout_states = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void dropout_states(size_t n, double p, double alpha, double* A, size_t lda, void* states) {
#ifdef EGBLAS_HAS_DDROPOUT_STATES
    inc_counter("egblas");
    egblas_ddropout_states(n, p, alpha, A, lda, states);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(states);

    cpp_unreachable("Invalid call to egblas::dropout_states");
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
inline void inv_dropout(size_t n, float p, float alpha, float* A, size_t lda) {
#ifdef EGBLAS_HAS_SINV_DROPOUT
    inc_counter("egblas");
    egblas_sinv_dropout(n, p, alpha, A, lda);
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
inline void inv_dropout(size_t n, double p, double alpha, double* A, size_t lda) {
#ifdef EGBLAS_HAS_DINV_DROPOUT
    inc_counter("egblas");
    egblas_dinv_dropout(n, p, alpha, A, lda);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);

    cpp_unreachable("Invalid call to egblas::inv_dropout");
#endif
}

// inv_dropout_seed

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
inline void inv_dropout_seed(size_t n, float p, float alpha, float* A, size_t lda, size_t seed) {
#ifdef EGBLAS_HAS_SINV_DROPOUT_SEED
    inc_counter("egblas");
    egblas_sinv_dropout_seed(n, p, alpha, A, lda, seed);
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
inline void inv_dropout_seed(size_t n, double p, double alpha, double* A, size_t lda, size_t seed) {
#ifdef EGBLAS_HAS_DINV_DROPOUT_SEED
    inc_counter("egblas");
    egblas_dinv_dropout_seed(n, p, alpha, A, lda, seed);
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

// inv_dropout_states

/*!
 * \brief Indicates if EGBLAS has single-precision dropout.
 */
#ifdef EGBLAS_HAS_SINV_DROPOUT_STATES
static constexpr bool has_sinv_dropout_states = true;
#else
static constexpr bool has_sinv_dropout_states = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void inv_dropout_states(size_t n, float p, float alpha, float* A, size_t lda, void* states) {
#ifdef EGBLAS_HAS_SINV_DROPOUT_STATES
    inc_counter("egblas");
    egblas_sinv_dropout_states(n, p, alpha, A, lda, states);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(states);

    cpp_unreachable("Invalid call to egblas::inv_dropout_states");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision dropout.
 */
#ifdef EGBLAS_HAS_DINV_DROPOUT_STATES
static constexpr bool has_dinv_dropout_states = true;
#else
static constexpr bool has_dinv_dropout_states = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas dropout
 * \param n The size of the vector
 * \param p The dropout probability
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 */
inline void inv_dropout_states(size_t n, double p, double alpha, double* A, size_t lda, void* states) {
#ifdef EGBLAS_HAS_DINV_DROPOUT_STATES
    inc_counter("egblas");
    egblas_dinv_dropout_states(n, p, alpha, A, lda, states);
#else
    cpp_unused(n);
    cpp_unused(p);
    cpp_unused(alpha);
    cpp_unused(A);
    cpp_unused(lda);
    cpp_unused(states);

    cpp_unreachable("Invalid call to egblas::inv_dropout_states");
#endif
}

} //end of namespace egblas
} //end of namespace impl
} //end of namespace etl
