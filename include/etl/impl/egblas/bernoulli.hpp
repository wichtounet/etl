//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the bernoulli_sample operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

// bernoulli_sample prepare

/*!
 * \brief Indicates if EGBLAS has bernoulli_sample_prepare
 */
#ifdef EGBLAS_HAS_BERNOULLI_SAMPLE_PREPARE
static constexpr bool has_bernoulli_sample_prepare = true;
#else
static constexpr bool has_bernoulli_sample_prepare = false;
#endif

/*!
 * \brief Prepare random states for bernoulli_sample
 * \return random states for bernoulli_sample
 */
inline void* bernoulli_sample_prepare() {
#ifdef EGBLAS_HAS_BERNOULLI_SAMPLE_PREPARE
    inc_counter("egblas");
    return egblas_bernoulli_sample_prepare();
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample_prepare");
    return nullptr;
#endif
}

/*!
 * \brief Indicates if EGBLAS has bernoulli_sample_prepare
 */
#ifdef EGBLAS_HAS_BERNOULLI_SAMPLE_PREPARE_SEED
static constexpr bool has_bernoulli_sample_prepare_seed = true;
#else
static constexpr bool has_bernoulli_sample_prepare_seed = false;
#endif

/*!
 * \brief Prepare random states for bernoulli_sample with the given seed
 * \param seed The seed
 * \return random states for bernoulli_sample
 */
inline void* bernoulli_sample_prepare_seed([[maybe_unused]] size_t seed) {
#ifdef EGBLAS_HAS_BERNOULLI_SAMPLE_PREPARE_SEED
    inc_counter("egblas");
    return egblas_bernoulli_sample_prepare_seed(seed);
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample_prepare_seed");
    return nullptr;
#endif
}

/*!
 * \brief Indicates if EGBLAS has bernoulli_sample_prepare
 */
#ifdef EGBLAS_HAS_BERNOULLI_SAMPLE_RELEASE
static constexpr bool has_bernoulli_sample_release = true;
#else
static constexpr bool has_bernoulli_sample_release = false;
#endif

/*!
 * \brief Prepare random states for bernoulli_sample with the given seed
 * \param seed The seed
 * \return random states for bernoulli_sample
 */
inline void bernoulli_sample_release([[maybe_unused]] void* state) {
#ifdef EGBLAS_HAS_BERNOULLI_SAMPLE_RELEASE
    inc_counter("egblas");
    egblas_bernoulli_sample_release(state);
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample_release");
#endif
}

// bernoulli_sample

/*!
 * \brief Indicates if EGBLAS has single-precision bernoulli_sample.
 */
#ifdef EGBLAS_HAS_SBERNOULLI_SAMPLE
static constexpr bool has_sbernoulli_sample = true;
#else
static constexpr bool has_sbernoulli_sample = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bernoulli_sample
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void bernoulli_sample([[maybe_unused]] size_t       n,
                           [[maybe_unused]] float        alpha,
                           [[maybe_unused]] const float* A,
                           [[maybe_unused]] size_t       lda,
                           [[maybe_unused]] float*       C,
                           [[maybe_unused]] size_t       ldc) {
#ifdef EGBLAS_HAS_SBERNOULLI_SAMPLE
    inc_counter("egblas");
    egblas_sbernoulli_sample(n, alpha, A, lda, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision bernoulli_sample.
 */
#ifdef EGBLAS_HAS_DBERNOULLI_SAMPLE
static constexpr bool has_dbernoulli_sample = true;
#else
static constexpr bool has_dbernoulli_sample = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas bernoulli_sample
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void bernoulli_sample([[maybe_unused]] size_t        n,
                           [[maybe_unused]] double        alpha,
                           [[maybe_unused]] const double* A,
                           [[maybe_unused]] size_t        lda,
                           [[maybe_unused]] double*       C,
                           [[maybe_unused]] size_t        ldc) {
#ifdef EGBLAS_HAS_DBERNOULLI_SAMPLE
    inc_counter("egblas");
    egblas_dbernoulli_sample(n, alpha, A, lda, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample");
#endif
}

// bernoulli_sample_seed

/*!
 * \brief Indicates if EGBLAS has single-precision bernoulli_sample.
 */
#ifdef EGBLAS_HAS_SBERNOULLI_SAMPLE_SEED
static constexpr bool has_sbernoulli_sample_seed = true;
#else
static constexpr bool has_sbernoulli_sample_seed = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bernoulli_sample
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void bernoulli_sample_seed([[maybe_unused]] size_t       n,
                                [[maybe_unused]] float        alpha,
                                [[maybe_unused]] const float* A,
                                [[maybe_unused]] size_t       lda,
                                [[maybe_unused]] float*       C,
                                [[maybe_unused]] size_t       ldc,
                                [[maybe_unused]] size_t       seed) {
#ifdef EGBLAS_HAS_SBERNOULLI_SAMPLE_SEED
    inc_counter("egblas");
    egblas_sbernoulli_sample_seed(n, alpha, A, lda, C, ldc, seed);
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision bernoulli_sample.
 */
#ifdef EGBLAS_HAS_DBERNOULLI_SAMPLE_SEED
static constexpr bool has_dbernoulli_sample_seed = true;
#else
static constexpr bool has_dbernoulli_sample_seed = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas bernoulli_sample
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void bernoulli_sample_seed([[maybe_unused]] size_t        n,
                                [[maybe_unused]] double        alpha,
                                [[maybe_unused]] const double* A,
                                [[maybe_unused]] size_t        lda,
                                [[maybe_unused]] double*       C,
                                [[maybe_unused]] size_t        ldc,
                                [[maybe_unused]] size_t        seed) {
#ifdef EGBLAS_HAS_DBERNOULLI_SAMPLE_SEED
    inc_counter("egblas");
    egblas_dbernoulli_sample_seed(n, alpha, A, lda, C, ldc, seed);
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample");
#endif
}

// bernoulli_sample_states

/*!
 * \brief Indicates if EGBLAS has single-precision bernoulli_sample.
 */
#ifdef EGBLAS_HAS_SBERNOULLI_SAMPLE_STATES
static constexpr bool has_sbernoulli_sample_states = true;
#else
static constexpr bool has_sbernoulli_sample_states = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas bernoulli_sample
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void bernoulli_sample_states([[maybe_unused]] size_t       n,
                                  [[maybe_unused]] float        alpha,
                                  [[maybe_unused]] const float* A,
                                  [[maybe_unused]] size_t       lda,
                                  [[maybe_unused]] float*       C,
                                  [[maybe_unused]] size_t       ldc,
                                  [[maybe_unused]] void*        states) {
#ifdef EGBLAS_HAS_SBERNOULLI_SAMPLE_STATES
    inc_counter("egblas");
    egblas_sbernoulli_sample_states(n, alpha, A, lda, C, ldc, states);
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample_states");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision bernoulli_sample.
 */
#ifdef EGBLAS_HAS_DBERNOULLI_SAMPLE_STATES
static constexpr bool has_dbernoulli_sample_states = true;
#else
static constexpr bool has_dbernoulli_sample_states = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas bernoulli_sample
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void bernoulli_sample_states([[maybe_unused]] size_t        n,
                                  [[maybe_unused]] double        alpha,
                                  [[maybe_unused]] const double* A,
                                  [[maybe_unused]] size_t        lda,
                                  [[maybe_unused]] double*       C,
                                  [[maybe_unused]] size_t        ldc,
                                  [[maybe_unused]] void*         states) {
#ifdef EGBLAS_HAS_DBERNOULLI_SAMPLE_STATES
    inc_counter("egblas");
    egblas_dbernoulli_sample_states(n, alpha, A, lda, C, ldc, states);
#else
    cpp_unreachable("Invalid call to egblas::bernoulli_sample_states");
#endif
}

} //end of namespace etl::impl::egblas
