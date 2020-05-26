//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief EGBLAS wrappers for the logistic_noise operation.
 */

#pragma once

#ifdef ETL_EGBLAS_MODE

#include "etl/impl/cublas/cuda.hpp"

#include <egblas.hpp>

#endif

namespace etl::impl::egblas {

// logistic_noise prepare

/*!
 * \brief Indicates if EGBLAS has logistic_noise_prepare
 */
#ifdef EGBLAS_HAS_LOGISTIC_NOISE_PREPARE
static constexpr bool has_logistic_noise_prepare = true;
#else
static constexpr bool has_logistic_noise_prepare = false;
#endif

/*!
 * \brief Prepare random states for logistic_noise
 * \return random states for logistic_noise
 */
inline void* logistic_noise_prepare() {
#ifdef EGBLAS_HAS_LOGISTIC_NOISE_PREPARE
    inc_counter("egblas");
    return egblas_logistic_noise_prepare();
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise_prepare");
    return nullptr;
#endif
}

/*!
 * \brief Indicates if EGBLAS has logistic_noise_prepare
 */
#ifdef EGBLAS_HAS_LOGISTIC_NOISE_PREPARE_SEED
static constexpr bool has_logistic_noise_prepare_seed = true;
#else
static constexpr bool has_logistic_noise_prepare_seed = false;
#endif

/*!
 * \brief Prepare random states for logistic_noise with the given seed
 * \param seed The seed
 * \return random states for logistic_noise
 */
inline void* logistic_noise_prepare_seed([[maybe_unused]] size_t seed) {
#ifdef EGBLAS_HAS_LOGISTIC_NOISE_PREPARE_SEED
    inc_counter("egblas");
    return egblas_logistic_noise_prepare_seed(seed);
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise_prepare_seed");
    return nullptr;
#endif
}

/*!
 * \brief Indicates if EGBLAS has logistic_noise_prepare
 */
#ifdef EGBLAS_HAS_LOGISTIC_NOISE_RELEASE
static constexpr bool has_logistic_noise_release = true;
#else
static constexpr bool has_logistic_noise_release = false;
#endif

/*!
 * \brief Prepare random states for logistic_noise with the given seed
 * \param seed The seed
 * \return random states for logistic_noise
 */
inline void logistic_noise_release([[maybe_unused]] void* state) {
#ifdef EGBLAS_HAS_LOGISTIC_NOISE_RELEASE
    inc_counter("egblas");
    egblas_logistic_noise_release(state);
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise_release");
#endif
}

// logistic_noise

/*!
 * \brief Indicates if EGBLAS has single-precision logistic_noise.
 */
#ifdef EGBLAS_HAS_SLOGISTIC_NOISE
static constexpr bool has_slogistic_noise = true;
#else
static constexpr bool has_slogistic_noise = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas logistic_noise
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void logistic_noise([[maybe_unused]] size_t       n,
                           [[maybe_unused]] float        alpha,
                           [[maybe_unused]] const float* A,
                           [[maybe_unused]] size_t       lda,
                           [[maybe_unused]] float*       C,
                           [[maybe_unused]] size_t       ldc) {
#ifdef EGBLAS_HAS_SLOGISTIC_NOISE
    inc_counter("egblas");
    egblas_slogistic_noise(n, alpha, A, lda, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision logistic_noise.
 */
#ifdef EGBLAS_HAS_DLOGISTIC_NOISE
static constexpr bool has_dlogistic_noise = true;
#else
static constexpr bool has_dlogistic_noise = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas logistic_noise
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void logistic_noise([[maybe_unused]] size_t        n,
                           [[maybe_unused]] double        alpha,
                           [[maybe_unused]] const double* A,
                           [[maybe_unused]] size_t        lda,
                           [[maybe_unused]] double*       C,
                           [[maybe_unused]] size_t        ldc) {
#ifdef EGBLAS_HAS_DLOGISTIC_NOISE
    inc_counter("egblas");
    egblas_dlogistic_noise(n, alpha, A, lda, C, ldc);
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise");
#endif
}

// logistic_noise_seed

/*!
 * \brief Indicates if EGBLAS has single-precision logistic_noise.
 */
#ifdef EGBLAS_HAS_SLOGISTIC_NOISE_SEED
static constexpr bool has_slogistic_noise_seed = true;
#else
static constexpr bool has_slogistic_noise_seed = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas logistic_noise
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void logistic_noise_seed([[maybe_unused]] size_t       n,
                                [[maybe_unused]] float        alpha,
                                [[maybe_unused]] const float* A,
                                [[maybe_unused]] size_t       lda,
                                [[maybe_unused]] float*       C,
                                [[maybe_unused]] size_t       ldc,
                                [[maybe_unused]] size_t       seed) {
#ifdef EGBLAS_HAS_SLOGISTIC_NOISE_SEED
    inc_counter("egblas");
    egblas_slogistic_noise_seed(n, alpha, A, lda, C, ldc, seed);
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision logistic_noise.
 */
#ifdef EGBLAS_HAS_DLOGISTIC_NOISE_SEED
static constexpr bool has_dlogistic_noise_seed = true;
#else
static constexpr bool has_dlogistic_noise_seed = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas logistic_noise
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void logistic_noise_seed([[maybe_unused]] size_t        n,
                                [[maybe_unused]] double        alpha,
                                [[maybe_unused]] const double* A,
                                [[maybe_unused]] size_t        lda,
                                [[maybe_unused]] double*       C,
                                [[maybe_unused]] size_t        ldc,
                                [[maybe_unused]] size_t        seed) {
#ifdef EGBLAS_HAS_DLOGISTIC_NOISE_SEED
    inc_counter("egblas");
    egblas_dlogistic_noise_seed(n, alpha, A, lda, C, ldc, seed);
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise");
#endif
}

// logistic_noise_states

/*!
 * \brief Indicates if EGBLAS has single-precision logistic_noise.
 */
#ifdef EGBLAS_HAS_SLOGISTIC_NOISE_STATES
static constexpr bool has_slogistic_noise_states = true;
#else
static constexpr bool has_slogistic_noise_states = false;
#endif

/*!
 * \brief Wrappers for single-precision egblas logistic_noise
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void logistic_noise_states([[maybe_unused]] size_t       n,
                                  [[maybe_unused]] float        alpha,
                                  [[maybe_unused]] const float* A,
                                  [[maybe_unused]] size_t       lda,
                                  [[maybe_unused]] float*       C,
                                  [[maybe_unused]] size_t       ldc,
                                  [[maybe_unused]] void*        states) {
#ifdef EGBLAS_HAS_SLOGISTIC_NOISE_STATES
    inc_counter("egblas");
    egblas_slogistic_noise_states(n, alpha, A, lda, C, ldc, states);
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise_states");
#endif
}

/*!
 * \brief Indicates if EGBLAS has double-precision logistic_noise.
 */
#ifdef EGBLAS_HAS_DLOGISTIC_NOISE_STATES
static constexpr bool has_dlogistic_noise_states = true;
#else
static constexpr bool has_dlogistic_noise_states = false;
#endif

/*!
 * \brief Wrappers for double-precision egblas logistic_noise
 * \param n The size of the vector
 * \param alpha The scaling factor alpha
 * \param A The memory of the vector a
 * \param lda The leading dimension of a
 * \param C The memory of the vector c
 * \param ldc The leading dimension of c
 */
inline void logistic_noise_states([[maybe_unused]] size_t        n,
                                  [[maybe_unused]] double        alpha,
                                  [[maybe_unused]] const double* A,
                                  [[maybe_unused]] size_t        lda,
                                  [[maybe_unused]] double*       C,
                                  [[maybe_unused]] size_t        ldc,
                                  [[maybe_unused]] void*         states) {
#ifdef EGBLAS_HAS_DLOGISTIC_NOISE_STATES
    inc_counter("egblas");
    egblas_dlogistic_noise_states(n, alpha, A, lda, C, ldc, states);
#else
    cpp_unreachable("Invalid call to egblas::logistic_noise_states");
#endif
}

} //end of namespace etl::impl::egblas
