//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file vectorization.hpp
 * \brief Contains vectorization utilities for the vectorized assignments (done by the evaluator).
 *
 * This file automatically includes the correct header based on which vectorization utility is supported (AVX -> SSE -> NONE).
 */

#pragma once

#include <immintrin.h>

//Include al the vector implementation
#include "etl/avx512_vectorization.hpp"
#include "etl/avx_vectorization.hpp"
#include "etl/sse_vectorization.hpp"
#include "etl/no_vectorization.hpp"
#include "etl/inline.hpp"

namespace etl {

/*!
 * \brief Traits to get the intrinsic traits for a vector mode
 * \tparam V The vector mode
 */
template <vector_mode_t V>
struct get_intrinsic_traits {
    /*!
     * \brief The type of the intrinsic traits for T
     * \tparam T The type to get the intrinsic traits for
     */
    template <typename T>
    using type = no_intrinsic_traits<T>;
};

/*!
 * \brief Traits to get the vector implementation for a vector mode
 * \tparam V The vector mode
 */
template <vector_mode_t V>
struct get_vector_impl {
    using type = no_vec; ///< The vector implementation
};

#ifdef __AVX512F__

/*!
 * \brief get_intrinsic_traits for AVX-512
 */
template <>
struct get_intrinsic_traits<vector_mode_t::AVX512> {
    template <typename T>
    using type = avx512_intrinsic_traits<T>;
};

template <>
struct get_vector_impl<vector_mode_t::AVX512> {
    using type = avx512_vec;
};

#endif

#ifdef __AVX__

/*!
 * \brief get_intrinsic_traits for AVX
 */
template <>
struct get_intrinsic_traits<vector_mode_t::AVX> {
    template <typename T>
    using type = avx_intrinsic_traits<T>;
};

template <>
struct get_vector_impl<vector_mode_t::AVX> {
    using type = avx_vec;
};

#endif

#ifdef __SSE3__

/*!
 * \brief get_intrinsic_traits for SSE
 */
template <>
struct get_intrinsic_traits<vector_mode_t::SSE3> {
    template <typename T>
    using type = sse_intrinsic_traits<T>;
};

template <>
struct get_vector_impl<vector_mode_t::SSE3> {
    using type = sse_vec;
};

#endif

#ifdef ETL_VECTORIZE_EXPR

#ifdef __AVX512F__

using default_vec = avx512_vec;

template <typename T>
using intrinsic_traits = avx512_intrinsic_traits<T>;

#elif defined(__AVX__)

using default_vec = avx_vec;

template <typename T>
using intrinsic_traits = avx_intrinsic_traits<T>;

#elif defined(__SSE3__)

using default_vec = sse_vec;

template <typename T>
using intrinsic_traits = sse_intrinsic_traits<T>;

#else

using default_vec = no_vec;

template <typename T>
using intrinsic_traits = no_intrinsic_traits<T>;

#endif //defined(__SSE__)

#else //ETL_VECTORIZE_EXPR

/*!
 * \brief The defautl vectorization scheme
 */
using default_vec = no_vec;

/*!
 * \brief The default intrinsic traits
 */
template <typename T>
using intrinsic_traits = no_intrinsic_traits<T>;

#endif //ETL_VECTORIZE_EXPR

/*!
 * \brief Helper to get the intrinsic corresponding type of a vectorizable type.
 */
template <typename T>
using intrinsic_type = typename intrinsic_traits<T>::intrinsic_type;

} //end of namespace etl
