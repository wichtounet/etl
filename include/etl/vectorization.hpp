//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file vectorization.hpp
 * \brief Contais vectorization utilities for the vectorized assignments (done by the evaluator).
 *
 * This file automatically includes the correct header based on which vectorization utility is supported (AVX -> SSE -> NONE).
 */

#pragma once

#include <immintrin.h>

namespace etl {

/*!
 * \brief Define traits to get vectorization information for types.
 *
 * This traits are overloaded by SSE/AVX implementation for types that are vectorizable.
 */
template <typename T>
struct intrinsic_traits {
    static constexpr const bool vectorizable     = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr const std::size_t size      = 1;          ///< Numbers of elements done at once
    static constexpr const std::size_t alignment = alignof(T); ///< Necessary number of bytes of alignment for this type

    using intrinsic_type = T;
};

/*!
 * \brief Helper to get the intrinsic corresponding type of a vectorizable type.
 */
template <typename T>
using intrinsic_type = typename intrinsic_traits<T>::intrinsic_type;

} //end of namespace etl

#ifdef ETL_VECTORIZE_EXPR

#ifdef __AVX512F__

#include "etl/avx512_vectorization.hpp"

#elif defined(__AVX__)

#include "etl/avx_vectorization.hpp"

#elif defined(__SSE3__)

#include "etl/sse_vectorization.hpp"

#else

#include "etl/no_vectorization.hpp"

#endif //defined(__SSE__)

#else //ETL_VECTORIZE_EXPR

#include "etl/no_vectorization.hpp"

#endif //ETL_VECTORIZE_EXPR
