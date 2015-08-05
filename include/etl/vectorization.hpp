//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <immintrin.h>

namespace etl {

template<typename T>
struct intrinsic_traits {
    static constexpr const bool vectorizable = false;
    static constexpr const std::size_t alignment = 1;

    using intrinsic_type = T;
};

template<typename T>
using intrinsic_type = typename intrinsic_traits<T>::intrinsic_type;

} //end of namespace etl

#ifdef ETL_VECTORIZE_EXPR

#ifdef __AVX__

#include "etl/avx_vectorization.hpp"

#elif defined(__SSE3__)

#include "etl/sse_vectorization.hpp"

#else

#include "etl/no_vectorization.hpp"

#endif //defined(__SSE__)

#else //ETL_VECTORIZE_EXPR

#include "etl/no_vectorization.hpp"

#endif //ETL_VECTORIZE_EXPR
