//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_VECTORIZATION_HPP
#define ETL_VECTORIZATION_HPP

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

#include "avx_vectorization.hpp"

#elif defined(__SSE__)

#include "sse_vectorization.hpp"

#endif //defined(__SSE__)

#else //ETL_VECTORIZE_EXPR

#include "no_vectorization.hpp"

#endif //ETL_VECTORIZE_EXPR

#endif //ETL_VECTORIZATION_HPP
