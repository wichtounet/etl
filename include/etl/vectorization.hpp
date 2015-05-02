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
};

#ifdef ETL_VECTORIZE

#ifdef __AVX__

template<>
struct intrinsic_traits <double> {
    static constexpr const bool vectorizable = true;

    using intrinsic_type = __m256d;
};

template<>
struct intrinsic_traits <float> {
    static constexpr const bool vectorizable = true;

    using intrinsic_type = __m256;
};

inline void store(double* memory, __m256d value){
    _mm256_storeu_pd(memory, value);
}

inline void store(float* memory, __m256 value){
    _mm256_storeu_ps(memory, value);
}

#elif defined(__SSE__)

template<>
struct intrinsic_traits <double> {
    static constexpr const bool vectorizable = true;

    using intrinsic_type = __m128d;
};

template<>
struct intrinsic_traits <float> {
    static constexpr const bool vectorizable = true;

    using intrinsic_type = __m128;
};

inline void store(double* memory, __m128d value){
    _mm_storeu_pd(memory, value);
}

inline void store(float* memory, __m128 value){
    _mm_storeu_ps(memory, value);
}

#endif

#endif

} //end of namespace etl

#endif
