//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_SSE_VECTORIZATION_HPP
#define ETL_SSE_VECTORIZATION_HPP

#include <immintrin.h>

namespace etl {

template<>
struct intrinsic_traits <double> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 2;

    using intrinsic_type = __m128d;
};

template<>
struct intrinsic_traits <float> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 4;

    using intrinsic_type = __m128;
};

namespace vec {

inline void store(double* memory, __m128d value){
    _mm_storeu_pd(memory, value);
}

inline void store(float* memory, __m128 value){
    _mm_storeu_ps(memory, value);
}

inline __m128d loadu(const double* memory){
    return _mm_loadu_pd(memory);
}

inline __m128 loadu(const float* memory){
    return _mm_loadu_ps(memory);
}

inline __m128d set(double value){
    return _mm_set1_pd(value);
}

inline __m128 set(float value){
    return _mm_set1_ps(value);
}

inline __m128d add(__m128d lhs, __m128d rhs){
    return _mm_add_pd(lhs, rhs);
}

inline __m128d sub(__m128d lhs, __m128d rhs){
    return _mm_sub_pd(lhs, rhs);
}

inline __m128d mul(__m128d lhs, __m128d rhs){
    return _mm_mul_pd(lhs, rhs);
}

inline __m128d div(__m128d lhs, __m128d rhs){
    return _mm_div_pd(lhs, rhs);
}

inline __m128d sqrt(__m128d x){
    return _mm_sqrt_pd(x);
}

inline __m128d minus(__m128d x){
    return _mm_xor_pd(x, _mm_set1_pd(-0.f));
}

inline __m128 add(__m128 lhs, __m128 rhs){
    return _mm_add_ps(lhs, rhs);
}

inline __m128 sub(__m128 lhs, __m128 rhs){
    return _mm_sub_ps(lhs, rhs);
}

inline __m128 mul(__m128 lhs, __m128 rhs){
    return _mm_mul_ps(lhs, rhs);
}

inline __m128 div(__m128 lhs, __m128 rhs){
    return _mm_div_ps(lhs, rhs);
}

inline __m128 sqrt(__m128 x){
    return _mm_sqrt_ps(x);
}

inline __m128 minus(__m128 x){
    return _mm_xor_ps(x, _mm_set1_ps(-0.f));
}

} //end of namespace vec

} //end of namespace etl

#endif
