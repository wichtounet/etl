//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_AVX_VECTORIZATION_HPP
#define ETL_AVX_VECTORIZATION_HPP

#include <immintrin.h>

#ifdef __INTEL_COMPILER
#define ETL_INLINE_VEC_256 inline __m256 __attribute__((__always_inline__)) 
#define ETL_INLINE_VEC_256D inline __m256d __attribute__((__always_inline__)) 
#else
#define ETL_INLINE_VEC_256 inline __m256 __attribute__((__always_inline__, __nodebug__)) 
#define ETL_INLINE_VEC_256D inline __m256d __attribute__((__always_inline__, __nodebug__)) 
#endif

namespace etl {

template<>
struct intrinsic_traits <double> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 4;

    using intrinsic_type = __m256d;
};

template<>
struct intrinsic_traits <float> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 8;

    using intrinsic_type = __m256;
};

namespace vec {

inline void store(double* memory, __m256d value){
    _mm256_storeu_pd(memory, value);
}

inline void store(float* memory, __m256 value){
    _mm256_storeu_ps(memory, value);
}

ETL_INLINE_VEC_256D loadu(const double* memory){
    return _mm256_loadu_pd(memory);
}

ETL_INLINE_VEC_256 loadu(const float* memory){
    return _mm256_loadu_ps(memory);
}

ETL_INLINE_VEC_256D set(double value){
    return _mm256_set1_pd(value);
}

ETL_INLINE_VEC_256 set(float value){
    return _mm256_set1_ps(value);
}

ETL_INLINE_VEC_256D add(__m256d lhs, __m256d rhs){
    return _mm256_add_pd(lhs, rhs);
}

ETL_INLINE_VEC_256D sub(__m256d lhs, __m256d rhs){
    return _mm256_sub_pd(lhs, rhs);
}

ETL_INLINE_VEC_256D mul(__m256d lhs, __m256d rhs){
    return _mm256_mul_pd(lhs, rhs);
}

ETL_INLINE_VEC_256D div(__m256d lhs, __m256d rhs){
    return _mm256_div_pd(lhs, rhs);
}

ETL_INLINE_VEC_256D sqrt(__m256d x){
    return _mm256_sqrt_pd(x);
}

ETL_INLINE_VEC_256D minus(__m256d x){
    return _mm256_xor_pd(x, _mm256_set1_pd(-0.f));
}

ETL_INLINE_VEC_256 add(__m256 lhs, __m256 rhs){
    return _mm256_add_ps(lhs, rhs);
}

ETL_INLINE_VEC_256 sub(__m256 lhs, __m256 rhs){
    return _mm256_sub_ps(lhs, rhs);
}

ETL_INLINE_VEC_256 mul(__m256 lhs, __m256 rhs){
    return _mm256_mul_ps(lhs, rhs);
}

ETL_INLINE_VEC_256 div(__m256 lhs, __m256 rhs){
    return _mm256_div_ps(lhs, rhs);
}

ETL_INLINE_VEC_256 sqrt(__m256 lhs){
    return _mm256_sqrt_ps(lhs);
}

ETL_INLINE_VEC_256 minus(__m256 x){
    return _mm256_xor_ps(x, _mm256_set1_ps(-0.f));
}

#ifdef __INTEL_COMPILER

ETL_INLINE_VEC_256D exp(__m256d x){
    return _mm256_sqrt_pd(x);
}

ETL_INLINE_VEC_256 exp(__m256 x){
    return _mm256_sqrt_ps(x);
}

#endif

} //end of namespace vec

} //end of namespace etl

#endif
