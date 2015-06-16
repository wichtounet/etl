//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <immintrin.h>

#ifdef __clang__
#define ETL_INLINE_VEC_128 inline __m128 __attribute__((__always_inline__, __nodebug__))
#define ETL_INLINE_VEC_128D inline __m128d __attribute__((__always_inline__, __nodebug__))
#else
#define ETL_INLINE_VEC_128 inline __m128 __attribute__((__always_inline__))
#define ETL_INLINE_VEC_128D inline __m128d __attribute__((__always_inline__))
#endif

namespace etl {

template<>
struct intrinsic_traits <double> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 2;
    static constexpr const std::size_t alignment = 16;

    using intrinsic_type = __m128d;
};

template<>
struct intrinsic_traits <float> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 4;
    static constexpr const std::size_t alignment = 16;

    using intrinsic_type = __m128;
};

namespace vec {

inline void storeu(double* memory, __m128d value){
    _mm_storeu_pd(memory, value);
}

inline void storeu(float* memory, __m128 value){
    _mm_storeu_ps(memory, value);
}

inline void store(double* memory, __m128d value){
    _mm_store_pd(memory, value);
}

inline void store(float* memory, __m128 value){
    _mm_store_ps(memory, value);
}

ETL_INLINE_VEC_128D loadu(const double* memory){
    return _mm_loadu_pd(memory);
}

ETL_INLINE_VEC_128 loadu(const float* memory){
    return _mm_loadu_ps(memory);
}

ETL_INLINE_VEC_128D set(double value){
    return _mm_set1_pd(value);
}

ETL_INLINE_VEC_128 set(float value){
    return _mm_set1_ps(value);
}

ETL_INLINE_VEC_128D add(__m128d lhs, __m128d rhs){
    return _mm_add_pd(lhs, rhs);
}

ETL_INLINE_VEC_128D sub(__m128d lhs, __m128d rhs){
    return _mm_sub_pd(lhs, rhs);
}

ETL_INLINE_VEC_128D mul(__m128d lhs, __m128d rhs){
    return _mm_mul_pd(lhs, rhs);
}

ETL_INLINE_VEC_128D div(__m128d lhs, __m128d rhs){
    return _mm_div_pd(lhs, rhs);
}

ETL_INLINE_VEC_128D sqrt(__m128d x){
    return _mm_sqrt_pd(x);
}

ETL_INLINE_VEC_128D minus(__m128d x){
    return _mm_xor_pd(x, _mm_set1_pd(-0.f));
}

ETL_INLINE_VEC_128 add(__m128 lhs, __m128 rhs){
    return _mm_add_ps(lhs, rhs);
}

ETL_INLINE_VEC_128 sub(__m128 lhs, __m128 rhs){
    return _mm_sub_ps(lhs, rhs);
}

ETL_INLINE_VEC_128 mul(__m128 lhs, __m128 rhs){
    return _mm_mul_ps(lhs, rhs);
}

ETL_INLINE_VEC_128 div(__m128 lhs, __m128 rhs){
    return _mm_div_ps(lhs, rhs);
}

ETL_INLINE_VEC_128 sqrt(__m128 x){
    return _mm_sqrt_ps(x);
}

ETL_INLINE_VEC_128 minus(__m128 x){
    return _mm_xor_ps(x, _mm_set1_ps(-0.f));
}

//The Intel C++ Compiler (icc) has more intrinsics.
//ETL uses them when compiled with icc

#ifdef __INTEL_COMPILER

//Exponential

ETL_INLINE_VEC_128D exp(__m128d x){
    return _mm_exp_pd(x);
}

ETL_INLINE_VEC_128 exp(__m128 x){
    return _mm_exp_ps(x);
}

//Logarithm

ETL_INLINE_VEC_128D log(__m128d x){
    return _mm_log_pd(x);
}

ETL_INLINE_VEC_128 log(__m128 x){
    return _mm_log_ps(x);
}

//Min

ETL_INLINE_VEC_128D min(__m128d lhs, __m128d rhs){
    return _mm_min_pd(lhs, rhs);
}

ETL_INLINE_VEC_128 min(__m128 lhs, __m128 rhs){
    return _mm_min_ps(lhs, rhs);
}

//Max

ETL_INLINE_VEC_128D max(__m128d lhs, __m128d rhs){
    return _mm_max_pd(lhs, rhs);
}

ETL_INLINE_VEC_128 max(__m128 lhs, __m128 rhs){
    return _mm_max_ps(lhs, rhs);
}

#endif

} //end of namespace vec

} //end of namespace etl
