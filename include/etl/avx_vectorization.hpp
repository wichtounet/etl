//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_AVX_VECTORIZATION_HPP
#define ETL_AVX_VECTORIZATION_HPP

#include <immintrin.h>

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

inline __m256d loadu(const double* memory){
    return _mm256_loadu_pd(memory);
}

inline __m256 loadu(const float* memory){
    return _mm256_loadu_ps(memory);
}

inline __m256d set(double value){
    return _mm256_set1_pd(value);
}

inline __m256 set(float value){
    return _mm256_set1_ps(value);
}

inline __m256d add(__m256d lhs, __m256d rhs){
    return _mm256_add_pd(lhs, rhs);
}

inline __m256d sub(__m256d lhs, __m256d rhs){
    return _mm256_sub_pd(lhs, rhs);
}

inline __m256d mul(__m256d lhs, __m256d rhs){
    return _mm256_mul_pd(lhs, rhs);
}

inline __m256d div(__m256d lhs, __m256d rhs){
    return _mm256_div_pd(lhs, rhs);
}

inline __m256 add(__m256 lhs, __m256 rhs){
    return _mm256_add_ps(lhs, rhs);
}

inline __m256 sub(__m256 lhs, __m256 rhs){
    return _mm256_sub_ps(lhs, rhs);
}

inline __m256 mul(__m256 lhs, __m256 rhs){
    return _mm256_mul_ps(lhs, rhs);
}

inline __m256 div(__m256 lhs, __m256 rhs){
    return _mm256_div_ps(lhs, rhs);
}

} //end of namespace vec

} //end of namespace etl

#endif
