//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file avx512_vectorization.hpp
 * \brief Contains AVX-512 vectorized functions for the vectorized assignment of expressions
 */

//TODO Implementation of AVX-512 complex multiplication and division

#pragma once

#include <immintrin.h>

#ifdef VECT_DEBUG
#include <iostream>
#endif

#ifdef __clang__
#define ETL_INLINE_VEC_VOID inline void __attribute__((__always_inline__, __nodebug__))
#define ETL_INLINE_VEC_512 inline __m512 __attribute__((__always_inline__, __nodebug__))
#define ETL_INLINE_VEC_512D inline __m512d __attribute__((__always_inline__, __nodebug__))
#else
#define ETL_INLINE_VEC_VOID inline void __attribute__((__always_inline__))
#define ETL_INLINE_VEC_512 inline __m512 __attribute__((__always_inline__))
#define ETL_INLINE_VEC_512D inline __m512d __attribute__((__always_inline__))
#endif

namespace etl {

template <>
struct intrinsic_traits<float> {
    static constexpr const bool vectorizable     = true;
    static constexpr const std::size_t size      = 16;
    static constexpr const std::size_t alignment = 64;

    using intrinsic_type = __m512;
};

template <>
struct intrinsic_traits<double> {
    static constexpr const bool vectorizable     = true;
    static constexpr const std::size_t size      = 8;
    static constexpr const std::size_t alignment = 64;

    using intrinsic_type = __m512d;
};

template <>
struct intrinsic_traits<std::complex<float>> {
    static constexpr const bool vectorizable     = true;
    static constexpr const std::size_t size      = 8;
    static constexpr const std::size_t alignment = 64;

    using intrinsic_type = __m512;
};

template <>
struct intrinsic_traits<std::complex<double>> {
    static constexpr const bool vectorizable     = true;
    static constexpr const std::size_t size      = 4;
    static constexpr const std::size_t alignment = 64;

    using intrinsic_type = __m512d;
};

namespace vec {

#ifdef VEC_DEBUG

template <typename T>
std::string debug_d(T value) {
    union test {
        __m512d vec;
        double array[8];
        test(__m512d vec)
                : vec(vec) {}
    };

    test u_value = value;
    std::cout << "["
              << u_value.array[0] << "," << u_value.array[1] << "," << u_value.array[2] << "," << u_value.array[3]
              << "," << u_value.array[4] << "," << u_value.array[5] << "," << u_value.array[6] << "," << u_value.array[7]
              << "]" << std::endl;
}

template <typename T>
std::string debug_s(T value) {
    union test {
        __m512 vec;
        float array[16];
        test(__m512 vec)
                : vec(vec) {}
    };

    test u_value = value;
    std::cout << "["
              << u_value.array[0] << "," << u_value.array[1] << "," << u_value.array[2] << "," << u_value.array[3]
              << "," << u_value.array[4] << "," << u_value.array[5] << "," << u_value.array[6] << "," << u_value.array[7]
              << "," << u_value.array[8] << "," << u_value.array[9] << "," << u_value.array[10] << "," << u_value.array[11]
              << "," << u_value.array[12] << "," << u_value.array[13] << "," << u_value.array[14] << "," << u_value.array[15]
              << "]" << std::endl;
}

#else

template <typename T>
std::string debug_d(T) {
    return "";
}

template <typename T>
std::string debug_s(T) {
    return "";
}

#endif

ETL_INLINE_VEC_VOID storeu(float* memory, __m512 value) {
    _mm512_storeu_ps(memory, value);
}

ETL_INLINE_VEC_VOID storeu(double* memory, __m512d value) {
    _mm512_storeu_pd(memory, value);
}

ETL_INLINE_VEC_VOID storeu(std::complex<float>* memory, __m512 value) {
    _mm512_storeu_ps(reinterpret_cast<float*>(memory), value);
}

ETL_INLINE_VEC_VOID storeu(std::complex<double>* memory, __m512d value) {
    _mm512_storeu_pd(reinterpret_cast<double*>(memory), value);
}

ETL_INLINE_VEC_VOID store(float* memory, __m512 value) {
    _mm512_store_ps(memory, value);
}

ETL_INLINE_VEC_VOID store(double* memory, __m512d value) {
    _mm512_store_pd(memory, value);
}

ETL_INLINE_VEC_VOID store(std::complex<float>* memory, __m512 value) {
    _mm512_store_ps(reinterpret_cast<float*>(memory), value);
}

ETL_INLINE_VEC_VOID store(std::complex<double>* memory, __m512d value) {
    _mm512_store_pd(reinterpret_cast<double*>(memory), value);
}

ETL_INLINE_VEC_512 load(const float* memory) {
    return _mm512_load_ps(memory);
}

ETL_INLINE_VEC_512D load(const double* memory) {
    return _mm512_load_pd(memory);
}

ETL_INLINE_VEC_512 load(const std::complex<float>* memory) {
    return _mm512_load_ps(reinterpret_cast<const float*>(memory));
}

ETL_INLINE_VEC_512D load(const std::complex<double>* memory) {
    return _mm512_load_pd(reinterpret_cast<const double*>(memory));
}

ETL_INLINE_VEC_512 loadu(const float* memory) {
    return _mm512_loadu_ps(memory);
}

ETL_INLINE_VEC_512D loadu(const double* memory) {
    return _mm512_loadu_pd(memory);
}

ETL_INLINE_VEC_512 loadu(const std::complex<float>* memory) {
    return _mm512_loadu_ps(reinterpret_cast<const float*>(memory));
}

ETL_INLINE_VEC_512D loadu(const std::complex<double>* memory) {
    return _mm512_loadu_pd(reinterpret_cast<const double*>(memory));
}

ETL_INLINE_VEC_512D set(double value) {
    return _mm512_set1_pd(value);
}

ETL_INLINE_VEC_512 set(float value) {
    return _mm512_set1_ps(value);
}

ETL_INLINE_VEC_512D add(__m512d lhs, __m512d rhs) {
    return _mm512_add_pd(lhs, rhs);
}

ETL_INLINE_VEC_512D sub(__m512d lhs, __m512d rhs) {
    return _mm512_sub_pd(lhs, rhs);
}

ETL_INLINE_VEC_512D sqrt(__m512d x) {
    return _mm512_sqrt_pd(x);
}

ETL_INLINE_VEC_512D minus(__m512d x) {
    return _mm512_xor_pd(x, _mm512_set1_pd(-0.f));
}

ETL_INLINE_VEC_512 add(__m512 lhs, __m512 rhs) {
    return _mm512_add_ps(lhs, rhs);
}

ETL_INLINE_VEC_512 sub(__m512 lhs, __m512 rhs) {
    return _mm512_sub_ps(lhs, rhs);
}

ETL_INLINE_VEC_512 sqrt(__m512 lhs) {
    return _mm512_sqrt_ps(lhs);
}

ETL_INLINE_VEC_512 minus(__m512 x) {
    return _mm512_xor_ps(x, _mm512_set1_ps(-0.f));
}

template <bool Complex = false>
ETL_INLINE_VEC_512 mul(__m512 lhs, __m512 rhs) {
    return _mm512_mul_ps(lhs, rhs);
}

template <>
ETL_INLINE_VEC_512 mul<true>(__m512 lhs, __m512 rhs) {
    cpp_unreachable("Not yet implemented");
    return lhs;
}

template <bool Complex = false>
ETL_INLINE_VEC_512D mul(__m512d lhs, __m512d rhs) {
    return _mm512_mul_pd(lhs, rhs);
}

template <>
ETL_INLINE_VEC_512D mul<true>(__m512d lhs, __m512d rhs) {
    cpp_unreachable("Not yet implemented");
    return lhs;
}

template <bool Complex = false>
ETL_INLINE_VEC_512 div(__m512 lhs, __m512 rhs) {
    return _mm512_div_ps(lhs, rhs);
}

template <>
ETL_INLINE_VEC_512 div<true>(__m512 lhs, __m512 rhs) {
    cpp_unreachable("Not yet implemented");
    return lhs;
}

template <bool Complex = false>
ETL_INLINE_VEC_512D div(__m512d lhs, __m512d rhs) {
    return _mm512_div_pd(lhs, rhs);
}

template <>
ETL_INLINE_VEC_512D div<true>(__m512d lhs, __m512d rhs) {
    cpp_unreachable("Not yet implemented");
    return lhs;
}

#ifdef __INTEL_COMPILER

//Exponential

ETL_INLINE_VEC_512D exp(__m512d x) {
    return _mm512_exp_pd(x);
}

ETL_INLINE_VEC_512 exp(__m512 x) {
    return _mm512_exp_ps(x);
}

//Logarithm

ETL_INLINE_VEC_512D log(__m512d x) {
    return _mm512_log_pd(x);
}

ETL_INLINE_VEC_512 log(__m512 x) {
    return _mm512_log_ps(x);
}

//Min

ETL_INLINE_VEC_512D min(__m512d lhs, __m512d rhs) {
    return _mm512_min_pd(lhs, rhs);
}

ETL_INLINE_VEC_512 min(__m512 lhs, __m512 rhs) {
    return _mm512_min_ps(lhs, rhs);
}

//Max

ETL_INLINE_VEC_512D max(__m512d lhs, __m512d rhs) {
    return _mm512_max_pd(lhs, rhs);
}

ETL_INLINE_VEC_512 max(__m512 lhs, __m512 rhs) {
    return _mm512_max_ps(lhs, rhs);
}

#endif

} //end of namespace vec

} //end of namespace etl
