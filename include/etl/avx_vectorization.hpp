//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <immintrin.h>

#ifdef VECT_DEBUG
#include <iostream>
#endif

#ifdef __clang__
#define ETL_INLINE_VEC_256 inline __m256 __attribute__((__always_inline__, __nodebug__))
#define ETL_INLINE_VEC_256D inline __m256d __attribute__((__always_inline__, __nodebug__))
#else
#define ETL_INLINE_VEC_256 inline __m256 __attribute__((__always_inline__))
#define ETL_INLINE_VEC_256D inline __m256d __attribute__((__always_inline__))
#endif

namespace etl {

template<>
struct intrinsic_traits <float> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 8;
    static constexpr const std::size_t alignment = 32;

    using intrinsic_type = __m256;
};

template<>
struct intrinsic_traits <double> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 4;
    static constexpr const std::size_t alignment = 32;

    using intrinsic_type = __m256d;
};

template<>
struct intrinsic_traits <std::complex<float>> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 4;
    static constexpr const std::size_t alignment = 32;

    using intrinsic_type = __m256;
};

template<>
struct intrinsic_traits <std::complex<double>> {
    static constexpr const bool vectorizable = true;
    static constexpr const std::size_t size = 2;
    static constexpr const std::size_t alignment = 32;

    using intrinsic_type = __m256d;
};

namespace vec {

#ifdef VEC_DEBUG

template<typename T>
std::string debug_d(T value){
    union test {
        __m256d vec; // a data field, maybe a register, maybe not
        double array[4];
        test(__m256d vec) : vec(vec) {}
    };

    test u_value = value;
    std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << ","<< u_value.array[2] << ","<< u_value.array[3] << "]" << std::endl;
}

template<typename T>
std::string debug_s(T value){
    union test {
        __m256 vec; // a data field, maybe a register, maybe not
        float array[8];
        test(__m256 vec) : vec(vec) {}
    };

    test u_value = value;
    std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << ","<< u_value.array[2] << ","<< u_value.array[3]
              << "," << u_value.array[4] << "," << u_value.array[5] << ","<< u_value.array[6] << ","<< u_value.array[7] << "]" << std::endl;
}

#else

template<typename T>
std::string debug_d(T){}

template<typename T>
std::string debug_s(T){}

#endif

inline void storeu(float* memory, __m256 value){
    _mm256_storeu_ps(memory, value);
}

inline void storeu(double* memory, __m256d value){
    _mm256_storeu_pd(memory, value);
}

inline void storeu(std::complex<float>* memory, __m256 value){
    _mm256_storeu_ps(reinterpret_cast<float*>(memory), value);
}

inline void storeu(std::complex<double>* memory, __m256d value){
    _mm256_storeu_pd(reinterpret_cast<double*>(memory), value);
}

inline void store(float* memory, __m256 value){
    _mm256_store_ps(memory, value);
}

inline void store(double* memory, __m256d value){
    _mm256_store_pd(memory, value);
}

inline void store(std::complex<float>* memory, __m256 value){
    _mm256_store_ps(reinterpret_cast<float*>(memory), value);
}

inline void store(std::complex<double>* memory, __m256d value){
    _mm256_store_pd(reinterpret_cast<double*>(memory), value);
}

ETL_INLINE_VEC_256 loadu(const float* memory){
    return _mm256_loadu_ps(memory);
}

ETL_INLINE_VEC_256D loadu(const double* memory){
    return _mm256_loadu_pd(memory);
}

ETL_INLINE_VEC_256 loadu(const std::complex<float>* memory){
    return _mm256_loadu_ps(reinterpret_cast<const float*>(memory));
}

ETL_INLINE_VEC_256D loadu(const std::complex<double>* memory){
    return _mm256_loadu_pd(reinterpret_cast<const double*>(memory));
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

ETL_INLINE_VEC_256 sqrt(__m256 lhs){
    return _mm256_sqrt_ps(lhs);
}

ETL_INLINE_VEC_256 minus(__m256 x){
    return _mm256_xor_ps(x, _mm256_set1_ps(-0.f));
}

template<bool Complex = false>
ETL_INLINE_VEC_256 div(__m256 lhs, __m256 rhs){
    return _mm256_div_ps(lhs, rhs);
}

template<bool Complex = false>
ETL_INLINE_VEC_256D div(__m256d lhs, __m256d rhs){
    return _mm256_div_pd(lhs, rhs);
}

template<bool Complex = false>
ETL_INLINE_VEC_256 mul(__m256 lhs, __m256 rhs){
    return _mm256_mul_ps(lhs, rhs);
}

template<>
ETL_INLINE_VEC_256 mul<true>(__m256 lhs, __m256 rhs){
    //lhs = [x1.real, x1.img, x2.real, x2.img, ...]
    //rhs = [y1.real, y1.img, y2.real, y2.img, ...]

    //ymm1 = [y1.real, y1.real, y2.real, y2.real, ...]
    __m256 ymm1 = _mm256_moveldup_ps(rhs);

    //ymm2 = lhs * ymm1
    __m256 ymm2 = _mm256_mul_ps(lhs, ymm1);

    //ymm3 = [x1.img, x1.real, x2.img, x2.real]
    __m256 ymm3 = _mm256_permute_ps(lhs, _MM_SHUFFLE(2, 3, 0, 1));

    //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
    ymm1 = _mm256_movehdup_ps(rhs);

    //ymm4 = ymm3 * ymm1
    __m256 ymm4 = _mm256_mul_ps(ymm3, ymm1);

    //result = [ymm2 -+ ymm4];
    return _mm256_addsub_ps(ymm2, ymm4);
}

template<bool Complex = false>
ETL_INLINE_VEC_256D mul(__m256d lhs, __m256d rhs){
    return _mm256_mul_pd(lhs, rhs);
}

template<>
ETL_INLINE_VEC_256D mul<true>(__m256d lhs, __m256d rhs){
    //lhs = [x1.real, x1.img, x2.real, x2.img]
    //rhs = [y1.real, y1.img, y2.real, y2.img]

    //ymm1 = [y1.real, y1.real, y2.real, y2.real]
    __m256d ymm1 = _mm256_movedup_pd(rhs);

    //ymm2 = lhs * ymm1
    __m256d ymm2 = _mm256_mul_pd(lhs, ymm1);

    //ymm3 = [x1.img, x1.real, x2.img, x2.real]
    __m256d ymm3 = _mm256_permute_pd(lhs, 5);

    //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
    __m256d ymm5 = _mm256_permute_pd(rhs, 5);
    ymm1 = _mm256_movedup_pd(ymm5);

    //ymm4 = ymm3 * ymm1
    __m256d ymm4 = _mm256_mul_pd(ymm3, ymm1);

    //result = [ymm2 -+ ymm4];
    return _mm256_addsub_pd(ymm2, ymm4);
}

#ifdef __INTEL_COMPILER

//Exponential

ETL_INLINE_VEC_256D exp(__m256d x){
    return _mm256_exp_pd(x);
}

ETL_INLINE_VEC_256 exp(__m256 x){
    return _mm256_exp_ps(x);
}

//Logarithm

ETL_INLINE_VEC_256D log(__m256d x){
    return _mm256_log_pd(x);
}

ETL_INLINE_VEC_256 log(__m256 x){
    return _mm256_log_ps(x);
}

//Min

ETL_INLINE_VEC_256D min(__m256d lhs, __m256d rhs){
    return _mm256_min_pd(lhs, rhs);
}

ETL_INLINE_VEC_256 min(__m256 lhs, __m256 rhs){
    return _mm256_min_ps(lhs, rhs);
}

//Max

ETL_INLINE_VEC_256D max(__m256d lhs, __m256d rhs){
    return _mm256_max_pd(lhs, rhs);
}

ETL_INLINE_VEC_256 max(__m256 lhs, __m256 rhs){
    return _mm256_max_ps(lhs, rhs);
}

#endif

} //end of namespace vec

} //end of namespace etl
