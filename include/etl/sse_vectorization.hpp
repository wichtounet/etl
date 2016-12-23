//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file sse_vectorization.hpp
 * \brief Contains SSE vectorized functions for the vectorized assignment of expressions
 */

#pragma once

#ifdef __SSE3__

#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#include "etl/inline.hpp"
#include "etl/sse_exp.hpp"

#ifdef VECT_DEBUG
#include <iostream>
#endif

#ifdef __clang__
#define _mm_undefined_ps _mm_setzero_ps
#define _mm_undefined_pd _mm_setzero_pd
#endif

#define ETL_INLINE_VEC_VOID ETL_STATIC_INLINE(void)
#define ETL_INLINE_VEC_128I ETL_STATIC_INLINE(__m128i)
#define ETL_INLINE_VEC_128 ETL_STATIC_INLINE(__m128)
#define ETL_INLINE_VEC_128D ETL_STATIC_INLINE(__m128d)
#define ETL_OUT_VEC_128I ETL_OUT_INLINE(__m128i)
#define ETL_OUT_VEC_128 ETL_OUT_INLINE(__m128)
#define ETL_OUT_VEC_128D ETL_OUT_INLINE(__m128d)

namespace etl {

/*!
 * \brief Define traits to get vectorization information for types in SSE vector mode.
 */
template <typename T>
struct sse_intrinsic_traits {
    static constexpr bool vectorizable     = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr std::size_t size      = 1;          ///< Numbers of elements done at once
    static constexpr std::size_t alignment = alignof(T); ///< Necessary number of bytes of alignment for this type

    using intrinsic_type = T; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for float
 */
template <>
struct sse_intrinsic_traits<float> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 4; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 16; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m128; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for double
 */
template <>
struct sse_intrinsic_traits<double> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 2; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 16; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m128d; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for std::complex<float>
 */
template <>
struct sse_intrinsic_traits<std::complex<float>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 2; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 16;///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m128; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for std::complex<double>
 */
template <>
struct sse_intrinsic_traits<std::complex<double>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 1; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 16;///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m128d; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for etl::complex<float>
 */
template <>
struct sse_intrinsic_traits<etl::complex<float>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 2; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 16;///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m128; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for etl::complex<double>
 */
template <>
struct sse_intrinsic_traits<etl::complex<double>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 1; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 16;///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m128d; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for float
 */
template <>
struct sse_intrinsic_traits<int> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 4;    ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m128i; ///< The vector type
};

/*!
 * \brief Streaming SIMD (SSE) operations implementation.
 */
struct sse_vec {
    template <typename T>
    using traits = sse_intrinsic_traits<T>; ///< The traits for this vector implementation

    template <typename T>
    using vec_type = typename traits<T>::intrinsic_type; ///< The vector type for the given vector type for this vector implementation

#ifdef VEC_DEBUG

    template <typename T>
    static void debug_d(T value) {
        union test {
            __m128d vec; // a data field, maybe a register, maybe not
            double array[2];
            test(__m128d vec)
                    : vec(vec) {}
        };

        test u_value = value;
        std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << "]" << std::endl;
    }

    template <typename T>
    static void debug_s(T value) {
        union test {
            __m128 vec; // a data field, maybe a register, maybe not
            float array[4];
            test(__m128 vec)
                    : vec(vec) {}
        };

        test u_value = value;
        std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << "," << u_value.array[2] << "," << u_value.array[3] << "]" << std::endl;
    }

#else

    template <typename T>
    static std::string debug_d(T) {
        return "";
    }

    template <typename T>
    static std::string debug_s(T) {
        return "";
    }

#endif

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(int* memory, __m128i value) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(float* memory, __m128 value) {
        _mm_storeu_ps(memory, value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(double* memory, __m128d value) {
        _mm_storeu_pd(memory, value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(std::complex<float>* memory, __m128 value) {
        _mm_storeu_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(std::complex<double>* memory, __m128d value) {
        _mm_storeu_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(etl::complex<float>* memory, __m128 value) {
        _mm_storeu_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(etl::complex<double>* memory, __m128d value) {
        _mm_storeu_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(int* memory, __m128i value) {
        _mm_store_si128(reinterpret_cast<__m128i*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(float* memory, __m128 value) {
        _mm_store_ps(memory, value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(double* memory, __m128d value) {
        _mm_store_pd(memory, value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(std::complex<float>* memory, __m128 value) {
        _mm_store_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(std::complex<double>* memory, __m128d value) {
        _mm_store_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(etl::complex<float>* memory, __m128 value) {
        _mm_store_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(etl::complex<double>* memory, __m128d value) {
        _mm_store_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(int* memory, __m128i value) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(float* memory, __m128 value) {
        _mm_stream_ps(memory, value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(double* memory, __m128d value) {
        _mm_stream_pd(memory, value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(std::complex<float>* memory, __m128 value) {
        _mm_stream_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(std::complex<double>* memory, __m128d value) {
        _mm_stream_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(etl::complex<float>* memory, __m128 value) {
        _mm_stream_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(etl::complex<double>* memory, __m128d value) {
        _mm_stream_pd(reinterpret_cast<double*>(memory), value);
    }

    template<typename T>
    ETL_TMP_INLINE(typename sse_intrinsic_traits<T>::intrinsic_type) zero();

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_128I load(const int* memory) {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_128 load(const float* memory) {
        return _mm_load_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_128D load(const double* memory) {
        return _mm_load_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_128 load(const std::complex<float>* memory) {
        return _mm_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_128D load(const std::complex<double>* memory) {
        return _mm_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_128 load(const etl::complex<float>* memory) {
        return _mm_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_128D load(const etl::complex<double>* memory) {
        return _mm_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_128I loadu(const int* memory) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_128 loadu(const float* memory) {
        return _mm_loadu_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_128D loadu(const double* memory) {
        return _mm_loadu_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_128 loadu(const std::complex<float>* memory) {
        return _mm_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_128D loadu(const std::complex<double>* memory) {
        return _mm_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_128 loadu(const etl::complex<float>* memory) {
        return _mm_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_128D loadu(const etl::complex<double>* memory) {
        return _mm_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_128I set(int value) {
        return _mm_set1_epi32(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_128D set(double value) {
        return _mm_set1_pd(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_128 set(float value) {
        return _mm_set1_ps(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_128 set(std::complex<float> value) {
        std::complex<float> tmp[]{value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_128D set(std::complex<double> value) {
        std::complex<double> tmp[]{value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_128 set(etl::complex<float> value) {
        etl::complex<float> tmp[]{value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_128D set(etl::complex<double> value) {
        etl::complex<double> tmp[]{value};
        return loadu(tmp);
    }

    // Addition

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_INLINE_VEC_128I add(__m128i lhs, __m128i rhs) {
        return _mm_add_epi32(lhs, rhs);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_INLINE_VEC_128 add(__m128 lhs, __m128 rhs) {
        return _mm_add_ps(lhs, rhs);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_INLINE_VEC_128D add(__m128d lhs, __m128d rhs) {
        return _mm_add_pd(lhs, rhs);
    }

    // Subtraction

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_INLINE_VEC_128I sub(__m128i lhs, __m128i rhs) {
        return _mm_sub_epi32(lhs, rhs);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_INLINE_VEC_128D sub(__m128d lhs, __m128d rhs) {
        return _mm_sub_pd(lhs, rhs);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_INLINE_VEC_128 sub(__m128 lhs, __m128 rhs) {
        return _mm_sub_ps(lhs, rhs);
    }

    // Square Root

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_INLINE_VEC_128 sqrt(__m128 x) {
        return _mm_sqrt_ps(x);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_INLINE_VEC_128D sqrt(__m128d x) {
        return _mm_sqrt_pd(x);
    }

    // Negation

    // TODO negation epi32

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_INLINE_VEC_128D minus(__m128d x) {
        return _mm_xor_pd(x, _mm_set1_pd(-0.f));
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_INLINE_VEC_128 minus(__m128 x) {
        return _mm_xor_ps(x, _mm_set1_ps(-0.f));
    }

    // Multiplication

    template <bool Complex = false>
    ETL_INLINE_VEC_128I mul(__m128i lhs, __m128i rhs) {
        return _mm_mullo_epi32(lhs, rhs);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m128) mul(__m128 lhs, __m128 rhs) {
        return _mm_mul_ps(lhs, rhs);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m128d) mul(__m128d lhs, __m128d rhs) {
        return _mm_mul_pd(lhs, rhs);
    }

    // Fused-Multiply-Add (FMA)

    ETL_INLINE_VEC_128I fmadd(__m128i a, __m128i b, __m128i c){
        return add(mul(a, b), c);
    }

    /*!
     * \brief Fused-Multiply-Add of a b and c (r = (a * b) + c)
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m128) fmadd(__m128 a, __m128 b, __m128 c);

    /*!
     * \brief Fused-Multiply-Add of a b and c (r = (a * b) + c)
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m128d) fmadd(__m128d a, __m128d b, __m128d c);

    // Division

    /*!
     * \brief Divide the two given vectors
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m128) div(__m128 lhs, __m128 rhs) {
        return _mm_div_ps(lhs, rhs);
    }

    /*!
     * \brief Divide the two given vectors
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m128d) div(__m128d lhs, __m128d rhs) {
        return _mm_div_pd(lhs, rhs);
    }

    // Cosinus

    /*!
     * \brief Compute the cosinus of each element of the given vector
     */
    ETL_INLINE_VEC_128 cos(__m128 x) {
        return etl::cos_ps(x);
    }

    // Sinus

    /*!
     * \brief Compute the sinus of each element of the given vector
     */
    ETL_INLINE_VEC_128 sin(__m128 x) {
        return etl::sin_ps(x);
    }

//The Intel C++ Compiler (icc) has more intrinsics.
//ETL uses them when compiled with icc

#ifndef __INTEL_COMPILER

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_128 exp(__m128 x) {
        return etl::exp_ps(x);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_128D exp(__m128d x) {
        return etl::exp_pd(x);
    }

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_INLINE_VEC_128 log(__m128 x) {
        return etl::log_ps(x);
    }

#else //__INTEL_COMPILER

    //Exponential

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_128D exp(__m128d x) {
        return _mm_exp_pd(x);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_128 exp(__m128 x) {
        return _mm_exp_ps(x);
    }

    //Logarithm

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_INLINE_VEC_128D log(__m128d x) {
        return _mm_log_pd(x);
    }

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_INLINE_VEC_128 log(__m128 x) {
        return _mm_log_ps(x);
    }

#endif //__INTEL_COMPILER

    //Min

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_128D min(__m128d lhs, __m128d rhs) {
        return _mm_min_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_128 min(__m128 lhs, __m128 rhs) {
        return _mm_min_ps(lhs, rhs);
    }

    //Max

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_128D max(__m128d lhs, __m128d rhs) {
        return _mm_max_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_128 max(__m128 lhs, __m128 rhs) {
        return _mm_max_ps(lhs, rhs);
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template <typename T = float>
    static inline T ETL_INLINE_ATTR_VEC hadd(__m128 in) {
        __m128 shuf = _mm_movehdup_ps(in);
        __m128 sums = _mm_add_ps(in, shuf);
        shuf        = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template <typename T = double>
    static inline T ETL_INLINE_ATTR_VEC hadd(__m128d in) {
        __m128 undef   = _mm_undefined_ps();
        __m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(in));
        __m128d shuf = _mm_castps_pd(shuftmp);
        return _mm_cvtsd_f64(_mm_add_sd(in, shuf));
    }
};

//TODO Vectorize the two following functions

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
template <>
inline std::complex<float> ETL_INLINE_ATTR_VEC sse_vec::hadd<std::complex<float>>(__m128 in) {
    std::complex<float> tmp_result[2];
    sse_vec::storeu(tmp_result, in);
    return tmp_result[0] + tmp_result[1];
}

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
template <>
inline std::complex<double> ETL_INLINE_ATTR_VEC sse_vec::hadd<std::complex<double>>(__m128d in) {
    std::complex<double> tmp_result[1];
    sse_vec::storeu(tmp_result, in);
    return tmp_result[0];
}

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
template <>
inline etl::complex<float> ETL_INLINE_ATTR_VEC sse_vec::hadd<etl::complex<float>>(__m128 in) {
    etl::complex<float> tmp_result[2];
    sse_vec::storeu(tmp_result, in);
    return tmp_result[0] + tmp_result[1];
}

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
template <>
inline etl::complex<double> ETL_INLINE_ATTR_VEC sse_vec::hadd<etl::complex<double>>(__m128d in) {
    etl::complex<double> tmp_result[1];
    sse_vec::storeu(tmp_result, in);
    return tmp_result[0];
}

/*!
 * \copydoc sse_vec::mul
 */
template <>
ETL_OUT_VEC_128 sse_vec::mul<true>(__m128 lhs, __m128 rhs) {
    //lhs = [x1.real, x1.img, x2.real, x2.img]
    //rhs = [y1.real, y1.img, y2.real, y2.img]

    //ymm1 = [y1.real, y1.real, y2.real, y2.real]
    __m128 ymm1 = _mm_moveldup_ps(rhs);

    //ymm2 = lhs * ymm1
    __m128 ymm2 = _mm_mul_ps(lhs, ymm1);

    //ymm3 = [x1.img, x1.real, x2.img, x2.real]
    __m128 ymm3 = _mm_shuffle_ps(lhs, lhs, _MM_SHUFFLE(2, 3, 0, 1));

    //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
    ymm1 = _mm_movehdup_ps(rhs);

    //ymm4 = ymm3 * ymm1
    __m128 ymm4 = _mm_mul_ps(ymm3, ymm1);

    //result = [ymm2 -+ ymm4];
    return _mm_addsub_ps(ymm2, ymm4);
}

/*!
 * \copydoc sse_vec::mul
 */
template <>
ETL_OUT_VEC_128D sse_vec::mul<true>(__m128d lhs, __m128d rhs) {
    //lhs = [x.real, x.img]
    //rhs = [y.real, y.img]

    //ymm1 = [y.real, y.real]
    __m128d ymm1 = _mm_movedup_pd(rhs);

    //ymm2 = [x.real * y.real, x.img * y.real]
    __m128d ymm2 = _mm_mul_pd(lhs, ymm1);

    //ymm1 = [x.img, x.real]
    ymm1 = _mm_shuffle_pd(lhs, lhs, _MM_SHUFFLE2(0, 1));

    //ymm3 =  [y.img, y.img]
    __m128d ymm3 = _mm_shuffle_pd(rhs, rhs, _MM_SHUFFLE2(1, 1));

    //ymm4 = [x.img * y.img, x.real * y.img]
    __m128d ymm4 = _mm_mul_pd(ymm1, ymm3);

    //result = [x.real * y.real - x.img * y.img, x.img * y.real - x.real * y.img]
    return _mm_addsub_pd(ymm2, ymm4);
}

/*!
 * \copydoc sse_vec::fmadd
 */
template <>
ETL_OUT_VEC_128 sse_vec::fmadd<false>(__m128 a, __m128 b, __m128 c) {
#ifdef __FMA__
    return _mm_fmadd_ps(a, b, c);
#else
    return add(mul<false>(a, b), c);
#endif
}

/*!
 * \copydoc sse_vec::fmadd
 */
template <>
ETL_OUT_VEC_128D sse_vec::fmadd<false>(__m128d a, __m128d b, __m128d c) {
#ifdef __FMA__
    return _mm_fmadd_pd(a, b, c);
#else
    return add(mul<false>(a, b), c);
#endif
}

/*!
 * \copydoc sse_vec::fmadd
 */
template <>
ETL_OUT_VEC_128 sse_vec::fmadd<true>(__m128 a, __m128 b, __m128 c) {
    return add(mul<true>(a, b), c);
}

/*!
 * \copydoc sse_vec::fmadd
 */
template <>
ETL_OUT_VEC_128D sse_vec::fmadd<true>(__m128d a, __m128d b, __m128d c) {
    return add(mul<true>(a, b), c);
}

/*!
 * \copydoc sse_vec::div
 */
template <>
ETL_OUT_VEC_128 sse_vec::div<true>(__m128 lhs, __m128 rhs) {
    //lhs = [x1.real, x1.img, x2.real, x2.img]
    //rhs = [y1.real, y1.img, y2.real, y2.img]

    //ymm0 = [y1.real, y1.real, y2.real, y2.real]
    __m128 ymm0 = _mm_moveldup_ps(rhs);

    //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
    __m128 ymm1 = _mm_movehdup_ps(rhs);

    //ymm2 = [x.real * y.real, x.img * y.real, ...]
    __m128 ymm2 = _mm_mul_ps(lhs, ymm0);

    //ymm3 = [x1.img, x1.real, x2.img, x2.real]
    __m128 ymm3 = _mm_shuffle_ps(lhs, lhs, _MM_SHUFFLE(2, 3, 0, 1));

    //ymm4 = [x.img * y.img, x.real * y.img, ...]
    __m128 ymm4 = _mm_mul_ps(ymm3, ymm1);

    //ymm4 = subadd(ymm2, ymm4)
    ymm3 = _mm_sub_ps(_mm_set1_ps(0.0), ymm4);
    ymm4 = _mm_addsub_ps(ymm2, ymm3);

    //ymm2 = [y.real^2, y.real^2]
    ymm2 = _mm_mul_ps(ymm0, ymm0);

    //ymm3 = [y.imag^2, y.imag^2]
    ymm3 = _mm_mul_ps(ymm1, ymm1);

    //ymm0 = [y.real^2 + y.imag^2, y.real^2 + y.imag^2]
    ymm0 = _mm_add_ps(ymm2, ymm3);

    //result = ymm4 / ymm0
    return _mm_div_ps(ymm4, ymm0);
}

/*!
 * \copydoc sse_vec::div
 */
template <>
ETL_OUT_VEC_128D sse_vec::div<true>(__m128d lhs, __m128d rhs) {
    //lhs = [x.real, x.img]
    //rhs = [y.real, y.img]

    //ymm0 = [y.real, y.real]
    __m128d ymm0 = _mm_movedup_pd(rhs);

    //ymm1 =  [y.img, y.img]
    __m128d ymm1 = _mm_shuffle_pd(rhs, rhs, _MM_SHUFFLE2(1, 1));

    //ymm2 = [x.real * y.real, x.img * y.real]
    __m128d ymm2 = _mm_mul_pd(lhs, ymm0);

    //ymm3 = [x.img, x.real]
    __m128d ymm3 = _mm_shuffle_pd(lhs, lhs, _MM_SHUFFLE2(0, 1));

    //ymm4 = [x.img * y.img, x.real * y.img]
    __m128d ymm4 = _mm_mul_pd(ymm3, ymm1);

    //ymm4 = subadd(ymm2, ymm4)
    ymm3 = _mm_sub_pd(_mm_set1_pd(0.0), ymm4);
    ymm4 = _mm_addsub_pd(ymm2, ymm3);

    //ymm2 = [y.real^2, y.real^2]
    ymm2 = _mm_mul_pd(ymm0, ymm0);

    //ymm3 = [y.imag^2, y.imag^2]
    ymm3 = _mm_mul_pd(ymm1, ymm1);

    //ymm0 = [y.real^2 + y.imag^2, y.real^2 + y.imag^2]
    ymm0 = _mm_add_pd(ymm2, ymm3);

    //result = ymm4 / ymm0
    return _mm_div_pd(ymm4, ymm0);
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_128I sse_vec::zero<int>() {
    return _mm_setzero_si128();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_128 sse_vec::zero<float>() {
    return _mm_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_128D sse_vec::zero<double>() {
    return _mm_setzero_pd();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_128 sse_vec::zero<etl::complex<float>>() {
    return _mm_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_128D sse_vec::zero<etl::complex<double>>() {
    return _mm_setzero_pd();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_128 sse_vec::zero<std::complex<float>>() {
    return _mm_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_128D sse_vec::zero<std::complex<double>>() {
    return _mm_setzero_pd();
}

} //end of namespace etl

#endif //__SSE3__
