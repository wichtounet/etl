//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

namespace etl {

/*!
 * \brief SSE SIMD float type
 */
using sse_simd_float = simd_pack<vector_mode_t::SSE3, float, __m128>;

/*!
 * \brief SSE SIMD double type
 */
using sse_simd_double = simd_pack<vector_mode_t::SSE3, double, __m128d>;

/*!
 * \brief SSE SIMD complex float type
 */
template <typename T>
using sse_simd_complex_float = simd_pack<vector_mode_t::SSE3, T, __m128>;

/*!
 * \brief SSE SIMD complex double type
 */
template <typename T>
using sse_simd_complex_double = simd_pack<vector_mode_t::SSE3, T, __m128d>;

/*!
 * \brief SSE SIMD byte type
 */
using sse_simd_byte = simd_pack<vector_mode_t::SSE3, int8_t, __m128i>;

/*!
 * \brief SSE SIMD short type
 */
using sse_simd_short = simd_pack<vector_mode_t::SSE3, int16_t, __m128i>;

/*!
 * \brief SSE SIMD int type
 */
using sse_simd_int = simd_pack<vector_mode_t::SSE3, int32_t, __m128i>;

/*!
 * \brief SSE SIMD long type
 */
using sse_simd_long = simd_pack<vector_mode_t::SSE3, int64_t, __m128i>;

/*!
 * \brief Define traits to get vectorization information for types in SSE vector mode.
 */
template <typename T>
struct sse_intrinsic_traits {
    static constexpr bool vectorizable = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr size_t size       = 1;          ///< Numbers of elements done at once
    static constexpr size_t alignment  = alignof(T); ///< Necessary number of bytes of alignment for this type

    using intrinsic_type = T; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for float
 */
template <>
struct sse_intrinsic_traits<float> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 4;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_float; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for double
 */
template <>
struct sse_intrinsic_traits<double> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 2;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_double; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for std::complex<float>
 */
template <>
struct sse_intrinsic_traits<std::complex<float>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 2;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_complex_float<std::complex<float>>; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for std::complex<double>
 */
template <>
struct sse_intrinsic_traits<std::complex<double>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 1;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_complex_double<std::complex<double>>; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for etl::complex<float>
 */
template <>
struct sse_intrinsic_traits<etl::complex<float>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 2;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_complex_float<etl::complex<float>>; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for etl::complex<double>
 */
template <>
struct sse_intrinsic_traits<etl::complex<double>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 1;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_complex_double<etl::complex<double>>; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for int32_t
 */
template <>
struct sse_intrinsic_traits<int8_t> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 16;   ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_byte; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for int32_t
 */
template <>
struct sse_intrinsic_traits<int16_t> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 8;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_short; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for int32_t
 */
template <>
struct sse_intrinsic_traits<int32_t> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 4;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_int; ///< The vector type
};

/*!
 * \brief specialization of sse_intrinsic_traits for int64_t
 */
template <>
struct sse_intrinsic_traits<int64_t> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 2;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 16;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = sse_simd_long; ///< The vector type
};

/*!
 * \brief Streaming SIMD (SSE) operations implementation.
 */
struct sse_vec {
    /*!
     * \brief The traits for this vector implementation
     */
    template <typename T>
    using traits = sse_intrinsic_traits<T>;

    /*!
     * \brief The vector type for the given type for this vector implementation
     */
    template <typename T>
    using vec_type = typename traits<T>::intrinsic_type;

#ifdef VEC_DEBUG

    /*!
     * \brief Print the value of a SSE vector of double
     */
    template <typename T>
    static void debug_d(T value) {
        union test {
            __m128d vec; // a data field, maybe a register, maybe not
            double array[2];
            test(__m128d vec) : vec(vec) {}
        };

        test u_value = value;
        std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << "]" << std::endl;
    }

    /*!
     * \brief Print the value of a SSE vector of float
     */
    template <typename T>
    static void debug_s(T value) {
        union test {
            __m128 vec; // a data field, maybe a register, maybe not
            float array[4];
            test(__m128 vec) : vec(vec) {}
        };

        test u_value = value;
        std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << "," << u_value.array[2] << "," << u_value.array[3] << "]" << std::endl;
    }

#else

    /*!
     * \brief Print the value of a SSE vector of double
     */
    template <typename T>
    static std::string debug_d(T) {
        return "";
    }

    /*!
     * \brief Print the value of a SSE vector of float
     */
    template <typename T>
    static std::string debug_s(T) {
        return "";
    }

#endif

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(int8_t* memory, sse_simd_byte value) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(int16_t* memory, sse_simd_short value) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(int32_t* memory, sse_simd_int value) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(int64_t* memory, sse_simd_long value) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(float* memory, sse_simd_float value) {
        _mm_storeu_ps(memory, value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(double* memory, sse_simd_double value) {
        _mm_storeu_pd(memory, value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(std::complex<float>* memory, sse_simd_complex_float<std::complex<float>> value) {
        _mm_storeu_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(std::complex<double>* memory, sse_simd_complex_double<std::complex<double>> value) {
        _mm_storeu_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(etl::complex<float>* memory, sse_simd_complex_float<etl::complex<float>> value) {
        _mm_storeu_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(etl::complex<double>* memory, sse_simd_complex_double<etl::complex<double>> value) {
        _mm_storeu_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int8_t* memory, sse_simd_byte value) {
        _mm_store_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int16_t* memory, sse_simd_short value) {
        _mm_store_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int32_t* memory, sse_simd_int value) {
        _mm_store_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int64_t* memory, sse_simd_long value) {
        _mm_store_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(float* memory, sse_simd_float value) {
        _mm_store_ps(memory, value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(double* memory, sse_simd_double value) {
        _mm_store_pd(memory, value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(std::complex<float>* memory, sse_simd_complex_float<std::complex<float>> value) {
        _mm_store_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(std::complex<double>* memory, sse_simd_complex_double<std::complex<double>> value) {
        _mm_store_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(etl::complex<float>* memory, sse_simd_complex_float<etl::complex<float>> value) {
        _mm_store_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(etl::complex<double>* memory, sse_simd_complex_double<etl::complex<double>> value) {
        _mm_store_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int8_t* memory, sse_simd_byte value) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int16_t* memory, sse_simd_short value) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int32_t* memory, sse_simd_int value) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int64_t* memory, sse_simd_long value) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(float* memory, sse_simd_float value) {
        _mm_stream_ps(memory, value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(double* memory, sse_simd_double value) {
        _mm_stream_pd(memory, value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(std::complex<float>* memory, sse_simd_complex_float<std::complex<float>> value) {
        _mm_stream_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(std::complex<double>* memory, sse_simd_complex_double<std::complex<double>> value) {
        _mm_stream_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(etl::complex<float>* memory, sse_simd_complex_float<etl::complex<float>> value) {
        _mm_stream_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(etl::complex<double>* memory, sse_simd_complex_double<etl::complex<double>> value) {
        _mm_stream_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Create a vector filled of zero for the given type
     */
    template <typename T>
    ETL_TMP_INLINE(typename sse_intrinsic_traits<T>::intrinsic_type)
    zero();

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_byte) load(const int8_t* memory) {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_short) load(const int16_t* memory) {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_int) load(const int32_t* memory) {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_long) load(const int64_t* memory) {
        return _mm_load_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_float) load(const float* memory) {
        return _mm_load_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_double) load(const double* memory) {
        return _mm_load_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_complex_float<std::complex<float>>) load(const std::complex<float>* memory) {
        return _mm_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_complex_double<std::complex<double>>) load(const std::complex<double>* memory) {
        return _mm_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_complex_float<etl::complex<float>>) load(const etl::complex<float>* memory) {
        return _mm_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_complex_double<etl::complex<double>>) load(const etl::complex<double>* memory) {
        return _mm_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_byte) loadu(const int8_t* memory) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_short) loadu(const int16_t* memory) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_int) loadu(const int32_t* memory) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_long) loadu(const int64_t* memory) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_float) loadu(const float* memory) {
        return _mm_loadu_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_double) loadu(const double* memory) {
        return _mm_loadu_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_complex_float<std::complex<float>>) loadu(const std::complex<float>* memory) {
        return _mm_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_complex_double<std::complex<double>>) loadu(const std::complex<double>* memory) {
        return _mm_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_complex_float<etl::complex<float>>) loadu(const etl::complex<float>* memory) {
        return _mm_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(sse_simd_complex_double<etl::complex<double>>) loadu(const etl::complex<double>* memory) {
        return _mm_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_byte) set(int8_t value) {
        return _mm_set1_epi8(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_short) set(int16_t value) {
        return _mm_set1_epi16(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_int) set(int32_t value) {
        return _mm_set1_epi32(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_long) set(int64_t value) {
        return _mm_set1_epi64x(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_double) set(double value) {
        return _mm_set1_pd(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_float) set(float value) {
        return _mm_set1_ps(value);
    }

    /*!
     * \brief Round up each values of the vector and return them
     */
    ETL_STATIC_INLINE(sse_simd_float) round_up(sse_simd_float x) {
        return _mm_round_ps(x.value, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
    }

    /*!
     * \brief Round up each values of the vector and return them
     */
    ETL_STATIC_INLINE(sse_simd_double) round_up(sse_simd_double x) {
        return _mm_round_pd(x.value, (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC));
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_complex_float<std::complex<float>>) set(std::complex<float> value) {
        std::complex<float> tmp[]{value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_complex_double<std::complex<double>>) set(std::complex<double> value) {
        std::complex<double> tmp[]{value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_complex_float<etl::complex<float>>) set(etl::complex<float> value) {
        etl::complex<float> tmp[]{value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(sse_simd_complex_double<etl::complex<double>>) set(etl::complex<double> value) {
        etl::complex<double> tmp[]{value};
        return loadu(tmp);
    }

    // Addition

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_byte) add(sse_simd_byte lhs, sse_simd_byte rhs) {
        return _mm_add_epi8(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_short) add(sse_simd_short lhs, sse_simd_short rhs) {
        return _mm_add_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_int) add(sse_simd_int lhs, sse_simd_int rhs) {
        return _mm_add_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_long) add(sse_simd_long lhs, sse_simd_long rhs) {
        return _mm_add_epi64(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_float) add(sse_simd_float lhs, sse_simd_float rhs) {
        return _mm_add_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_double) add(sse_simd_double lhs, sse_simd_double rhs) {
        return _mm_add_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_float<T>)
    add(sse_simd_complex_float<T> lhs, sse_simd_complex_float<T> rhs) {
        return _mm_add_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_double<T>)
    add(sse_simd_complex_double<T> lhs, sse_simd_complex_double<T> rhs) {
        return _mm_add_pd(lhs.value, rhs.value);
    }

    // Subtraction

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_byte) sub(sse_simd_byte lhs, sse_simd_byte rhs) {
        return _mm_sub_epi8(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_short) sub(sse_simd_short lhs, sse_simd_short rhs) {
        return _mm_sub_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_int) sub(sse_simd_int lhs, sse_simd_int rhs) {
        return _mm_sub_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_long) sub(sse_simd_long lhs, sse_simd_long rhs) {
        return _mm_sub_epi64(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_float) sub(sse_simd_float lhs, sse_simd_float rhs) {
        return _mm_sub_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(sse_simd_double) sub(sse_simd_double lhs, sse_simd_double rhs) {
        return _mm_sub_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_float<T>)
    sub(sse_simd_complex_float<T> lhs, sse_simd_complex_float<T> rhs) {
        return _mm_sub_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_double<T>)
    sub(sse_simd_complex_double<T> lhs, sse_simd_complex_double<T> rhs) {
        return _mm_sub_pd(lhs.value, rhs.value);
    }

    // Square Root

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_STATIC_INLINE(sse_simd_float) sqrt(sse_simd_float x) {
        return _mm_sqrt_ps(x.value);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_STATIC_INLINE(sse_simd_double) sqrt(sse_simd_double x) {
        return _mm_sqrt_pd(x.value);
    }

    // Negation

    // TODO negation epi32

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_STATIC_INLINE(sse_simd_float) minus(sse_simd_float x) {
        return _mm_xor_ps(x.value, _mm_set1_ps(-0.f));
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_STATIC_INLINE(sse_simd_double) minus(sse_simd_double x) {
        return _mm_xor_pd(x.value, _mm_set1_pd(-0.));
    }

    // Multiplication

    /*!
     * \brief Multiply the two given vectors of byte
     */
    ETL_STATIC_INLINE(sse_simd_byte) mul(sse_simd_byte lhs, sse_simd_byte rhs) {
        __m128i even = _mm_mullo_epi16(lhs.value, rhs.value);
        __m128i odd  = _mm_mullo_epi16(_mm_srli_epi16(lhs.value, 8), _mm_srli_epi16(rhs.value, 8));
        return _mm_or_si128(_mm_slli_epi16(odd, 8), _mm_srli_epi16(_mm_slli_epi16(even, 8), 8));
    }

    /*!
     * \brief Multiply the two given vectors of short
     */
    ETL_STATIC_INLINE(sse_simd_short) mul(sse_simd_short lhs, sse_simd_short rhs) {
        return _mm_mullo_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors of int
     */
    ETL_STATIC_INLINE(sse_simd_int) mul(sse_simd_int lhs, sse_simd_int rhs) {
        return _mm_mullo_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors of long
     */
    ETL_STATIC_INLINE(sse_simd_long) mul(sse_simd_long lhs, sse_simd_long rhs) {
        int64_t result[2];
        result[0] = lhs[0] * rhs[0];
        result[1] = lhs[1] * rhs[1];
        return loadu(&result[0]);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    ETL_STATIC_INLINE(sse_simd_float) mul(sse_simd_float lhs, sse_simd_float rhs) {
        return _mm_mul_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    ETL_STATIC_INLINE(sse_simd_double) mul(sse_simd_double lhs, sse_simd_double rhs) {
        return _mm_mul_pd(lhs.value, rhs.value);
    }

    /*!
     * \copydoc sse_vec::mul
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_float<T>)
    mul(sse_simd_complex_float<T> lhs, sse_simd_complex_float<T> rhs) {
        //lhs = [x1.real, x1.img, x2.real, x2.img]
        //rhs = [y1.real, y1.img, y2.real, y2.img]

        //ymm1 = [y1.real, y1.real, y2.real, y2.real]
        __m128 ymm1 = _mm_moveldup_ps(rhs.value);

        //ymm2 = lhs * ymm1
        __m128 ymm2 = _mm_mul_ps(lhs.value, ymm1);

        //ymm3 = [x1.img, x1.real, x2.img, x2.real]
        __m128 ymm3 = _mm_shuffle_ps(lhs.value, lhs.value, _MM_SHUFFLE(2, 3, 0, 1));

        //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
        ymm1 = _mm_movehdup_ps(rhs.value);

        //ymm4 = ymm3 * ymm1
        __m128 ymm4 = _mm_mul_ps(ymm3, ymm1);

        //result = [ymm2 -+ ymm4];
        return _mm_addsub_ps(ymm2, ymm4);
    }

    /*!
     * \copydoc sse_vec::mul
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_double<T>)
    mul(sse_simd_complex_double<T> lhs, sse_simd_complex_double<T> rhs) {
        //lhs = [x.real, x.img]
        //rhs = [y.real, y.img]

        //ymm1 = [y.real, y.real]
        __m128d ymm1 = _mm_movedup_pd(rhs.value);

        //ymm2 = [x.real * y.real, x.img * y.real]
        __m128d ymm2 = _mm_mul_pd(lhs.value, ymm1);

        //ymm1 = [x.img, x.real]
        ymm1 = _mm_shuffle_pd(lhs.value, lhs.value, _MM_SHUFFLE2(0, 1));

        //ymm3 =  [y.img, y.img]
        __m128d ymm3 = _mm_shuffle_pd(rhs.value, rhs.value, _MM_SHUFFLE2(1, 1));

        //ymm4 = [x.img * y.img, x.real * y.img]
        __m128d ymm4 = _mm_mul_pd(ymm1, ymm3);

        //result = [x.real * y.real - x.img * y.img, x.img * y.real - x.real * y.img]
        return _mm_addsub_pd(ymm2, ymm4);
    }

    // Fused-Multiply-Add (FMA)

    /*!
     * \brief Fused-Multiply Add of the three given vector of bytes
     */
    ETL_STATIC_INLINE(sse_simd_byte) fmadd(sse_simd_byte a, sse_simd_byte b, sse_simd_byte c) {
        return add(mul(a, b), c);
    }

    /*!
     * \brief Fused-Multiply Add of the three given vector of short
     */
    ETL_STATIC_INLINE(sse_simd_short) fmadd(sse_simd_short a, sse_simd_short b, sse_simd_short c) {
        return add(mul(a, b), c);
    }

    /*!
     * \brief Fused-Multiply Add of the three given vector of int
     */
    ETL_STATIC_INLINE(sse_simd_int) fmadd(sse_simd_int a, sse_simd_int b, sse_simd_int c) {
        return add(mul(a, b), c);
    }

    /*!
     * \brief Fused-Multiply Add of the three given vector of long
     */
    ETL_STATIC_INLINE(sse_simd_long) fmadd(sse_simd_long a, sse_simd_long b, sse_simd_long c) {
        return add(mul(a, b), c);
    }

    /*!
     * \copydoc sse_vec::fmadd
     */
    ETL_STATIC_INLINE(sse_simd_float) fmadd(sse_simd_float a, sse_simd_float b, sse_simd_float c) {
#ifdef __FMA__
        return _mm_fmadd_ps(a.value, b.value, c.value);
#else
        return add(mul(a, b), c);
#endif
    }

    /*!
     * \copydoc sse_vec::fmadd
     */
    ETL_STATIC_INLINE(sse_simd_double) fmadd(sse_simd_double a, sse_simd_double b, sse_simd_double c) {
#ifdef __FMA__
        return _mm_fmadd_pd(a.value, b.value, c.value);
#else
        return add(mul(a, b), c);
#endif
    }

    /*!
     * \copydoc sse_vec::fmadd
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_float<T>)
    fmadd(sse_simd_complex_float<T> a, sse_simd_complex_float<T> b, sse_simd_complex_float<T> c) {
        return add(mul(a, b), c);
    }

    /*!
     * \copydoc sse_vec::fmadd
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_double<T>)
    fmadd(sse_simd_complex_double<T> a, sse_simd_complex_double<T> b, sse_simd_complex_double<T> c) {
        return add(mul(a, b), c);
    }

    // Division

    /*!
     * \brief Divide the two given vectors
     */
    ETL_STATIC_INLINE(sse_simd_float) div(sse_simd_float lhs, sse_simd_float rhs) {
        return _mm_div_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Divide the two given vectors
     */
    ETL_STATIC_INLINE(sse_simd_double) div(sse_simd_double lhs, sse_simd_double rhs) {
        return _mm_div_pd(lhs.value, rhs.value);
    }

    /*!
     * \copydoc sse_vec::div
     */
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_float<T>)
    div(sse_simd_complex_float<T> lhs, sse_simd_complex_float<T> rhs) {
        //lhs = [x1.real, x1.img, x2.real, x2.img]
        //rhs = [y1.real, y1.img, y2.real, y2.img]

        //ymm0 = [y1.real, y1.real, y2.real, y2.real]
        __m128 ymm0 = _mm_moveldup_ps(rhs.value);

        //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
        __m128 ymm1 = _mm_movehdup_ps(rhs.value);

        //ymm2 = [x.real * y.real, x.img * y.real, ...]
        __m128 ymm2 = _mm_mul_ps(lhs.value, ymm0);

        //ymm3 = [x1.img, x1.real, x2.img, x2.real]
        __m128 ymm3 = _mm_shuffle_ps(lhs.value, lhs.value, _MM_SHUFFLE(2, 3, 0, 1));

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
    template <typename T>
    ETL_STATIC_INLINE(sse_simd_complex_double<T>)
    div(sse_simd_complex_double<T> lhs, sse_simd_complex_double<T> rhs) {
        //lhs = [x.real, x.img]
        //rhs = [y.real, y.img]

        //ymm0 = [y.real, y.real]
        __m128d ymm0 = _mm_movedup_pd(rhs.value);

        //ymm1 =  [y.img, y.img]
        __m128d ymm1 = _mm_shuffle_pd(rhs.value, rhs.value, _MM_SHUFFLE2(1, 1));

        //ymm2 = [x.real * y.real, x.img * y.real]
        __m128d ymm2 = _mm_mul_pd(lhs.value, ymm0);

        //ymm3 = [x.img, x.real]
        __m128d ymm3 = _mm_shuffle_pd(lhs.value, lhs.value, _MM_SHUFFLE2(0, 1));

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

    // Cosinus

    /*!
     * \brief Compute the cosinus of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_float) cos(sse_simd_float x) {
        return etl::cos_ps(x.value);
    }

    // Sinus

    /*!
     * \brief Compute the sinus of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_float) sin(sse_simd_float x) {
        return etl::sin_ps(x.value);
    }

        //The Intel C++ Compiler (icc) has more intrinsics.
        //ETL uses them when compiled with icc

#ifndef __INTEL_COMPILER

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_float) exp(sse_simd_float x) {
        return etl::exp_ps(x.value);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_double) exp(sse_simd_double x) {
        return etl::exp_pd(x.value);
    }

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_float) log(sse_simd_float x) {
        return etl::log_ps(x.value);
    }

#else //__INTEL_COMPILER

    //Exponential

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_double) exp(sse_simd_double x) {
        return _mm_exp_pd(x.value);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_float) exp(sse_simd_float x) {
        return _mm_exp_ps(x.value);
    }

    //Logarithm

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_double) log(sse_simd_double x) {
        return _mm_log_pd(x.value);
    }

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_STATIC_INLINE(sse_simd_float) log(sse_simd_float x) {
        return _mm_log_ps(x.value);
    }

#endif //__INTEL_COMPILER

    //Min

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(sse_simd_double) min(sse_simd_double lhs, sse_simd_double rhs) {
        return _mm_min_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(sse_simd_float) min(sse_simd_float lhs, sse_simd_float rhs) {
        return _mm_min_ps(lhs.value, rhs.value);
    }

    //Max

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(sse_simd_double) max(sse_simd_double lhs, sse_simd_double rhs) {
        return _mm_max_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(sse_simd_float) max(sse_simd_float lhs, sse_simd_float rhs) {
        return _mm_max_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(float) hadd(sse_simd_float in) {
        __m128 shuf = _mm_movehdup_ps(in.value);
        __m128 sums = _mm_add_ps(in.value, shuf);
        shuf        = _mm_movehl_ps(shuf, sums);
        sums        = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(double) hadd(sse_simd_double in) {
        __m128 undef   = _mm_undefined_ps();
        __m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(in.value));
        __m128d shuf   = _mm_castps_pd(shuftmp);
        return _mm_cvtsd_f64(_mm_add_sd(in.value, shuf));
    }

    //TODO Vectorize the following functions

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int8_t) hadd(sse_simd_byte in) {
        return in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7] + in[8] + in[9] + in[10] + in[11] + in[12] + in[13] + in[14] + in[15];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int16_t) hadd(sse_simd_short in) {
        return in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int32_t) hadd(sse_simd_int in) {
        return in[0] + in[1] + in[2] + in[3];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int64_t) hadd(sse_simd_long in) {
        return in[0] + in[1];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template <typename T>
    ETL_STATIC_INLINE(T)
    hadd(sse_simd_complex_float<T> in) {
        return in[0] + in[1];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template <typename T>
    ETL_STATIC_INLINE(T)
    hadd(sse_simd_complex_double<T> in) {
        return in[0];
    }
};

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_byte)
sse_vec::zero<int8_t>() {
    return _mm_setzero_si128();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_short)
sse_vec::zero<int16_t>() {
    return _mm_setzero_si128();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_int)
sse_vec::zero<int32_t>() {
    return _mm_setzero_si128();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_long)
sse_vec::zero<int64_t>() {
    return _mm_setzero_si128();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_float)
sse_vec::zero<float>() {
    return _mm_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_double)
sse_vec::zero<double>() {
    return _mm_setzero_pd();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_complex_float<etl::complex<float>>)
sse_vec::zero<etl::complex<float>>() {
    return _mm_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_complex_double<etl::complex<double>>)
sse_vec::zero<etl::complex<double>>() {
    return _mm_setzero_pd();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_complex_float<std::complex<float>>)
sse_vec::zero<std::complex<float>>() {
    return _mm_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template <>
ETL_OUT_INLINE(sse_simd_complex_double<std::complex<double>>)
sse_vec::zero<std::complex<double>>() {
    return _mm_setzero_pd();
}

} //end of namespace etl

#endif //__SSE3__
