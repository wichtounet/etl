//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

#ifdef __AVX512F__

#include <immintrin.h>

#include "etl/inline.hpp"

#ifdef VECT_DEBUG
#include <iostream>
#endif

#define ETL_INLINE_VEC_VOID ETL_STATIC_INLINE(void)
#define ETL_INLINE_VEC_512 ETL_STATIC_INLINE(__m512)
#define ETL_INLINE_VEC_512D ETL_STATIC_INLINE(__m512d)
#define ETL_OUT_VEC_2512ETL_OUT_INLINE(__m512)
#define ETL_OUT_VEC_512D ETL_OUT_INLINE(__m512d)

namespace etl {

/*!
 * \brief AVX-512 SIMD float type
 */
using avx_512_simd_float = simd_pack<vector_mode_t::AVX512, float, __m512>;

/*!
 * \brief AVX-512 SIMD double type
 */
using avx_512_simd_double = simd_pack<vector_mode_t::AVX512, double, __m512d>;

/*!
 * \brief AVX-512 SIMD complex float type
 */
template <typename T>
using avx_512_simd_complex_float = simd_pack<vector_mode_t::AVX512, T, __m512>;

/*!
 * \brief AVX-512 SIMD complex double type
 */
template <typename T>
using avx_512_simd_complex_double = simd_pack<vector_mode_t::AVX512, T, __m512d>;

/*!
 * \brief AVX-512 SIMD byte type
 */
using avx_512_simd_byte = simd_pack<vector_mode_t::AVX512, int8_t, __m512i>;

/*!
 * \brief AVX-512 SIMD short type
 */
using avx_512_simd_short = simd_pack<vector_mode_t::AVX512, int16_t, __m512i>;

/*!
 * \brief AVX-512 SIMD int type
 */
using avx_512_simd_int = simd_pack<vector_mode_t::AVX512, int32_t, __m512i>;

/*!
 * \brief AVX-512 SIMD long type
 */
using avx_512_simd_long = simd_pack<vector_mode_t::AVX512, int64_t, __m512i>;

/*!
 * \brief Define traits to get vectorization information for types in AVX512 vector mode.
 */
template <typename T>
struct avx512_intrinsic_traits {
    static constexpr bool vectorizable = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr size_t size       = 1;          ///< Numbers of elements done at once
    static constexpr size_t alignment  = alignof(T); ///< Necessary number of bytes of alignment for this type

    using intrinsic_type = T; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<float> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 16;   ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_float; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<double> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 8;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_double; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<std::complex<float>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 8;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_complex_float<std::complex<float>>; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<std::complex<double>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 4;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_complex_double<std::complex<double>>; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<etl::complex<float>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 8;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_complex_float<etl::complex<float>>; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<etl::complex<double>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 4;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_complex_double<etl::complex<double>>; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<int8_t> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 64;           ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;           ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_byte; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<int16_t> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 32;           ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;           ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_short; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<int32_t> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 16;            ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;           ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_int; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<int64_t> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 8;            ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;           ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_512_simd_long; ///< The vector type
};

/*!
 * \brief Advanced Vector eXtensions 512 (AVX-512) operations implementation.
 */
struct avx512_vec {
    /*!
     * \brief The traits for this vector implementation
     */
    template <typename T>
    using traits = avx512_intrinsic_traits<T>;

    /*!
     * \brief The vector type for the given type for this vector implementation
     */
    template <typename T>
    using vec_type = typename traits<T>::intrinsic_type;

#ifdef VEC_DEBUG

    /*!
     * \brief Print the value of a AVX-512 vector of double
     */
    template <typename T>
    static std::string debug_d(T value) {
        union test {
            __m512d vec;
            double array[8];
            test(__m512d vec) : vec(vec) {}
        };

        test u_value = value;
        std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << "," << u_value.array[2] << "," << u_value.array[3] << "," << u_value.array[4] << ","
                  << u_value.array[5] << "," << u_value.array[6] << "," << u_value.array[7] << "]" << std::endl;
    }

    /*!
     * \brief Print the value of a AVX-512 vector of float
     */
    template <typename T>
    static std::string debug_s(T value) {
        union test {
            __m512 vec;
            float array[16];
            test(__m512 vec) : vec(vec) {}
        };

        test u_value = value;
        std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << "," << u_value.array[2] << "," << u_value.array[3] << "," << u_value.array[4] << ","
                  << u_value.array[5] << "," << u_value.array[6] << "," << u_value.array[7] << "," << u_value.array[8] << "," << u_value.array[9] << ","
                  << u_value.array[10] << "," << u_value.array[11] << "," << u_value.array[12] << "," << u_value.array[13] << "," << u_value.array[14] << ","
                  << u_value.array[15] << "]" << std::endl;
    }

#else

    /*!
     * \brief Print the value of a AVX-512 vector of double
     */
    template <typename T>
    static std::string debug_d(T) {
        return "";
    }

    /*!
     * \brief Print the value of a AVX-512 vector of float
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
    ETL_INLINE_VEC_VOID storeu(float* memory, avx_512_simd_float value) {
        _mm512_storeu_ps(memory, value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(double* memory, avx_512_simd_double value) {
        _mm512_storeu_pd(memory, value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(std::complex<float>* memory, avx_512_simd_complex_float<std::complex<float>> value) {
        _mm512_storeu_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(std::complex<double>* memory, avx_512_simd_complex_double<std::complex<double>> value) {
        _mm512_storeu_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(etl::complex<float>* memory, avx_512_simd_complex_float<etl::complex<float>> value) {
        _mm512_storeu_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(etl::complex<double>* memory, avx_512_simd_complex_double<etl::complex<double>> value) {
        _mm512_storeu_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int8_t* memory, avx_512_simd_byte value) {
        _mm512_store_si512(reinterpret_cast<__m512i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int16_t* memory, avx_512_simd_short value) {
        _mm512_store_si512(reinterpret_cast<__m512i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int32_t* memory, avx_512_simd_int value) {
        _mm512_store_si512(reinterpret_cast<__m512i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int64_t* memory, avx_512_simd_long value) {
        _mm512_store_si512(reinterpret_cast<__m512i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(float* memory, avx_512_simd_float value) {
        _mm512_store_ps(memory, value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(double* memory, avx_512_simd_double value) {
        _mm512_store_pd(memory, value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(std::complex<float>* memory, avx_512_simd_complex_float<std::complex<float>> value) {
        _mm512_store_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(std::complex<double>* memory, avx_512_simd_complex_double<std::complex<double>> value) {
        _mm512_store_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(etl::complex<float>* memory, avx_512_simd_complex_float<etl::complex<float>> value) {
        _mm512_store_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(etl::complex<double>* memory, avx_512_simd_complex_double<etl::complex<double>> value) {
        _mm512_store_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int8_t* memory, avx_512_simd_byte value) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int16_t* memory, avx_512_simd_short value) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int32_t* memory, avx_512_simd_int value) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int64_t* memory, avx_512_simd_long value) {
        _mm512_stream_si512(reinterpret_cast<__m512i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(float* memory, avx_512_simd_float value) {
        _mm512_stream_ps(memory, value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(double* memory, avx_512_simd_double value) {
        _mm512_stream_pd(memory, value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(std::complex<float>* memory, avx_512_simd_complex_float<std::complex<float>> value) {
        _mm512_stream_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(std::complex<double>* memory, avx_512_simd_complex_double<std::complex<double>> value) {
        _mm512_stream_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(etl::complex<float>* memory, avx_512_simd_complex_float<etl::complex<float>> value) {
        _mm512_stream_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(etl::complex<double>* memory, avx_512_simd_complex_double<etl::complex<double>> value) {
        _mm512_stream_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_byte) load(const int8_t* memory) {
        return _mm512_load_si512(reinterpret_cast<const __m512i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_short) load(const int16_t* memory) {
        return _mm512_load_si512(reinterpret_cast<const __m512i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_int) load(const int32_t* memory) {
        return _mm512_load_si512(reinterpret_cast<const __m512i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_long) load(const int64_t* memory) {
        return _mm512_load_si512(reinterpret_cast<const __m512i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_float) load(const float* memory) {
        return _mm512_load_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_double) load(const double* memory) {
        return _mm512_load_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_float<std::complex<float>>) load(const std::complex<float>* memory) {
        return _mm512_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_double<std::complex<double>>) load(const std::complex<double>* memory) {
        return _mm512_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_float<etl::complex<float>>) load(const etl::complex<float>* memory) {
        return _mm512_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_double<etl::complex<double>>) load(const etl::complex<double>* memory) {
        return _mm512_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_float) loadu(const float* memory) {
        return _mm512_loadu_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_double) loadu(const double* memory) {
        return _mm512_loadu_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_float<std::complex<float>>) loadu(const std::complex<float>* memory) {
        return _mm512_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_double<std::complex<double>>) loadu(const std::complex<double>* memory) {
        return _mm512_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_float<etl::complex<float>>) loadu(const etl::complex<float>* memory) {
        return _mm512_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_double<etl::complex<double>>) loadu(const etl::complex<double>* memory) {
        return _mm512_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_byte) set(int8_t value) {
        return _mm512_set1_epi8(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_short) set(int16_t value) {
        return _mm512_set1_epi16(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_int) set(int32_t value) {
        return _mm512_set1_epi32(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_long) set(int64_t value) {
        return _mm512_set1_epi64(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_double) set(double value) {
        return _mm512_set1_pd(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_float) set(float value) {
        return _mm512_set1_ps(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_float<std::complex<float>>) set(std::complex<float> value) {
        std::complex<float> tmp[]{value, value, value, value, value, value, value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_double<std::complex<double>>) set(std::complex<double> value) {
        std::complex<double> tmp[]{value, value, value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_float<etl::complex<float>>) set(etl::complex<float> value) {
        etl::complex<float> tmp[]{value, value, value, value, value, value, value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_512_simd_complex_double<etl::complex<double>>) set(etl::complex<double> value) {
        etl::complex<double> tmp[]{value, value, value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Return a packed vector of zeroes of the given type
     */
    template <typename T>
    ETL_TMP_INLINE(typename avx512_intrinsic_traits<T>::intrinsic_type)
    zero();

    /*!
     * \brief Round up each values of the vector and return them
     */
    ETL_STATIC_INLINE(avx_512_simd_float) round_up(avx_512_simd_float x) {
        return _mm512_roundscale_ps(x.value, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    /*!
     * \brief Round up each values of the vector and return them
     */
    ETL_STATIC_INLINE(avx_512_simd_double) round_up(avx_512_simd_double x) {
        return _mm512_roundscale_pd(x.value, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_byte) add(avx_512_simd_byte lhs, avx_512_simd_byte rhs) {
        return _mm512_add_epi8(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_short) add(avx_512_simd_short lhs, avx_512_simd_short rhs) {
        return _mm512_add_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_int) add(avx_512_simd_int lhs, avx_512_simd_int rhs) {
        return _mm512_add_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_long) add(avx_512_simd_long lhs, avx_512_simd_long rhs) {
        return _mm512_add_epi64(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_float) add(avx_512_simd_float lhs, avx_512_simd_float rhs) {
        return _mm512_add_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_double) add(avx_512_simd_double lhs, avx_512_simd_double rhs) {
        return _mm512_add_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    template <typename T>
    ETL_STATIC_INLINE(avx_512_simd_complex_float<T>)
    add(avx_512_simd_complex_float<T> lhs, avx_512_simd_complex_float<T> rhs) {
        return _mm512_add_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    template <typename T>
    ETL_STATIC_INLINE(avx_512_simd_complex_double<T>)
    add(avx_512_simd_complex_double<T> lhs, avx_512_simd_complex_double<T> rhs) {
        return _mm512_add_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_byte) sub(avx_512_simd_byte lhs, avx_512_simd_byte rhs) {
        return _mm512_sub_epi8(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_short) sub(avx_512_simd_short lhs, avx_512_simd_short rhs) {
        return _mm512_sub_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_int) sub(avx_512_simd_int lhs, avx_512_simd_int rhs) {
        return _mm512_sub_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_long) sub(avx_512_simd_long lhs, avx_512_simd_long rhs) {
        return _mm512_sub_epi64(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_float) sub(avx_512_simd_float lhs, avx_512_simd_float rhs) {
        return _mm512_sub_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_512_simd_double) sub(avx_512_simd_double lhs, avx_512_simd_double rhs) {
        return _mm512_sub_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    template <typename T>
    ETL_STATIC_INLINE(avx_512_simd_complex_float<T>)
    sub(avx_512_simd_complex_float<T> lhs, avx_512_simd_complex_float<T> rhs) {
        return _mm512_sub_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    template <typename T>
    ETL_STATIC_INLINE(avx_512_simd_complex_double<T>)
    sub(avx_512_simd_complex_double<T> lhs, avx_512_simd_complex_double<T> rhs) {
        return _mm512_sub_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_STATIC_INLINE(avx_512_simd_float) sqrt(avx_512_simd_float x) {
        return _mm512_sqrt_ps(x.value);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_STATIC_INLINE(avx_512_simd_double) sqrt(avx_512_simd_double x) {
        return _mm512_sqrt_pd(x.value);
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_STATIC_INLINE(avx_512_simd_float) minus(avx_512_simd_float x) {
        return _mm512_xor_ps(x.value, _mm512_set1_ps(-0.f));
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_STATIC_INLINE(avx_512_simd_double) minus(avx_512_simd_double x) {
        return _mm512_xor_pd(x.value, _mm512_set1_pd(-0.));
    }

    /*!
     * \brief Multiply the two given vectors of byte
     */
    ETL_STATIC_INLINE(avx_512_simd_byte) mul(avx_512_simd_byte lhs, avx_512_simd_byte rhs) {
        // Split in multiple vectors (odd and even)
        __m512i lhs_odd = _mm512_srli_epi16(lhs.value, 8);
        __m512i rhs_odd = _mm512_srli_epi16(rhs.value, 8);
        // Do the multiplication on each side
        __m512i mul_even = _mm512_mullo_epi16(lhs.value, rhs.value);
        __m512i mul_odd  = _mm512_mullo_epi16(lhs_odd, rhs_odd);
        // Combine again
        __m512i temp = _mm512_slli_epi16(mul_odd, 8);
        return _mm512_mask_mov_epi8(mul_even, 0xAAAAAAAAAAAAAAAA, temp);
    }

    /*!
     * \brief Multiply the two given vectors of short
     */
    ETL_STATIC_INLINE(avx_512_simd_short) mul(avx_512_simd_short lhs, avx_512_simd_short rhs) {
        return _mm512_mullo_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors of int
     */
    ETL_STATIC_INLINE(avx_512_simd_int) mul(avx_512_simd_int lhs, avx_512_simd_int rhs) {
        return _mm512_mullo_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors of long
     */
    ETL_STATIC_INLINE(avx_512_simd_long) mul(avx_512_simd_long lhs, avx_512_simd_long rhs) {
        return _mm512_mullo_epi64(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    ETL_STATIC_INLINE(avx_512_simd_float) mul(avx_512_simd_float lhs, avx_512_simd_float rhs) {
        return _mm512_mul_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    ETL_STATIC_INLINE(avx_512_simd_double) mul(avx_512_simd_double lhs, avx_512_simd_double rhs) {
        return _mm512_mul_pd(lhs.value, rhs.value);
    }

    /*!
     * \copydoc avx_512_vec::mul
     */
    template <typename T>
    ETL_STATIC_INLINE(avx_512_simd_complex_float<T>)
    mul(avx_512_simd_complex_float<T> lhs, avx_512_simd_complex_float<T> rhs) {
        //lhs = [x1.real, x1.img, x2.real, x2.img, ...]
        //rhs = [y1.real, y1.img, y2.real, y2.img, ...]

        //ymm1 = [y1.real, y1.real, y2.real, y2.real, ...]
        __m512 ymm1 = _mm512_moveldup_ps(rhs.value);

        //ymm2 = [x1.img, x1.real, x2.img, x2.real]
        __m512 ymm2 = _mm512_permute_ps(lhs.value, 0b10110001);

        //ymm3 = [y1.imag, y1.imag, y2.imag, y2.imag]
        __m512 ymm3 = _mm512_movehdup_ps(rhs.value);

        //ymm4 = ymm2 * ymm3
        __m512 ymm4 = _mm512_mul_ps(ymm2, ymm3);

        //result = [(lhs * ymm1) -+ ymm4];

        return _mm512_fmaddsub_ps(lhs.value, ymm1, ymm4);
    }

    /*!
     * \copydoc avx_512_vec::mul
     */
    template <typename T>
    ETL_STATIC_INLINE(avx_512_simd_complex_double<T>)
    mul(avx_512_simd_complex_double<T> lhs, avx_512_simd_complex_double<T> rhs) {
        __m512d ymm1 = _mm512_shuffle_pd(rhs.value, rhs.value, 0x55);
        __m512d ymm2 = _mm512_shuffle_pd(lhs.value, lhs.value, 0xFF);
        __m512d ymm3 = _mm512_shuffle_pd(lhs.value, lhs.value, 0);
        __m512d ymm4 = _mm512_mul_pd(ymm2, ymm1);
        return _mm512_fmaddsub_pd(ymm3, rhs.value, ymm4);
    }

    /*!
     * \brief Fused-Multiply Add of the three given vector of float
     */
    ETL_STATIC_INLINE(avx_512_simd_float) fmadd(avx_512_simd_float a, avx_512_simd_float b, avx_512_simd_float c) {
        return _mm512_fmadd_ps(a.value, b.value, c.value);
    }

    /*!
     * \brief Fused-Multiply Add of the three given vector of double
     */
    ETL_STATIC_INLINE(avx_512_simd_double) fmadd(avx_512_simd_double a, avx_512_simd_double b, avx_512_simd_double c) {
        return _mm512_fmadd_pd(a.value, b.value, c.value);
    }

    /*!
     * \copydoc avx_512_vec::fmadd
     */
    template <typename T>
    ETL_STATIC_INLINE(avx_512_simd_complex_float<T>)
    fmadd(avx_512_simd_complex_float<T> a, avx_512_simd_complex_float<T> b, avx_512_simd_complex_float<T> c) {
        return add(mul(a, b), c);
    }

    /*!
     * \copydoc avx_512_vec::fmadd
     */
    template <typename T>
    ETL_STATIC_INLINE(avx_512_simd_complex_double<T>)
    fmadd(avx_512_simd_complex_double<T> a, avx_512_simd_complex_double<T> b, avx_512_simd_complex_double<T> c) {
        return add(mul(a, b), c);
    }

    /*!
     * \brief Divide the two given vectors
     */
    ETL_STATIC_INLINE(avx_512_simd_float) div(avx_512_simd_float lhs, avx_512_simd_float rhs) {
        return _mm512_div_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Divide the two given vectors
     */
    ETL_STATIC_INLINE(avx_512_simd_double) div(avx_512_simd_double lhs, avx_512_simd_double rhs) {
        return _mm512_div_pd(lhs.value, rhs.value);
    }

    //Min

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(avx_512_simd_double) min(avx_512_simd_double lhs, avx_512_simd_double rhs) {
        return _mm512_min_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(avx_512_simd_float) min(avx_512_simd_float lhs, avx_512_simd_float rhs) {
        return _mm512_min_ps(lhs.value, rhs.value);
    }

    //Max

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(avx_512_simd_double) max(avx_512_simd_double lhs, avx_512_simd_double rhs) {
        return _mm512_max_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(avx_512_simd_float) max(avx_512_simd_float lhs, avx_512_simd_float rhs) {
        return _mm512_max_ps(lhs.value, rhs.value);
    }

    // Horizontal sum reductions
    // TODO "Vectorize" these reductions

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int8_t) hadd(avx_512_simd_byte in) {
        int8_t acc = 0;
        for (size_t i = 0; i < 64; ++i) {
            acc += in[i];
        }
        return acc;
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int16_t) hadd(avx_512_simd_short in) {
        int16_t acc = 0;
        for (size_t i = 0; i < 32; ++i) {
            acc += in[i];
        }
        return acc;
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int32_t) hadd(avx_512_simd_int in) {
        int32_t acc = 0;
        for (size_t i = 0; i < 16; ++i) {
            acc += in[i];
        }
        return acc;
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int64_t) hadd(avx_512_simd_long in) {
        int64_t acc = 0;
        for (size_t i = 0; i < 8; ++i) {
            acc += in[i];
        }
        return acc;
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(float) hadd(avx_512_simd_float in) {
        return in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7] + in[8] + in[9] + in[10] + in[11] + in[12] + in[13] + in[14] + in[15];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(double) hadd(avx_512_simd_double in) {
        return in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template <typename T>
    ETL_STATIC_INLINE(T)
    hadd(avx_512_simd_complex_float<T> in) {
        return in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template <typename T>
    ETL_STATIC_INLINE(T)
    hadd(avx_512_simd_complex_double<T> in) {
        return in[0] + in[1] + in[2] + in[3];
    }
};

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_byte)
avx512_vec::zero<int8_t>() {
    return _mm512_setzero_si512();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_short)
avx512_vec::zero<int16_t>() {
    return _mm512_setzero_si512();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_int)
avx512_vec::zero<int32_t>() {
    return _mm512_setzero_si512();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_long)
avx512_vec::zero<int64_t>() {
    return _mm512_setzero_si512();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_float)
avx512_vec::zero<float>() {
    return _mm512_setzero_ps();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_double)
avx512_vec::zero<double>() {
    return _mm512_setzero_pd();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_complex_float<etl::complex<float>>)
avx512_vec::zero<etl::complex<float>>() {
    return _mm512_setzero_ps();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_complex_double<etl::complex<double>>)
avx512_vec::zero<etl::complex<double>>() {
    return _mm512_setzero_pd();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_complex_float<std::complex<float>>)
avx512_vec::zero<std::complex<float>>() {
    return _mm512_setzero_ps();
}

/*!
 * \copydoc avx512_vec::zero
 */
template <>
ETL_OUT_INLINE(avx_512_simd_complex_double<std::complex<double>>)
avx512_vec::zero<std::complex<double>>() {
    return _mm512_setzero_pd();
}

} //end of namespace etl

#endif //__AVX512F__
