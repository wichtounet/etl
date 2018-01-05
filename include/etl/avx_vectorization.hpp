//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file avx_vectorization.hpp
 * \brief Contains AVX vectorized functions for the vectorized assignment of expressions
 */

#pragma once

#ifdef __AVX__

#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>

#include "etl/inline.hpp"
#include "etl/avx_exp.hpp"

#ifdef VECT_DEBUG
#include <iostream>
#endif

#define ETL_INLINE_VEC_VOID ETL_STATIC_INLINE(void)
#define ETL_INLINE_VEC_256 ETL_STATIC_INLINE(__m256)
#define ETL_INLINE_VEC_256D ETL_STATIC_INLINE(__m256d)
#define ETL_OUT_VEC_256 ETL_OUT_INLINE(__m256)
#define ETL_OUT_VEC_256D ETL_OUT_INLINE(__m256d)

namespace etl {

/*!
 * \brief AVX SIMD float type
 */
using avx_simd_float = simd_pack<vector_mode_t::AVX, float, __m256>;

/*!
 * \brief AVX SIMD double type
 */
using avx_simd_double = simd_pack<vector_mode_t::AVX, double, __m256d>;

/*!
 * \brief AVX SIMD complex float type
 */
template<typename T>
using avx_simd_complex_float = simd_pack<vector_mode_t::AVX, T, __m256>;

/*!
 * \brief AVX SIMD complex double type
 */
template<typename T>
using avx_simd_complex_double = simd_pack<vector_mode_t::AVX, T, __m256d>;

/*!
 * \brief AVX SIMD byte type
 */
using avx_simd_byte = simd_pack<vector_mode_t::AVX, int8_t, __m256i>;

/*!
 * \brief AVX SIMD short type
 */
using avx_simd_short = simd_pack<vector_mode_t::AVX, int16_t, __m256i>;

/*!
 * \brief AVX SIMD int type
 */
using avx_simd_int = simd_pack<vector_mode_t::AVX, int32_t, __m256i>;

/*!
 * \brief AVX SIMD long type
 */
using avx_simd_long = simd_pack<vector_mode_t::AVX, int64_t, __m256i>;

/*!
 * \brief Define traits to get vectorization information for types in AVX vector mode.
 */
template <typename T>
struct avx_intrinsic_traits {
    static constexpr bool vectorizable     = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr size_t size      = 1;          ///< Numbers of elements done at once
    static constexpr size_t alignment = alignof(T); ///< Necessary number of bytes of alignment for this type

    using intrinsic_type = T; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<float> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 8; ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_float; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<double> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 4; ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_double; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<std::complex<float>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 4; ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_complex_float<std::complex<float>>; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<std::complex<double>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 2; ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_complex_double<std::complex<double>>; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<etl::complex<float>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 4; ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_complex_float<etl::complex<float>>; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<etl::complex<double>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 2; ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_complex_double<etl::complex<double>>; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<int8_t> {
    static constexpr bool vectorizable     = avx2_enabled; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 32;            ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32;           ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_byte; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<int16_t> {
    static constexpr bool vectorizable     = avx2_enabled; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 16;            ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32;           ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_short; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<int32_t> {
    static constexpr bool vectorizable     = avx2_enabled; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 8;            ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32;           ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_int; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<int64_t> {
    static constexpr bool vectorizable     = avx2_enabled; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size      = 4;            ///< Numbers of elements in a vector
    static constexpr size_t alignment = 32;           ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = avx_simd_long; ///< The vector type
};

/*!
 * \brief Advanced Vector eXtensions (AVX) operations implementation.
 */
struct avx_vec {
    /*!
     * \brief The traits for this vector implementation
     */
    template <typename T>
    using traits = avx_intrinsic_traits<T>;

    /*!
     * \brief The vector type for the given type for this vector implementation
     */
    template <typename T>
    using vec_type = typename traits<T>::intrinsic_type;

#ifdef VEC_DEBUG

    /*!
     * \brief Print the value of a AVX vector of double
     */
    template <typename T>
    static std::string debug_d(T value) {
        union test {
            __m256d vec; // a data field, maybe a register, maybe not
            double array[4];
            test(__m256d vec)
                    : vec(vec) {}
        };

        test u_value = value;
        std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << "," << u_value.array[2] << "," << u_value.array[3] << "]" << std::endl;
    }

    /*!
     * \brief Print the value of a AVX vector of float
     */
    template <typename T>
    static std::string debug_s(T value) {
        union test {
            __m256 vec; // a data field, maybe a register, maybe not
            float array[8];
            test(__m256 vec)
                    : vec(vec) {}
        };

        test u_value = value;
        std::cout << "[" << u_value.array[0] << "," << u_value.array[1] << "," << u_value.array[2] << "," << u_value.array[3]
                  << "," << u_value.array[4] << "," << u_value.array[5] << "," << u_value.array[6] << "," << u_value.array[7] << "]" << std::endl;
    }

#else

    /*!
     * \brief Print the value of a AVX vector of double
     */
    template <typename T>
    static std::string debug_d(T) {
        return "";
    }

    /*!
     * \brief Print the value of a AVX vector of float
     */
    template <typename T>
    static std::string debug_s(T) {
        return "";
    }

#endif

#ifdef __AVX2__
    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(int8_t* memory, avx_simd_byte value) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(int16_t* memory, avx_simd_short value) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(int32_t* memory, avx_simd_int value) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(int64_t* memory, avx_simd_long value) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }
#endif

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(float* memory, avx_simd_float value) {
        _mm256_storeu_ps(memory, value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(double* memory, avx_simd_double value) {
        _mm256_storeu_pd(memory, value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(std::complex<float>* memory, avx_simd_complex_float<std::complex<float>> value) {
        _mm256_storeu_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(std::complex<double>* memory, avx_simd_complex_double<std::complex<double>> value) {
        _mm256_storeu_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(etl::complex<float>* memory, avx_simd_complex_float<etl::complex<float>> value) {
        _mm256_storeu_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) storeu(etl::complex<double>* memory, avx_simd_complex_double<etl::complex<double>> value) {
        _mm256_storeu_pd(reinterpret_cast<double*>(memory), value.value);
    }

#ifdef __AVX2__
    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int8_t* memory, avx_simd_byte value) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int16_t* memory, avx_simd_short value) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int32_t* memory, avx_simd_int value) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(int64_t* memory, avx_simd_long value) {
        _mm256_stream_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }
#endif

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(float* memory, avx_simd_float value) {
        _mm256_stream_ps(memory, value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(double* memory, avx_simd_double value) {
        _mm256_stream_pd(memory, value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(std::complex<float>* memory, avx_simd_complex_float<std::complex<float>> value) {
        _mm256_stream_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(std::complex<double>* memory, avx_simd_complex_double<std::complex<double>> value) {
        _mm256_stream_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(etl::complex<float>* memory, avx_simd_complex_float<etl::complex<float>> value) {
        _mm256_stream_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) stream(etl::complex<double>* memory, avx_simd_complex_double<etl::complex<double>> value) {
        _mm256_stream_pd(reinterpret_cast<double*>(memory), value.value);
    }

#ifdef __AVX2__
    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int8_t* memory, avx_simd_byte value) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int16_t* memory, avx_simd_short value) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int32_t* memory, avx_simd_int value) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(int64_t* memory, avx_simd_long value) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(memory), value.value);
    }
#endif

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(float* memory, avx_simd_float value) {
        _mm256_store_ps(memory, value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(double* memory, avx_simd_double value) {
        _mm256_store_pd(memory, value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(std::complex<float>* memory, avx_simd_complex_float<std::complex<float>> value) {
        _mm256_store_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(std::complex<double>* memory, avx_simd_complex_double<std::complex<double>> value) {
        _mm256_store_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(etl::complex<float>* memory, avx_simd_complex_float<etl::complex<float>> value) {
        _mm256_store_ps(reinterpret_cast<float*>(memory), value.value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_STATIC_INLINE(void) store(etl::complex<double>* memory, avx_simd_complex_double<etl::complex<double>> value) {
        _mm256_store_pd(reinterpret_cast<double*>(memory), value.value);
    }

    /*!
     * \brief Return a packed vector of zeroes of the given type
     */
    template<typename T>
    ETL_TMP_INLINE(typename avx_intrinsic_traits<T>::intrinsic_type) zero();

#ifdef __AVX2__
    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_byte) load(const int8_t* memory) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_short) load(const int16_t* memory) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_int) load(const int32_t* memory) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_long) load(const int64_t* memory) {
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(memory));
    }
#endif

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_float) load(const float* memory) {
        return _mm256_load_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_double) load(const double* memory) {
        return _mm256_load_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_complex_float<std::complex<float>>) load(const std::complex<float>* memory) {
        return _mm256_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_complex_double<std::complex<double>>) load(const std::complex<double>* memory) {
        return _mm256_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_complex_float<etl::complex<float>>) load(const etl::complex<float>* memory) {
        return _mm256_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_complex_double<etl::complex<double>>) load(const etl::complex<double>* memory) {
        return _mm256_load_pd(reinterpret_cast<const double*>(memory));
    }

#ifdef __AVX2__
    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_byte) loadu(const int8_t* memory) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_short) loadu(const int16_t* memory) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_int) loadu(const int32_t* memory) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_long) loadu(const int64_t* memory) {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(memory));
    }
#endif

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_float) loadu(const float* memory) {
        return _mm256_loadu_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_double) loadu(const double* memory) {
        return _mm256_loadu_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_complex_float<std::complex<float>>) loadu(const std::complex<float>* memory) {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_complex_double<std::complex<double>>) loadu(const std::complex<double>* memory) {
        return _mm256_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_complex_float<etl::complex<float>>) loadu(const etl::complex<float>* memory) {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_STATIC_INLINE(avx_simd_complex_double<etl::complex<double>>) loadu(const etl::complex<double>* memory) {
        return _mm256_loadu_pd(reinterpret_cast<const double*>(memory));
    }

#ifdef __AVX2__
    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_byte) set(int8_t value) {
        return _mm256_set1_epi8(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_short) set(int16_t value) {
        return _mm256_set1_epi16(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_int) set(int32_t value) {
        return _mm256_set1_epi32(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_long) set(int64_t value) {
        return _mm256_set1_epi64x(value);
    }
#endif

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_double) set(double value) {
        return _mm256_set1_pd(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_float) set(float value) {
        return _mm256_set1_ps(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_complex_float<std::complex<float>>) set(std::complex<float> value) {
        std::complex<float> tmp[]{value, value, value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_complex_double<std::complex<double>>) set(std::complex<double> value) {
        std::complex<double> tmp[]{value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_complex_float<etl::complex<float>>) set(etl::complex<float> value) {
        etl::complex<float> tmp[]{value, value, value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_STATIC_INLINE(avx_simd_complex_double<etl::complex<double>>) set(etl::complex<double> value) {
        etl::complex<double> tmp[]{value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Round up each values of the vector and return them
     */
    ETL_STATIC_INLINE(avx_simd_float) round_up(avx_simd_float x) {
        return _mm256_round_ps(x.value, (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC));
    }

    /*!
     * \brief Round up each values of the vector and return them
     */
    ETL_STATIC_INLINE(avx_simd_double) round_up(avx_simd_double x) {
        return _mm256_round_pd(x.value, (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC));
    }

    // Addition

#ifdef __AVX2__
    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_byte) add(avx_simd_byte lhs, avx_simd_byte rhs) {
        return _mm256_add_epi8(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_short) add(avx_simd_short lhs, avx_simd_short rhs) {
        return _mm256_add_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_int) add(avx_simd_int lhs, avx_simd_int rhs) {
        return _mm256_add_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_long) add(avx_simd_long lhs, avx_simd_long rhs) {
        return _mm256_add_epi64(lhs.value, rhs.value);
    }
#endif

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_float) add(avx_simd_float lhs, avx_simd_float rhs) {
        return _mm256_add_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_double) add(avx_simd_double lhs, avx_simd_double rhs) {
        return _mm256_add_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_float<T>) add(avx_simd_complex_float<T> lhs, avx_simd_complex_float<T> rhs) {
        return _mm256_add_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_double<T>) add(avx_simd_complex_double<T> lhs, avx_simd_complex_double<T> rhs) {
        return _mm256_add_pd(lhs.value, rhs.value);
    }

    // Subtraction

#ifdef __AVX2__
    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_byte) sub(avx_simd_byte lhs, avx_simd_byte rhs) {
        return _mm256_sub_epi8(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_short) sub(avx_simd_short lhs, avx_simd_short rhs) {
        return _mm256_sub_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_int) sub(avx_simd_int lhs, avx_simd_int rhs) {
        return _mm256_sub_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_long) sub(avx_simd_long lhs, avx_simd_long rhs) {
        return _mm256_sub_epi64(lhs.value, rhs.value);
    }
#endif

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_float) sub(avx_simd_float lhs, avx_simd_float rhs) {
        return _mm256_sub_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_STATIC_INLINE(avx_simd_double) sub(avx_simd_double lhs, avx_simd_double rhs) {
        return _mm256_sub_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_float<T>) sub(avx_simd_complex_float<T> lhs, avx_simd_complex_float<T> rhs) {
        return _mm256_sub_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_double<T>) sub(avx_simd_complex_double<T> lhs, avx_simd_complex_double<T> rhs) {
        return _mm256_sub_pd(lhs.value, rhs.value);
    }

    // Square root

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_STATIC_INLINE(avx_simd_float) sqrt(avx_simd_float x) {
        return _mm256_sqrt_ps(x.value);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_STATIC_INLINE(avx_simd_double) sqrt(avx_simd_double x) {
        return _mm256_sqrt_pd(x.value);
    }

    // Negation

    // TODO negation epi32

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_STATIC_INLINE(avx_simd_float) minus(avx_simd_float x) {
        return _mm256_xor_ps(x.value, _mm256_set1_ps(-0.f));
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_STATIC_INLINE(avx_simd_double) minus(avx_simd_double x) {
        return _mm256_xor_pd(x.value, _mm256_set1_pd(-0.));
    }

    // Multiplication

#ifdef __AVX2__
    /*!
     * \brief Multiply the two given vectors of byte
     */
    ETL_STATIC_INLINE(avx_simd_byte) mul(avx_simd_byte lhs, avx_simd_byte rhs) {
        auto aodd    = _mm256_srli_epi16(lhs.value, 8);
        auto bodd    = _mm256_srli_epi16(rhs.value, 8);
        auto muleven = _mm256_mullo_epi16(lhs.value, rhs.value);
        auto mulodd  = _mm256_slli_epi16(_mm256_mullo_epi16(aodd, bodd), 8);
        return _mm256_blendv_epi8(mulodd, muleven, _mm256_set1_epi32(0x00FF00FF));
    }

    /*!
     * \brief Multiply the two given vectors of short
     */
    ETL_STATIC_INLINE(avx_simd_short) mul(avx_simd_short lhs, avx_simd_short rhs) {
        return _mm256_mullo_epi16(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors of int
     */
    ETL_STATIC_INLINE(avx_simd_int) mul(avx_simd_int lhs, avx_simd_int rhs) {
        return _mm256_mullo_epi32(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors of long
     */
    ETL_STATIC_INLINE(avx_simd_long) mul(avx_simd_long lhs, avx_simd_long rhs) {
        int64_t result[4];

        result[0] = lhs[0] * rhs[0];
        result[1] = lhs[1] * rhs[1];
        result[2] = lhs[2] * rhs[2];
        result[3] = lhs[3] * rhs[3];

        return loadu(&result[0]);
    }
#endif

    /*!
     * \brief Multiply the two given vectors
     */
    ETL_STATIC_INLINE(avx_simd_float) mul(avx_simd_float lhs, avx_simd_float rhs) {
        return _mm256_mul_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    ETL_STATIC_INLINE(avx_simd_double) mul(avx_simd_double lhs, avx_simd_double rhs) {
        return _mm256_mul_pd(lhs.value, rhs.value);
    }

    /*!
     * \copydoc avx_vec::mul
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_float<T>) mul(avx_simd_complex_float<T> lhs, avx_simd_complex_float<T> rhs) {
        //lhs = [x1.real, x1.img, x2.real, x2.img, ...]
        //rhs = [y1.real, y1.img, y2.real, y2.img, ...]

        //ymm1 = [y1.real, y1.real, y2.real, y2.real, ...]
        __m256 ymm1 = _mm256_moveldup_ps(rhs.value);

        //ymm2 = [x1.img, x1.real, x2.img, x2.real]
        __m256 ymm2 = _mm256_permute_ps(lhs.value, 0b10110001);

        //ymm3 = [y1.imag, y1.imag, y2.imag, y2.imag]
        __m256 ymm3 = _mm256_movehdup_ps(rhs.value);

        //ymm4 = ymm2 * ymm3
        __m256 ymm4 = _mm256_mul_ps(ymm2, ymm3);

         //result = [(lhs * ymm1) -+ ymm4];

#ifdef __FMA__
        return _mm256_fmaddsub_ps(lhs.value, ymm1, ymm4);
#elif defined(__FMA4__)
        return _mm256_maddsub_ps(lhs.value, ymm1, ymm4);
#else
        __m256 tmp = _mm256_mul_ps(lhs.value, ymm1);
        return _mm256_addsub_ps(tmp, ymm4);
#endif
    }

    /*!
     * \copydoc avx_vec::mul
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_double<T>) mul(avx_simd_complex_double<T> lhs, avx_simd_complex_double<T> rhs) {
        //lhs = [x1.real, x1.img, x2.real, x2.img]
        //rhs = [y1.real, y1.img, y2.real, y2.img]

        //ymm1 = [y1.real, y1.real, y2.real, y2.real]
        __m256d ymm1 = _mm256_movedup_pd(rhs.value);

        //ymm2 = [x1.img, x1.real, x2.img, x2.real]
        __m256d ymm2 = _mm256_permute_pd(lhs.value, 0b0101);

        //ymm3 = [y1.imag, y1.imag, y2.imag, y2.imag]
        __m256d ymm3 = _mm256_permute_pd(rhs.value, 0b1111);

        //ymm4 = ymm2 * ymm3
        __m256d ymm4 = _mm256_mul_pd(ymm2, ymm3);

        //result = [(lhs * ymm1) -+ ymm4];

#ifdef __FMA__
        return _mm256_fmaddsub_pd(lhs.value, ymm1, ymm4);
#elif defined(__FMA4__)
        return _mm256_maddsub_pd(lhs.value, ymm1, ymm4);
#else
        __m256d tmp = _mm256_mul_pd(lhs.value, ymm1);
        return _mm256_addsub_pd(tmp, ymm4);
#endif
    }

    // Fused Multiplay Add (FMA)

#ifdef __AVX2__
    /*!
     * \brief Fused-Multiply Add of the three given vector of bytes
     */
    ETL_STATIC_INLINE(avx_simd_byte) fmadd(avx_simd_byte a, avx_simd_byte b, avx_simd_byte c){
        return add(mul(a, b), c);
    }

    /*!
     * \brief Fused-Multiply Add of the three given vector of short
     */
    ETL_STATIC_INLINE(avx_simd_short) fmadd(avx_simd_short a, avx_simd_short b, avx_simd_short c){
        return add(mul(a, b), c);
    }

    /*!
     * \brief Fused-Multiply Add of the three given vector of int
     */
    ETL_STATIC_INLINE(avx_simd_int) fmadd(avx_simd_int a, avx_simd_int b, avx_simd_int c){
        return add(mul(a, b), c);
    }

    /*!
     * \brief Fused-Multiply Add of the three given vector of longs
     */
    ETL_STATIC_INLINE(avx_simd_long) fmadd(avx_simd_long a, avx_simd_long b, avx_simd_long c){
        return add(mul(a, b), c);
    }
#endif

    /*!
     * \copydoc avx_vec::fmadd
     */
    ETL_STATIC_INLINE(avx_simd_float) fmadd(avx_simd_float a, avx_simd_float b, avx_simd_float c) {
#ifdef __FMA__
        return _mm256_fmadd_ps(a.value, b.value, c.value);
#else
        return add(mul(a, b), c);
#endif
    }

    /*!
     * \copydoc avx_vec::fmadd
     */
    ETL_STATIC_INLINE(avx_simd_double) fmadd(avx_simd_double a, avx_simd_double b, avx_simd_double c) {
#ifdef __FMA__
        return _mm256_fmadd_pd(a.value, b.value, c.value);
#else
        return add(mul(a, b), c);
#endif
    }

    /*!
     * \copydoc avx_vec::fmadd
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_float<T>) fmadd(avx_simd_complex_float<T> a, avx_simd_complex_float<T> b, avx_simd_complex_float<T> c) {
        return add(mul(a, b), c);
    }

    /*!
     * \copydoc avx_vec::fmadd
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_double<T>) fmadd(avx_simd_complex_double<T> a, avx_simd_complex_double<T> b, avx_simd_complex_double<T> c) {
        return add(mul(a, b), c);
    }

    // Division

    /*!
     * \brief Divide the two given vectors
     */
    ETL_STATIC_INLINE(avx_simd_float) div(avx_simd_float lhs, avx_simd_float rhs) {
        return _mm256_div_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Divide the two given vectors
     */
    ETL_STATIC_INLINE(avx_simd_double) div(avx_simd_double lhs, avx_simd_double rhs) {
        return _mm256_div_pd(lhs.value, rhs.value);
    }

    /*!
     * \copydoc avx_vec::div
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_float<T>) div(avx_simd_complex_float<T> lhs, avx_simd_complex_float<T> rhs) {
        //lhs = [x1.real, x1.img, x2.real, x2.img ...]
        //rhs = [y1.real, y1.img, y2.real, y2.img ...]

        //ymm0 = [y1.real, y1.real, y2.real, y2.real, ...]
        __m256 ymm0 = _mm256_moveldup_ps(rhs.value);

        //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
        __m256 ymm1 = _mm256_movehdup_ps(rhs.value);

        //ymm2 = [x1.img, x1.real, x2.img, x2.real]
        __m256 ymm2 = _mm256_permute_ps(lhs.value, 0b10110001);

        //ymm4 = [x.img * y.img, x.real * y.img]
        __m256 ymm4 = _mm256_mul_ps(ymm2, ymm1);

    //ymm5 = subadd((lhs * ymm0), ymm4)

#ifdef __FMA__
        __m256 ymm5 = _mm256_fmsubadd_ps(lhs.value, ymm0, ymm4);
#else
        __m256 t1    = _mm256_mul_ps(lhs.value, ymm0);
        __m256 t2    = _mm256_sub_ps(_mm256_set1_ps(0.0), ymm4);
        __m256 ymm5  = _mm256_addsub_ps(t1, t2);
#endif

        //ymm3 = [y.imag^2, y.imag^2]
        __m256 ymm3 = _mm256_mul_ps(ymm1, ymm1);

        //ymm0 = (ymm0 * ymm0 + ymm3)

#ifdef __FMA__
        ymm0 = _mm256_fmadd_ps(ymm0, ymm0, ymm3);
#else
        __m256 t3    = _mm256_mul_ps(ymm0, ymm0);
        ymm0         = _mm256_add_ps(t3, ymm3);
#endif

        //result = ymm5 / ymm0
        return _mm256_div_ps(ymm5, ymm0);
    }

    /*!
     * \copydoc avx_vec::div
     */
    template<typename T>
    ETL_STATIC_INLINE(avx_simd_complex_double<T>) div(avx_simd_complex_double<T> lhs, avx_simd_complex_double<T> rhs) {
        //lhs = [x1.real, x1.img, x2.real, x2.img]
        //rhs = [y1.real, y1.img, y2.real, y2.img]

        //ymm0 = [y1.real, y1.real, y2.real, y2.real]
        __m256d ymm0 = _mm256_movedup_pd(rhs.value);

        //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
        __m256d ymm1 = _mm256_permute_pd(rhs.value, 0b1111);

        //ymm2 = [x1.img, x1.real, x2.img, x2.real]
        __m256d ymm2 = _mm256_permute_pd(lhs.value, 0b0101);

        //ymm4 = [x.img * y.img, x.real * y.img]
        __m256d ymm4 = _mm256_mul_pd(ymm2, ymm1);

        //ymm5 = subadd((lhs * ymm0), ymm4)

#ifdef __FMA__
        __m256d ymm5 = _mm256_fmsubadd_pd(lhs.value, ymm0, ymm4);
#else
        __m256d t1   = _mm256_mul_pd(lhs.value, ymm0);
        __m256d t2   = _mm256_sub_pd(_mm256_set1_pd(0.0), ymm4);
        __m256d ymm5 = _mm256_addsub_pd(t1, t2);
#endif

        //ymm3 = [y.imag^2, y.imag^2]
        __m256d ymm3 = _mm256_mul_pd(ymm1, ymm1);

    //ymm0 = (ymm0 * ymm0 + ymm3)

#ifdef __FMA__
        ymm0 = _mm256_fmadd_pd(ymm0, ymm0, ymm3);
#else
        __m256d t3   = _mm256_mul_pd(ymm0, ymm0);
        ymm0         = _mm256_add_pd(t3, ymm3);
#endif

        //result = ymm5 / ymm0
        return _mm256_div_pd(ymm5, ymm0);
    }

    // Cosinus

    /*!
     * \brief Compute the cosinus of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_float) cos(avx_simd_float x) {
        return etl::cos256_ps(x.value);
    }

    /*!
     * \brief Compute the sinus of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_float) sin(avx_simd_float x) {
        return etl::sin256_ps(x.value);
    }

#ifndef __INTEL_COMPILER

    //Exponential

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_float) exp(avx_simd_float x) {
        return etl::exp256_ps(x.value);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_double) exp(avx_simd_double x) {
        return etl::exp256_pd(x.value);
    }

    //Logarithm

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_float) log(avx_simd_float x) {
        return etl::log256_ps(x.value);
    }

#else //__INTEL_COMPILER

    //Exponential

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_double) exp(avx_simd_double x) {
        return _mm256_exp_pd(x.value);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_float) exp(avx_simd_float x) {
        return _mm256_exp_ps(x.value);
    }

    //Logarithm

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_double) log(avx_simd_double x) {
        return _mm256_log_pd(x.value);
    }

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_STATIC_INLINE(avx_simd_float) log(avx_simd_float x) {
        return _mm256_log_ps(x.value);
    }

#endif //__INTEL_COMPILER

    //Min

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(avx_simd_double) min(avx_simd_double lhs, avx_simd_double rhs) {
        return _mm256_min_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(avx_simd_float) min(avx_simd_float lhs, avx_simd_float rhs) {
        return _mm256_min_ps(lhs.value, rhs.value);
    }

    //Max

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(avx_simd_double) max(avx_simd_double lhs, avx_simd_double rhs) {
        return _mm256_max_pd(lhs.value, rhs.value);
    }

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_STATIC_INLINE(avx_simd_float) max(avx_simd_float lhs, avx_simd_float rhs) {
        return _mm256_max_ps(lhs.value, rhs.value);
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(float) hadd(avx_simd_float in) {
        const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(in.value, 1), _mm256_castps256_ps128(in.value));
        const __m128 x64  = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        const __m128 x32  = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        return _mm_cvtss_f32(x32);
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(double) hadd(avx_simd_double in) {
        const __m256d t1 = _mm256_hadd_pd(in.value, _mm256_permute2f128_pd(in.value, in.value, 1));
        const __m256d t2 = _mm256_hadd_pd(t1, t1);
        return _mm_cvtsd_f64(_mm256_castpd256_pd128(t2));
    }

    //TODO Vectorize the following functions

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int8_t) hadd(avx_simd_byte in) {
        return in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7]
             + in[8] + in[9] + in[10] + in[11] + in[12] + in[13] + in[14] + in[15]
             + in[16] + in[17] + in[18] + in[19] + in[20] + in[21] + in[22] + in[23]
             + in[24] + in[25] + in[26] + in[27] + in[28] + in[29] + in[30] + in[31];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int16_t) hadd(avx_simd_short in) {
        return in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7]
             + in[8] + in[9] + in[10] + in[11] + in[12] + in[13] + in[14] + in[15];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int32_t) hadd(avx_simd_int in) {
        return in[0] + in[1] + in[2] + in[3] + in[4] + in[5] + in[6] + in[7];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    ETL_STATIC_INLINE(int64_t) hadd(avx_simd_long in) {
        return in[0] + in[1] + in[2] + in[3];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template<typename T>
    ETL_STATIC_INLINE(T) hadd(avx_simd_complex_float<T> in) {
        return in[0] + in[1] + in[2] + in[3];
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template<typename T>
    ETL_STATIC_INLINE(T) hadd(avx_simd_complex_double<T> in) {
        return in[0] + in[1];
    }
};

#ifdef __AVX2__
/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_byte) avx_vec::zero<int8_t>() {
    return _mm256_setzero_si256();
}

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_short) avx_vec::zero<int16_t>() {
    return _mm256_setzero_si256();
}

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_int) avx_vec::zero<int32_t>() {
    return _mm256_setzero_si256();
}

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_long) avx_vec::zero<int64_t>() {
    return _mm256_setzero_si256();
}
#endif

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_float) avx_vec::zero<float>() {
    return _mm256_setzero_ps();
}

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_double) avx_vec::zero<double>() {
    return _mm256_setzero_pd();
}

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_complex_float<etl::complex<float>>) avx_vec::zero<etl::complex<float>>() {
    return _mm256_setzero_ps();
}

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_complex_double<etl::complex<double>>) avx_vec::zero<etl::complex<double>>() {
    return _mm256_setzero_pd();
}

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_complex_float<std::complex<float>>) avx_vec::zero<std::complex<float>>() {
    return _mm256_setzero_ps();
}

/*!
 * \copydoc avx_vec::zero
 */
template<>
ETL_OUT_INLINE(avx_simd_complex_double<std::complex<double>>) avx_vec::zero<std::complex<double>>() {
    return _mm256_setzero_pd();
}

} //end of namespace etl

#endif //__AVX__
