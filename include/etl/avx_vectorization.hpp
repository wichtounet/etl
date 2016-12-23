//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
 * \brief Define traits to get vectorization information for types in AVX vector mode.
 */
template <typename T>
struct avx_intrinsic_traits {
    static constexpr bool vectorizable     = false;      ///< Boolean flag indicating if the type is vectorizable or not
    static constexpr std::size_t size      = 1;          ///< Numbers of elements done at once
    static constexpr std::size_t alignment = alignof(T); ///< Necessary number of bytes of alignment for this type

    using intrinsic_type = T; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<float> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 8; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m256; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<double> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 4; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m256d; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<std::complex<float>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 4; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m256; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<std::complex<double>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 2; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m256d; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<etl::complex<float>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 4; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m256; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<etl::complex<double>> {
    static constexpr bool vectorizable     = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 2; ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 32; ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m256d; ///< The vector type
};

/*!
 * \copydoc avx_intrinsic_traits
 */
template <>
struct avx_intrinsic_traits<int> {
    static constexpr bool vectorizable     = false; ///< Boolean flag indicating is vectorizable or not
    static constexpr std::size_t size      = 8;     ///< Numbers of elements in a vector
    static constexpr std::size_t alignment = 32;    ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m256i; ///< The vector type
};

/*!
 * \brief Advanced Vector eXtensions (AVX) operations implementation.
 */
struct avx_vec {
    template <typename T>
    using traits = avx_intrinsic_traits<T>; ///< The traits for this vector implementation

    template <typename T>
    using vec_type = typename traits<T>::intrinsic_type; ///< The vector type for the given vector type for this vector implementation

#ifdef VEC_DEBUG

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
    ETL_INLINE_VEC_VOID storeu(float* memory, __m256 value) {
        _mm256_storeu_ps(memory, value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(double* memory, __m256d value) {
        _mm256_storeu_pd(memory, value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(std::complex<float>* memory, __m256 value) {
        _mm256_storeu_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(std::complex<double>* memory, __m256d value) {
        _mm256_storeu_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(etl::complex<float>* memory, __m256 value) {
        _mm256_storeu_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(etl::complex<double>* memory, __m256d value) {
        _mm256_storeu_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(float* memory, __m256 value) {
        _mm256_stream_ps(memory, value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(double* memory, __m256d value) {
        _mm256_stream_pd(memory, value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(std::complex<float>* memory, __m256 value) {
        _mm256_stream_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(std::complex<double>* memory, __m256d value) {
        _mm256_stream_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(etl::complex<float>* memory, __m256 value) {
        _mm256_stream_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(etl::complex<double>* memory, __m256d value) {
        _mm256_stream_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(float* memory, __m256 value) {
        _mm256_store_ps(memory, value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(double* memory, __m256d value) {
        _mm256_store_pd(memory, value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(std::complex<float>* memory, __m256 value) {
        _mm256_store_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(std::complex<double>* memory, __m256d value) {
        _mm256_store_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(etl::complex<float>* memory, __m256 value) {
        _mm256_store_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(etl::complex<double>* memory, __m256d value) {
        _mm256_store_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Return a packed vector of zeroes of the given type
     */
    template<typename T>
    ETL_TMP_INLINE(typename avx_intrinsic_traits<T>::intrinsic_type) zero();

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_256 load(const float* memory) {
        return _mm256_load_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_256D load(const double* memory) {
        return _mm256_load_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_256 load(const std::complex<float>* memory) {
        return _mm256_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_256D load(const std::complex<double>* memory) {
        return _mm256_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_256 load(const etl::complex<float>* memory) {
        return _mm256_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_256D load(const etl::complex<double>* memory) {
        return _mm256_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_256 loadu(const float* memory) {
        return _mm256_loadu_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_256D loadu(const double* memory) {
        return _mm256_loadu_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_256 loadu(const std::complex<float>* memory) {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_256D loadu(const std::complex<double>* memory) {
        return _mm256_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_256 loadu(const etl::complex<float>* memory) {
        return _mm256_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_256D loadu(const etl::complex<double>* memory) {
        return _mm256_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_256D set(double value) {
        return _mm256_set1_pd(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_256 set(float value) {
        return _mm256_set1_ps(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_256 set(std::complex<float> value) {
        std::complex<float> tmp[]{value, value, value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_256D set(std::complex<double> value) {
        std::complex<double> tmp[]{value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_256 set(etl::complex<float> value) {
        etl::complex<float> tmp[]{value, value, value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_256D set(etl::complex<double> value) {
        etl::complex<double> tmp[]{value, value};
        return loadu(tmp);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_INLINE_VEC_256D add(__m256d lhs, __m256d rhs) {
        return _mm256_add_pd(lhs, rhs);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_INLINE_VEC_256D sub(__m256d lhs, __m256d rhs) {
        return _mm256_sub_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_INLINE_VEC_256D sqrt(__m256d x) {
        return _mm256_sqrt_pd(x);
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_INLINE_VEC_256D minus(__m256d x) {
        return _mm256_xor_pd(x, _mm256_set1_pd(-0.f));
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_INLINE_VEC_256 add(__m256 lhs, __m256 rhs) {
        return _mm256_add_ps(lhs, rhs);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_INLINE_VEC_256 sub(__m256 lhs, __m256 rhs) {
        return _mm256_sub_ps(lhs, rhs);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_INLINE_VEC_256 sqrt(__m256 lhs) {
        return _mm256_sqrt_ps(lhs);
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_INLINE_VEC_256 minus(__m256 x) {
        return _mm256_xor_ps(x, _mm256_set1_ps(-0.f));
    }

    /*!
     * \brief Multiply the two given vectors
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m256) mul(__m256 lhs, __m256 rhs) {
        return _mm256_mul_ps(lhs, rhs);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m256d) mul(__m256d lhs, __m256d rhs) {
        return _mm256_mul_pd(lhs, rhs);
    }

    /*!
     * \brief Fused-Multiply-Add of a b and c (r = (a * b) + c)
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m256) fmadd(__m256 a, __m256 b, __m256 c);

    /*!
     * \brief Fused-Multiply-Add of a b and c (r = (a * b) + c)
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m256d) fmadd(__m256d a, __m256d b, __m256d c);

    /*!
     * \brief Divide the two given vectors
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m256) div(__m256 lhs, __m256 rhs) {
        return _mm256_div_ps(lhs, rhs);
    }

    /*!
     * \brief Divide the two given vectors
     */
    template <bool Complex = false>
    ETL_TMP_INLINE(__m256d) div(__m256d lhs, __m256d rhs) {
        return _mm256_div_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the cosinus of each element of the given vector
     */
    ETL_INLINE_VEC_256 cos(__m256 x) {
        return etl::cos256_ps(x);
    }

    /*!
     * \brief Compute the sinus of each element of the given vector
     */
    ETL_INLINE_VEC_256 sin(__m256 x) {
        return etl::sin256_ps(x);
    }

#ifndef __INTEL_COMPILER

    //Exponential

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_256D exp(__m256d x) {
        return etl::exp256_pd(x);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_256 exp(__m256 x) {
        return etl::exp256_ps(x);
    }

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_INLINE_VEC_256 log(__m256 x) {
        return etl::log256_ps(x);
    }

#else //__INTEL_COMPILER

    //Exponential

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_256D exp(__m256d x) {
        return _mm256_exp_pd(x);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_256 exp(__m256 x) {
        return _mm256_exp_ps(x);
    }

    //Logarithm

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_INLINE_VEC_256D log(__m256d x) {
        return _mm256_log_pd(x);
    }

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_INLINE_VEC_256 log(__m256 x) {
        return _mm256_log_ps(x);
    }

#endif //__INTEL_COMPILER

    //Min

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_256D min(__m256d lhs, __m256d rhs) {
        return _mm256_min_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_256 min(__m256 lhs, __m256 rhs) {
        return _mm256_min_ps(lhs, rhs);
    }

    //Max

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_256D max(__m256d lhs, __m256d rhs) {
        return _mm256_max_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_256 max(__m256 lhs, __m256 rhs) {
        return _mm256_max_ps(lhs, rhs);
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template <typename T = float>
    static inline T ETL_INLINE_ATTR_VEC hadd(__m256 in) {
        const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(in, 1), _mm256_castps256_ps128(in));
        const __m128 x64  = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
        const __m128 x32  = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
        return _mm_cvtss_f32(x32);
    }

    /*!
     * \brief Perform an horizontal sum of the given vector.
     * \param in The input vector type
     * \return the horizontal sum of the vector
     */
    template <typename T = double>
    static inline T ETL_INLINE_ATTR_VEC hadd(__m256d in) {
        const __m256d t1 = _mm256_hadd_pd(in, _mm256_permute2f128_pd(in, in, 1));
        const __m256d t2 = _mm256_hadd_pd(t1, t1);
        return _mm_cvtsd_f64(_mm256_castpd256_pd128(t2));
    }
};

//TODO Vectorize the two following functions

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
template <>
inline std::complex<float> ETL_INLINE_ATTR_VEC avx_vec::hadd<std::complex<float>>(__m256 in) {
    std::complex<float> tmp_result[4];
    avx_vec::storeu(tmp_result, in);
    return tmp_result[0] + tmp_result[1] + tmp_result[2] + tmp_result[3];
}

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
template <>
inline std::complex<double> ETL_INLINE_ATTR_VEC avx_vec::hadd<std::complex<double>>(__m256d in) {
    std::complex<double> tmp_result[2];
    avx_vec::storeu(tmp_result, in);
    return tmp_result[0] + tmp_result[1];
}

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
template <>
inline etl::complex<float> ETL_INLINE_ATTR_VEC avx_vec::hadd<etl::complex<float>>(__m256 in) {
    etl::complex<float> tmp_result[4];
    avx_vec::storeu(tmp_result, in);
    return tmp_result[0] + tmp_result[1] + tmp_result[2] + tmp_result[3];
}

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
template <>
inline etl::complex<double> ETL_INLINE_ATTR_VEC avx_vec::hadd<etl::complex<double>>(__m256d in) {
    etl::complex<double> tmp_result[2];
    avx_vec::storeu(tmp_result, in);
    return tmp_result[0] + tmp_result[1];
}

/*!
 * \copydoc sse_vec::mul
 */
template <>
ETL_OUT_VEC_256 avx_vec::mul<true>(__m256 lhs, __m256 rhs) {
    //lhs = [x1.real, x1.img, x2.real, x2.img, ...]
    //rhs = [y1.real, y1.img, y2.real, y2.img, ...]

    //ymm1 = [y1.real, y1.real, y2.real, y2.real, ...]
    __m256 ymm1 = _mm256_moveldup_ps(rhs);

    //ymm2 = [x1.img, x1.real, x2.img, x2.real]
    __m256 ymm2 = _mm256_permute_ps(lhs, 0b10110001);

    //ymm3 = [y1.imag, y1.imag, y2.imag, y2.imag]
    __m256 ymm3 = _mm256_movehdup_ps(rhs);

    //ymm4 = ymm2 * ymm3
    __m256 ymm4 = _mm256_mul_ps(ymm2, ymm3);

//result = [(lhs * ymm1) -+ ymm4];

#ifdef __FMA__
    return _mm256_fmaddsub_ps(lhs, ymm1, ymm4);
#elif defined(__FMA4__)
    return _mm256_maddsub_ps(lhs, ymm1, ymm4);
#else
    __m256 tmp = _mm256_mul_ps(lhs, ymm1);
    return _mm256_addsub_ps(tmp, ymm4);
#endif
}

/*!
 * \copydoc sse_vec::mul
 */
template <>
ETL_OUT_VEC_256D avx_vec::mul<true>(__m256d lhs, __m256d rhs) {
    //lhs = [x1.real, x1.img, x2.real, x2.img]
    //rhs = [y1.real, y1.img, y2.real, y2.img]

    //ymm1 = [y1.real, y1.real, y2.real, y2.real]
    __m256d ymm1 = _mm256_movedup_pd(rhs);

    //ymm2 = [x1.img, x1.real, x2.img, x2.real]
    __m256d ymm2 = _mm256_permute_pd(lhs, 0b0101);

    //ymm3 = [y1.imag, y1.imag, y2.imag, y2.imag]
    __m256d ymm3 = _mm256_permute_pd(rhs, 0b1111);

    //ymm4 = ymm2 * ymm3
    __m256d ymm4 = _mm256_mul_pd(ymm2, ymm3);

//result = [(lhs * ymm1) -+ ymm4];

#ifdef __FMA__
    return _mm256_fmaddsub_pd(lhs, ymm1, ymm4);
#elif defined(__FMA4__)
    return _mm256_maddsub_pd(lhs, ymm1, ymm4);
#else
    __m256d tmp = _mm256_mul_pd(lhs, ymm1);
    return _mm256_addsub_pd(tmp, ymm4);
#endif
}

/*!
 * \copydoc sse_vec::fmadd
 */
template <>
ETL_OUT_VEC_256 avx_vec::fmadd<false>(__m256 a, __m256 b, __m256 c) {
#ifdef __FMA__
    return _mm256_fmadd_ps(a, b, c);
#else
    return add(mul<false>(a, b), c);
#endif
}

/*!
 * \copydoc sse_vec::fmadd
 */
template <>
ETL_OUT_VEC_256D avx_vec::fmadd<false>(__m256d a, __m256d b, __m256d c) {
#ifdef __FMA__
    return _mm256_fmadd_pd(a, b, c);
#else
    return add(mul<false>(a, b), c);
#endif
}

/*!
 * \copydoc sse_vec::fmadd
 */
template <>
ETL_OUT_VEC_256 avx_vec::fmadd<true>(__m256 a, __m256 b, __m256 c) {
    return add(mul<true>(a, b), c);
}

/*!
 * \copydoc sse_vec::fmadd
 */
template <>
ETL_OUT_VEC_256D avx_vec::fmadd<true>(__m256d a, __m256d b, __m256d c) {
    return add(mul<true>(a, b), c);
}

/*!
 * \copydoc sse_vec::div
 */
template <>
ETL_OUT_VEC_256 avx_vec::div<true>(__m256 lhs, __m256 rhs) {
    //lhs = [x1.real, x1.img, x2.real, x2.img ...]
    //rhs = [y1.real, y1.img, y2.real, y2.img ...]

    //ymm0 = [y1.real, y1.real, y2.real, y2.real, ...]
    __m256 ymm0 = _mm256_moveldup_ps(rhs);

    //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
    __m256 ymm1 = _mm256_movehdup_ps(rhs);

    //ymm2 = [x1.img, x1.real, x2.img, x2.real]
    __m256 ymm2 = _mm256_permute_ps(lhs, 0b10110001);

    //ymm4 = [x.img * y.img, x.real * y.img]
    __m256 ymm4 = _mm256_mul_ps(ymm2, ymm1);

//ymm5 = subadd((lhs * ymm0), ymm4)

#ifdef __FMA__
    __m256 ymm5 = _mm256_fmsubadd_ps(lhs, ymm0, ymm4);
#else
    __m256 t1    = _mm256_mul_ps(lhs, ymm0);
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
 * \copydoc sse_vec::div
 */
template <>
ETL_OUT_VEC_256D avx_vec::div<true>(__m256d lhs, __m256d rhs) {
    //lhs = [x1.real, x1.img, x2.real, x2.img]
    //rhs = [y1.real, y1.img, y2.real, y2.img]

    //ymm0 = [y1.real, y1.real, y2.real, y2.real]
    __m256d ymm0 = _mm256_movedup_pd(rhs);

    //ymm1 = [y1.imag, y1.imag, y2.imag, y2.imag]
    __m256d ymm1 = _mm256_permute_pd(rhs, 0b1111);

    //ymm2 = [x1.img, x1.real, x2.img, x2.real]
    __m256d ymm2 = _mm256_permute_pd(lhs, 0b0101);

    //ymm4 = [x.img * y.img, x.real * y.img]
    __m256d ymm4 = _mm256_mul_pd(ymm2, ymm1);

//ymm5 = subadd((lhs * ymm0), ymm4)

#ifdef __FMA__
    __m256d ymm5 = _mm256_fmsubadd_pd(lhs, ymm0, ymm4);
#else
    __m256d t1   = _mm256_mul_pd(lhs, ymm0);
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

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_256 avx_vec::zero<float>() {
    return _mm256_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_256D avx_vec::zero<double>() {
    return _mm256_setzero_pd();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_256 avx_vec::zero<etl::complex<float>>() {
    return _mm256_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_256D avx_vec::zero<etl::complex<double>>() {
    return _mm256_setzero_pd();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_256 avx_vec::zero<std::complex<float>>() {
    return _mm256_setzero_ps();
}

/*!
 * \copydoc sse_vec::zero
 */
template<>
ETL_OUT_VEC_256D avx_vec::zero<std::complex<double>>() {
    return _mm256_setzero_pd();
}

} //end of namespace etl

#endif //__AVX__
