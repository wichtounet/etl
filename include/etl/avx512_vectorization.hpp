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

    using intrinsic_type = __m512; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<double> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 8;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m512d; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<std::complex<float>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 8;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m512; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<std::complex<double>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 4;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m512d; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<etl::complex<float>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 8;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m512; ///< The vector type
};

/*!
 * \copydoc avx512_intrinsic_traits
 */
template <>
struct avx512_intrinsic_traits<etl::complex<double>> {
    static constexpr bool vectorizable = true; ///< Boolean flag indicating is vectorizable or not
    static constexpr size_t size       = 4;    ///< Numbers of elements in a vector
    static constexpr size_t alignment  = 64;   ///< Necessary alignment, in bytes, for this type

    using intrinsic_type = __m512d; ///< The vector type
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
    ETL_INLINE_VEC_VOID storeu(float* memory, __m512 value) {
        _mm512_storeu_ps(memory, value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(double* memory, __m512d value) {
        _mm512_storeu_pd(memory, value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(std::complex<float>* memory, __m512 value) {
        _mm512_storeu_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(std::complex<double>* memory, __m512d value) {
        _mm512_storeu_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(etl::complex<float>* memory, __m512 value) {
        _mm512_storeu_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Unaligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID storeu(etl::complex<double>* memory, __m512d value) {
        _mm512_storeu_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(float* memory, __m512 value) {
        _mm512_store_ps(memory, value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(double* memory, __m512d value) {
        _mm512_store_pd(memory, value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(std::complex<float>* memory, __m512 value) {
        _mm512_store_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(std::complex<double>* memory, __m512d value) {
        _mm512_store_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(etl::complex<float>* memory, __m512 value) {
        _mm512_store_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Aligned store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID store(etl::complex<double>* memory, __m512d value) {
        _mm512_store_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(float* memory, __m512 value) {
        _mm512_stream_ps(memory, value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(double* memory, __m512d value) {
        _mm512_stream_pd(memory, value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(std::complex<float>* memory, __m512 value) {
        _mm512_stream_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(std::complex<double>* memory, __m512d value) {
        _mm512_stream_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(etl::complex<float>* memory, __m512 value) {
        _mm512_stream_ps(reinterpret_cast<float*>(memory), value);
    }

    /*!
     * \brief Non-temporal, aligned, store of the given packed vector at the
     * given memory position
     */
    ETL_INLINE_VEC_VOID stream(etl::complex<double>* memory, __m512d value) {
        _mm512_stream_pd(reinterpret_cast<double*>(memory), value);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_512 load(const float* memory) {
        return _mm512_load_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_512D load(const double* memory) {
        return _mm512_load_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_512 load(const std::complex<float>* memory) {
        return _mm512_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_512D load(const std::complex<double>* memory) {
        return _mm512_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_512 load(const etl::complex<float>* memory) {
        return _mm512_load_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given aligned memory location
     */
    ETL_INLINE_VEC_512D load(const etl::complex<double>* memory) {
        return _mm512_load_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_512 loadu(const float* memory) {
        return _mm512_loadu_ps(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_512D loadu(const double* memory) {
        return _mm512_loadu_pd(memory);
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_512 loadu(const std::complex<float>* memory) {
        return _mm512_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_512D loadu(const std::complex<double>* memory) {
        return _mm512_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_512 loadu(const etl::complex<float>* memory) {
        return _mm512_loadu_ps(reinterpret_cast<const float*>(memory));
    }

    /*!
     * \brief Load a packed vector from the given unaligned memory location
     */
    ETL_INLINE_VEC_512D loadu(const etl::complex<double>* memory) {
        return _mm512_loadu_pd(reinterpret_cast<const double*>(memory));
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_512D set(double value) {
        return _mm512_set1_pd(value);
    }

    /*!
     * \brief Fill a packed vector  by replicating a value
     */
    ETL_INLINE_VEC_512 set(float value) {
        return _mm512_set1_ps(value);
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_INLINE_VEC_512D add(__m512d lhs, __m512d rhs) {
        return _mm512_add_pd(lhs, rhs);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_INLINE_VEC_512D sub(__m512d lhs, __m512d rhs) {
        return _mm512_sub_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_INLINE_VEC_512D sqrt(__m512d x) {
        return _mm512_sqrt_pd(x);
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_INLINE_VEC_512D minus(__m512d x) {
        return _mm512_xor_pd(x, _mm512_set1_pd(-0.f));
    }

    /*!
     * \brief Add the two given values and return the result.
     */
    ETL_INLINE_VEC_512 add(__m512 lhs, __m512 rhs) {
        return _mm512_add_ps(lhs, rhs);
    }

    /*!
     * \brief Subtract the two given values and return the result.
     */
    ETL_INLINE_VEC_512 sub(__m512 lhs, __m512 rhs) {
        return _mm512_sub_ps(lhs, rhs);
    }

    /*!
     * \brief Compute the square root of each element in the given vector
     * \return a vector containing the square root of each input element
     */
    ETL_INLINE_VEC_512 sqrt(__m512 lhs) {
        return _mm512_sqrt_ps(lhs);
    }

    /*!
     * \brief Compute the negative of each element in the given vector
     * \return a vector containing the negative of each input element
     */
    ETL_INLINE_VEC_512 minus(__m512 x) {
        return _mm512_xor_ps(x, _mm512_set1_ps(-0.f));
    }

    /*!
     * \brief Multiply the two given vectors
     */
    template <bool Complex = false>
    ETL_INLINE_VEC_512 mul(__m512 lhs, __m512 rhs) {
        return _mm512_mul_ps(lhs, rhs);
    }

    /*!
     * \brief Multiply the two given vectors
     */
    template <>
    ETL_INLINE_VEC_512 mul<true>(__m512 lhs, __m512 rhs) {
        cpp_unreachable("Not yet implemented");
        return lhs;
    }

    /*!
     * \brief Multiply the two given complex vectors
     */
    template <bool Complex = false>
    ETL_INLINE_VEC_512D mul(__m512d lhs, __m512d rhs) {
        return _mm512_mul_pd(lhs, rhs);
    }

    /*!
     * \brief Multiply the two given complex vectors
     */
    template <>
    ETL_INLINE_VEC_512D mul<true>(__m512d lhs, __m512d rhs) {
        cpp_unreachable("Not yet implemented");
        return lhs;
    }

    /*!
     * \brief Divide the two given vectors
     */
    template <bool Complex = false>
    ETL_INLINE_VEC_512 div(__m512 lhs, __m512 rhs) {
        return _mm512_div_ps(lhs, rhs);
    }

    /*!
     * \brief Divide the two given vectors
     */
    template <>
    ETL_INLINE_VEC_512 div<true>(__m512 lhs, __m512 rhs) {
        cpp_unreachable("Not yet implemented");
        return lhs;
    }

    /*!
     * \brief Divide the two given vectors
     */
    template <bool Complex = false>
    ETL_INLINE_VEC_512D div(__m512d lhs, __m512d rhs) {
        return _mm512_div_pd(lhs, rhs);
    }

    /*!
     * \brief Divide the two given vectors
     */
    template <>
    ETL_INLINE_VEC_512D div<true>(__m512d lhs, __m512d rhs) {
        cpp_unreachable("Not yet implemented");
        return lhs;
    }

#ifdef __INTEL_COMPILER

    //Exponential

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_512D exp(__m512d x) {
        return _mm512_exp_pd(x);
    }

    /*!
     * \brief Compute the exponentials of each element of the given vector
     */
    ETL_INLINE_VEC_512 exp(__m512 x) {
        return _mm512_exp_ps(x);
    }

    //Logarithm

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_INLINE_VEC_512D log(__m512d x) {
        return _mm512_log_pd(x);
    }

    /*!
     * \brief Compute the logarithm of each element of the given vector
     */
    ETL_INLINE_VEC_512 log(__m512 x) {
        return _mm512_log_ps(x);
    }

    //Min

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_512D min(__m512d lhs, __m512d rhs) {
        return _mm512_min_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the minimum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_512 min(__m512 lhs, __m512 rhs) {
        return _mm512_min_ps(lhs, rhs);
    }

    //Max

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_512D max(__m512d lhs, __m512d rhs) {
        return _mm512_max_pd(lhs, rhs);
    }

    /*!
     * \brief Compute the maximum between each pair element of the given vectors
     */
    ETL_INLINE_VEC_512 max(__m512 lhs, __m512 rhs) {
        return _mm512_max_ps(lhs, rhs);
    }

#endif //__INTEL_COMPILER
};

/*!
 * \copydoc sse_vec::mul
 */
template <>
ETL_OUT_VEC_512 avx512_vec::mul<true>(__m512 lhs, __m512 rhs) {
    cpp_unreachable("Not yet implemented");
    return lhs;
}

/*!
 * \copydoc sse_vec::mul
 */
template <>
ETL_OUT_VEC_512D avx512_vec::mul<true>(__m512d lhs, __m512d rhs) {
    cpp_unreachable("Not yet implemented");
    return lhs;
}

/*!
 * \copydoc sse_vec::div
 */
template <>
ETL_OUT_VEC_512 avx512_vec::div<true>(__m512 lhs, __m512 rhs) {
    cpp_unreachable("Not yet implemented");
    return lhs;
}

/*!
 * \copydoc sse_vec::div
 */
template <>
ETL_OUT_VEC_512D avx512_vec::div<true>(__m512d lhs, __m512d rhs) {
    cpp_unreachable("Not yet implemented");
    return lhs;
}

} //end of namespace etl

#endif //__AVX512F__
