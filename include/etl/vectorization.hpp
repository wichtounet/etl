//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file vectorization.hpp
 * \brief Contains vectorization utilities for the vectorized assignments (done by the evaluator).
 *
 * This file automatically includes the correct header based on which vectorization utility is supported (AVX -> SSE -> NONE).
 */

#pragma once

#ifdef __ARM_ARCH
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif 


#include "etl/inline.hpp"

namespace etl {

/*!
 * \brief SIMD pack of some type, using a vector implementation type
 * \tparam V The vector implementation
 * \tparam T The real type
 * \tparam VT The vector type
 */
template <vector_mode_t V, typename T, typename VT>
struct simd_pack {
    using value_type     = T;  ///< The real value type
    using intrinsic_type = VT; ///< The used intrinsic type

    static constexpr vector_mode_t vector_mode = V; ///< The vector implementation mode

    intrinsic_type value; ///< The vector of value

    /*!
     * \brief Construct a new simd_pack around the given vector
     * \param value The vector value to build around
     */
    simd_pack(intrinsic_type value) : value(value) {
        // Nothing else to init
    }

    /*!
     * \brief Extract an element of the vector value
     * \param i The index of the element to get
     * \return The ith element from the vector
     */
    ETL_STRONG_INLINE(value_type) operator[](size_t i) const noexcept {
        return reinterpret_cast<const value_type*>(&value)[i];
    }
};

} // end of namespace etl

#if defined __GNUC__ && __GNUC__>=6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

//Include al the vector implementation
#include "etl/avx512_vectorization.hpp"
#include "etl/avx_vectorization.hpp"
#include "etl/sse_vectorization.hpp"
#include "etl/no_vectorization.hpp"

#if defined __GNUC__ && __GNUC__>=6
#pragma GCC diagnostic pop
#endif

namespace etl {

/*!
 * \brief Traits to get the intrinsic traits for a vector mode
 * \tparam V The vector mode
 */
template <vector_mode_t V>
struct get_intrinsic_traits {
    /*!
     * \brief The type of the intrinsic traits for T
     * \tparam T The type to get the intrinsic traits for
     */
    template <typename T>
    using type = no_intrinsic_traits<T>;
};

/*!
 * \brief Traits to get the vector implementation for a vector mode
 * \tparam V The vector mode
 */
template <vector_mode_t V>
struct get_vector_impl {
    using type = no_vec; ///< The vector implementation
};

#ifdef __AVX512F__

/*!
 * \brief get_intrinsic_traits for AVX-512
 */
template <>
struct get_intrinsic_traits<vector_mode_t::AVX512> {
    /*!
     * \brief The type of the intrinsic traits for T
     * \tparam T The type to get the intrinsic traits for
     */
    template <typename T>
    using type = avx512_intrinsic_traits<T>;
};

/*!
 * \copy get_vector_impl
 */
template <>
struct get_vector_impl<vector_mode_t::AVX512> {
    using type = avx512_vec; ///< The vector implementation
};

#endif

#ifdef __AVX__

/*!
 * \brief get_intrinsic_traits for AVX
 */
template <>
struct get_intrinsic_traits<vector_mode_t::AVX> {
    /*!
     * \brief The type of the intrinsic traits for T
     * \tparam T The type to get the intrinsic traits for
     */
    template <typename T>
    using type = avx_intrinsic_traits<T>;
};

/*!
 * \copy get_vector_impl
 */
template <>
struct get_vector_impl<vector_mode_t::AVX> {
    using type = avx_vec; ///< The vector implementation
};

#endif

#ifdef __SSE3__

/*!
 * \brief get_intrinsic_traits for SSE
 */
template <>
struct get_intrinsic_traits<vector_mode_t::SSE3> {
    /*!
     * \brief The type of the intrinsic traits for T
     * \tparam T The type to get the intrinsic traits for
     */
    template <typename T>
    using type = sse_intrinsic_traits<T>;
};

/*!
 * \copy get_vector_impl
 */
template <>
struct get_vector_impl<vector_mode_t::SSE3> {
    using type = sse_vec; ///< The vector implementation
};

#endif

#ifdef ETL_VECTORIZE_EXPR

#ifdef __AVX512F__

/*!
 * \brief The default vectorization scheme
 */
using default_vec = avx512_vec;

/*!
 * \brief The default intrinsic traits
 */
template <typename T>
using default_intrinsic_traits = avx512_intrinsic_traits<T>;

#elif defined(__AVX__)

/*!
 * \brief The default vectorization scheme
 */
using default_vec = avx_vec;

/*!
 * \brief The default intrinsic traits
 */
template <typename T>
using default_intrinsic_traits = avx_intrinsic_traits<T>;

#elif defined(__SSE3__)

/*!
 * \brief The default vectorization scheme
 */
using default_vec = sse_vec;

/*!
 * \brief The default intrinsic traits
 */
template <typename T>
using default_intrinsic_traits = sse_intrinsic_traits<T>;

#else

/*!
 * \brief The default vectorization scheme
 */
using default_vec = no_vec;

/*!
 * \brief The default intrinsic traits
 */
template <typename T>
using default_intrinsic_traits = no_intrinsic_traits<T>;

#endif //defined(__SSE__)

#else //ETL_VECTORIZE_EXPR

/*!
 * \brief The default vectorization scheme
 */
using default_vec = no_vec;

/*!
 * \brief The default intrinsic traits
 */
template <typename T>
using default_intrinsic_traits = no_intrinsic_traits<T>;

#endif //ETL_VECTORIZE_EXPR

/*!
 * \brief Helper to get the intrinsic corresponding type of a vectorizable type.
 */
template <typename T>
using default_intrinsic_type = typename default_intrinsic_traits<T>::intrinsic_type;

} //end of namespace etl
