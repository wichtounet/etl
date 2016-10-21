//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief SSE implementation of the "dot" reduction
 */

#pragma once

#if defined(ETL_VECTORIZE_IMPL) && defined(__SSE3__)

#include "common.hpp"

#endif

namespace etl {

namespace impl {

namespace sse {

#if defined(ETL_VECTORIZE_IMPL) && defined(__SSE3__)

namespace detail {

inline float dot_kernel(float* a, float* b, size_t n){
    __m128 r1 = _mm_setzero_ps();
    for(size_t i = 0; i + 3 < n; i += 4){
        __m128 a1 = _mm_loadu_ps(a + i);
        __m128 b1 = _mm_loadu_ps(b + i);

        __m128 t1 = _mm_mul_ps(a1, b1);

        r1 = _mm_add_ps(r1, t1);
    }

    auto product = mm_hadd_ss(r1);

    for(size_t i = n - n % 4; i < n; ++i){
        product += a[i] * b[i];
    }

    return product;
}

inline double dot_kernel(double* a, double* b, size_t n){
    __m128d r1 = _mm_setzero_pd();

    for(size_t i = 0; i + 1 < n; i += 2){
        __m128d a1 = _mm_loadu_pd(a + i);
        __m128d b1 = _mm_loadu_pd(b + i);

        __m128d t1 = _mm_mul_pd(a1, b1);

        r1 = _mm_add_pd(r1, t1);
    }

    auto product = mm_hadd_sd(r1);

    for(size_t i = n - n % 2; i < n; ++i){
        product += a[i] * b[i];
    }

    return product;
}

} // end of namespace detail

/*!
 * \brief Compute the dot product of a and b
 * \param a The lhs expression
 * \param b The rhs expression
 * \return the sum
 */
template <typename T>
T dot(const opaque_memory<T, 1>& a, const opaque_memory<T, 1>& b) {
    auto a_mem = a.memory_start();
    auto b_mem = b.memory_start();

    return detail::dot_kernel(a_mem, b_mem, a.size());
}

#else

/*!
 * \brief Compute the dot product of a and b
 * \param a The lhs expression
 * \param b The rhs expression
 * \return the sum
 */
template <typename T>
T dot(const opaque_memory<T, 1>& a, const opaque_memory<T, 1>& b) {
    cpp_unused(a);
    cpp_unused(b);
    cpp_unreachable("SSE not available/enabled");
}

#endif

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
