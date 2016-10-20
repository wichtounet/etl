//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief AVX implementation of the "dot" reduction
 */

#pragma once

#include "common.hpp"

namespace etl {

namespace impl {

namespace avx {

namespace detail {

inline float dot_kernel(float* a, float* b, size_t n){
    static constexpr size_t vec_size = 8;

    //TODO We should make sure it is aligned and do aligned loads
    //It will only be unaligned with custom vectors

    size_t i = 0;

    __m256 r1 = _mm256_setzero_ps();
    __m256 r2 = _mm256_setzero_ps();
    __m256 r3 = _mm256_setzero_ps();
    __m256 r4 = _mm256_setzero_ps();

    if (n < 1000000) {
        for (; i + (vec_size * 4) - 1 < n; i += 4 * vec_size) {
            __m256 a1 = _mm256_loadu_ps(a + i + vec_size * 0);
            __m256 a2 = _mm256_loadu_ps(a + i + vec_size * 1);
            __m256 a3 = _mm256_loadu_ps(a + i + vec_size * 2);
            __m256 a4 = _mm256_loadu_ps(a + i + vec_size * 3);

            __m256 b1 = _mm256_loadu_ps(b + i + vec_size * 0);
            __m256 b2 = _mm256_loadu_ps(b + i + vec_size * 1);
            __m256 b3 = _mm256_loadu_ps(b + i + vec_size * 2);
            __m256 b4 = _mm256_loadu_ps(b + i + vec_size * 3);

            __m256 t1 = _mm256_mul_ps(a1, b1);
            __m256 t2 = _mm256_mul_ps(a2, b2);
            __m256 t3 = _mm256_mul_ps(a3, b3);
            __m256 t4 = _mm256_mul_ps(a4, b4);

            r1 = _mm256_add_ps(r1, t1);
            r2 = _mm256_add_ps(r2, t2);
            r3 = _mm256_add_ps(r3, t3);
            r4 = _mm256_add_ps(r4, t4);
        }
    }

    for(; i + vec_size - 1 < n; i += vec_size){
        __m256 a1 = _mm256_loadu_ps(a + i);
        __m256 b1 = _mm256_loadu_ps(b + i);

        __m256 t1 = _mm256_mul_ps(a1, b1);

        r1 = _mm256_add_ps(r1, t1);
    }

    auto product = mm256_hadd_ss(r1) + mm256_hadd_ss(r2) + mm256_hadd_ss(r3) + mm256_hadd_ss(r4);

    for(; i < n; ++i){
        product += a[i] * b[i];
    }

    return product;
}

inline double dot_kernel(double* a, double* b, size_t n){
    static constexpr size_t vec_size = 4;

    size_t i = 0;

    __m256d r1 = _mm256_setzero_pd();

    for(; i + vec_size - 1 < n; i += vec_size){
        __m256d a1 = _mm256_loadu_pd(a + i);
        __m256d b1 = _mm256_loadu_pd(b + i);

        __m256d t1 = _mm256_mul_pd(a1, b1);

        r1 = _mm256_add_pd(r1, t1);
    }

    auto product = mm256_hadd_sd(r1);

    for(; i < n; ++i){
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

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
