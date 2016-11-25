//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <immintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>

namespace etl {

namespace impl {

namespace avx {

namespace detail {

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
ETL_INLINE(float) mm256_hadd_ss(__m256 in) {
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
ETL_INLINE(double) mm256_hadd_sd(__m256d in) {
    const __m256d t1 = _mm256_hadd_pd(in, _mm256_permute2f128_pd(in, in, 1));
    const __m256d t2 = _mm256_hadd_pd(t1, t1);
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(t2));
}

} // end of namespace detail
} //end of namespace avx
} //end of namespace impl
} //end of namespace etl
