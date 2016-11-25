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

namespace sse {

namespace detail {

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
ETL_INLINE(double) mm_hadd_sd(__m128d in) {
    __m128 undef   = _mm_undefined_ps();
    __m128 shuftmp = _mm_movehl_ps(undef, _mm_castpd_ps(in));
    __m128d shuf = _mm_castps_pd(shuftmp);
    return _mm_cvtsd_f64(_mm_add_sd(in, shuf));
}

/*!
 * \brief Perform an horizontal sum of the given vector.
 * \param in The input vector type
 * \return the horizontal sum of the vector
 */
ETL_INLINE(float) mm_hadd_ss(__m128 in) {
    __m128 shuf = _mm_movehdup_ps(in);
    __m128 sums = _mm_add_ps(in, shuf);
    shuf        = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

} //end of namespace detail
} //end of namespace sse
} //end of namespace impl
} //end of namespace etl
