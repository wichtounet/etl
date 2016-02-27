//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

// Most of the code has been taken from Julien Pommier and adapted
// for ETL

/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

/*!
 * \file
 * \brief Contains SSE functions for exp and log
 */

#pragma once

#ifdef __SSE3__

#include <immintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

#ifdef VECT_DEBUG
#include <iostream>
#endif

#ifdef __clang__
#define ETL_INLINE_VEC_VOID static inline void __attribute__((__always_inline__, __nodebug__))
#define ETL_INLINE_VEC_128 static inline __m128 __attribute__((__always_inline__, __nodebug__))
#define ETL_INLINE_VEC_128D static inline __m128d __attribute__((__always_inline__, __nodebug__))
#define ETL_OUT_VEC_128 inline __m128 __attribute__((__always_inline__, __nodebug__))
#define ETL_OUT_VEC_128D inline __m128d __attribute__((__always_inline__, __nodebug__))
#else
#define ETL_INLINE_VEC_VOID static inline void __attribute__((__always_inline__))
#define ETL_INLINE_VEC_128 static inline __m128 __attribute__((__always_inline__))
#define ETL_INLINE_VEC_128D static inline __m128d __attribute__((__always_inline__))
#define ETL_OUT_VEC_128 inline __m128 __attribute__((__always_inline__))
#define ETL_OUT_VEC_128D inline __m128d __attribute__((__always_inline__))
#endif

namespace etl {

# define ALIGN16_BEG
# define ALIGN16_END __attribute__((aligned(16)))

/* declare some SSE constants -- why can't I figure a better way to do that? */
#define PS_CONST(Name, Val)                                            \
  static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define PS_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

PS_CONST(1  , 1.0f);
PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
PS_CONST_TYPE(mant_mask, int, 0x7f800000);
PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

PS_CONST(cephes_SQRTHF, 0.707106781186547524);
PS_CONST(cephes_log_p0, 7.0376836292E-2);
PS_CONST(cephes_log_p1, - 1.1514610310E-1);
PS_CONST(cephes_log_p2, 1.1676998740E-1);
PS_CONST(cephes_log_p3, - 1.2420140846E-1);
PS_CONST(cephes_log_p4, + 1.4249322787E-1);
PS_CONST(cephes_log_p5, - 1.6668057665E-1);
PS_CONST(cephes_log_p6, + 2.0000714765E-1);
PS_CONST(cephes_log_p7, - 2.4999993993E-1);
PS_CONST(cephes_log_p8, + 3.3333331174E-1);
PS_CONST(cephes_log_q1, -2.12194440e-4);
PS_CONST(cephes_log_q2, 0.693359375);

/* natural logarithm computed for 4 simultaneous float
   return NaN for x <= 0
*/

ETL_INLINE_VEC_128 log_ps(__m128 x) {
  __m128i emm0;
  __m128 one = *(__m128*)_ps_1;

  __m128 invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

  x = _mm_max_ps(x, *(__m128*)_ps_min_norm_pos);  /* cut off denormalized stuff */

  emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

  /* keep only the fractional part */
  x = _mm_and_ps(x, *(__m128*)_ps_inv_mant_mask);
  x = _mm_or_ps(x, *(__m128*)_ps_0p5);

  emm0 = _mm_sub_epi32(emm0, *(__m128i*)_pi32_0x7f);
  __m128 e = _mm_cvtepi32_ps(emm0);

  e = _mm_add_ps(e, one);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  __m128 mask = _mm_cmplt_ps(x, *(__m128*)_ps_cephes_SQRTHF);
  __m128 tmp = _mm_and_ps(x, mask);
  x = _mm_sub_ps(x, one);
  e = _mm_sub_ps(e, _mm_and_ps(one, mask));
  x = _mm_add_ps(x, tmp);


  __m128 z = _mm_mul_ps(x,x);

  __m128 y = *(__m128*)_ps_cephes_log_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p5);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p6);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p7);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p8);
  y = _mm_mul_ps(y, x);

  y = _mm_mul_ps(y, z);


  tmp = _mm_mul_ps(e, *(__m128*)_ps_cephes_log_q1);
  y = _mm_add_ps(y, tmp);


  tmp = _mm_mul_ps(z, *(__m128*)_ps_0p5);
  y = _mm_sub_ps(y, tmp);

  tmp = _mm_mul_ps(e, *(__m128*)_ps_cephes_log_q2);
  x = _mm_add_ps(x, y);
  x = _mm_add_ps(x, tmp);
  x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
  return x;
}

PS_CONST(exp_hi, 88.3762626647949f);
PS_CONST(exp_lo, -88.3762626647949f);

PS_CONST(cephes_LOG2EF, 1.44269504088896341);
PS_CONST(cephes_exp_C1, 0.693359375);
PS_CONST(cephes_exp_C2, -2.12194440e-4);

PS_CONST(cephes_exp_p0, 1.9875691500E-4);
PS_CONST(cephes_exp_p1, 1.3981999507E-3);
PS_CONST(cephes_exp_p2, 8.3334519073E-3);
PS_CONST(cephes_exp_p3, 4.1665795894E-2);
PS_CONST(cephes_exp_p4, 1.6666665459E-1);
PS_CONST(cephes_exp_p5, 5.0000001201E-1);

ETL_INLINE_VEC_128 exp_ps(__m128 x) {
  __m128 tmp, fx;
  __m128i emm0;
  __m128 one = *(__m128*)_ps_1;

  x = _mm_min_ps(x, *(__m128*)_ps_exp_hi);
  x = _mm_max_ps(x, *(__m128*)_ps_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm_mul_ps(x, *(__m128*)_ps_cephes_LOG2EF);
  fx = _mm_add_ps(fx, *(__m128*)_ps_0p5);

  /* how to perform a floorf with SSE: just below */
  emm0 = _mm_cvttps_epi32(fx);
  tmp  = _mm_cvtepi32_ps(emm0);
  /* if greater, substract 1 */
  __m128 mask = _mm_cmpgt_ps(tmp, fx);
  mask = _mm_and_ps(mask, one);
  fx = _mm_sub_ps(tmp, mask);

  tmp = _mm_mul_ps(fx, *(__m128*)_ps_cephes_exp_C1);
  __m128 z = _mm_mul_ps(fx, *(__m128*)_ps_cephes_exp_C2);
  x = _mm_sub_ps(x, tmp);
  x = _mm_sub_ps(x, z);

  z = _mm_mul_ps(x,x);

  __m128 y = *(__m128*)_ps_cephes_exp_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_exp_p5);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, x);
  y = _mm_add_ps(y, one);

  /* build 2^n */
  emm0 = _mm_cvttps_epi32(fx);
  emm0 = _mm_add_epi32(emm0, *(__m128i*)_pi32_0x7f);
  emm0 = _mm_slli_epi32(emm0, 23);
  __m128 pow2n = _mm_castsi128_ps(emm0);
  y = _mm_mul_ps(y, pow2n);
  return y;
}

PS_CONST(minus_cephes_DP1, -0.78515625);
PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
PS_CONST(sincof_p0, -1.9515295891E-4);
PS_CONST(sincof_p1,  8.3321608736E-3);
PS_CONST(sincof_p2, -1.6666654611E-1);
PS_CONST(coscof_p0,  2.443315711809948E-005);
PS_CONST(coscof_p1, -1.388731625493765E-003);
PS_CONST(coscof_p2,  4.166664568298827E-002);
PS_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI


/* evaluation of 4 sines at onces, using only SSE1+MMX intrinsics so
   it runs also on old athlons XPs and the pentium III of your grand
   mother.

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

   Performance is also surprisingly good, 1.33 times faster than the
   macos vsinf SSE2 function, and 1.5 times faster than the
   __vrs4_sinf of amd's ACML (which is only available in 64 bits). Not
   too bad for an SSE1 function (with no special tuning) !
   However the latter libraries probably have a much better handling of NaN,
   Inf, denormalized and other special arguments..

   On my core 1 duo, the execution of this function takes approximately 95 cycles.

   From what I have observed on the experiments with Intel AMath lib, switching to an
   SSE2 version would improve the perf by only 10%.

   Since it is based on SSE intrinsics, it has to be compiled at -O2 to
   deliver full speed.
*/
ETL_INLINE_VEC_128 sin_ps(__m128 x) { // any x
  __m128 xmm1, xmm2, xmm3, sign_bit, y;

  __m128i emm0, emm2;
  sign_bit = x;
  /* take the absolute value */
  x = _mm_and_ps(x, *(__m128*)_ps_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit = _mm_and_ps(sign_bit, *(__m128*)_ps_sign_mask);

  /* scale by 4/Pi */
  y = _mm_mul_ps(x, *(__m128*)_ps_cephes_FOPI);

  /* store the integer part of y in mm0 */
  emm2 = _mm_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  emm2 = _mm_add_epi32(emm2, *(__m128i*)_pi32_1);
  emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_inv1);
  y = _mm_cvtepi32_ps(emm2);

  /* get the swap sign flag */
  emm0 = _mm_and_si128(emm2, *(__m128i*)_pi32_4);
  emm0 = _mm_slli_epi32(emm0, 29);
  /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_2);
  emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

  __m128 swap_sign_bit = _mm_castsi128_ps(emm0);
  __m128 poly_mask = _mm_castsi128_ps(emm2);
  sign_bit = _mm_xor_ps(sign_bit, swap_sign_bit);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(__m128*)_ps_minus_cephes_DP1;
  xmm2 = *(__m128*)_ps_minus_cephes_DP2;
  xmm3 = *(__m128*)_ps_minus_cephes_DP3;
  xmm1 = _mm_mul_ps(y, xmm1);
  xmm2 = _mm_mul_ps(y, xmm2);
  xmm3 = _mm_mul_ps(y, xmm3);
  x = _mm_add_ps(x, xmm1);
  x = _mm_add_ps(x, xmm2);
  x = _mm_add_ps(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(__m128*)_ps_coscof_p0;
  __m128 z = _mm_mul_ps(x,x);

  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, *(__m128*)_ps_coscof_p1);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, *(__m128*)_ps_coscof_p2);
  y = _mm_mul_ps(y, z);
  y = _mm_mul_ps(y, z);
  __m128 tmp = _mm_mul_ps(z, *(__m128*)_ps_0p5);
  y = _mm_sub_ps(y, tmp);
  y = _mm_add_ps(y, *(__m128*)_ps_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m128 y2 = *(__m128*)_ps_sincof_p0;
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, *(__m128*)_ps_sincof_p1);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, *(__m128*)_ps_sincof_p2);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_mul_ps(y2, x);
  y2 = _mm_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  y2 = _mm_and_ps(xmm3, y2); //, xmm3);
  y = _mm_andnot_ps(xmm3, y);
  y = _mm_add_ps(y,y2);
  /* update the sign */
  y = _mm_xor_ps(y, sign_bit);
  return y;
}

/* almost the same as sin_ps */
ETL_INLINE_VEC_128 cos_ps(__m128 x) { // any x
  __m128 xmm1, xmm2, xmm3, y;
  __m128i emm0, emm2;
  /* take the absolute value */
  x = _mm_and_ps(x, *(__m128*)_ps_inv_sign_mask);

  /* scale by 4/Pi */
  y = _mm_mul_ps(x, *(__m128*)_ps_cephes_FOPI);

  /* store the integer part of y in mm0 */
  emm2 = _mm_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  emm2 = _mm_add_epi32(emm2, *(__m128i*)_pi32_1);
  emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_inv1);
  y = _mm_cvtepi32_ps(emm2);

  emm2 = _mm_sub_epi32(emm2, *(__m128i*)_pi32_2);

  /* get the swap sign flag */
  emm0 = _mm_andnot_si128(emm2, *(__m128i*)_pi32_4);
  emm0 = _mm_slli_epi32(emm0, 29);
  /* get the polynom selection mask */
  emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_2);
  emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

  __m128 sign_bit = _mm_castsi128_ps(emm0);
  __m128 poly_mask = _mm_castsi128_ps(emm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(__m128*)_ps_minus_cephes_DP1;
  xmm2 = *(__m128*)_ps_minus_cephes_DP2;
  xmm3 = *(__m128*)_ps_minus_cephes_DP3;
  xmm1 = _mm_mul_ps(y, xmm1);
  xmm2 = _mm_mul_ps(y, xmm2);
  xmm3 = _mm_mul_ps(y, xmm3);
  x = _mm_add_ps(x, xmm1);
  x = _mm_add_ps(x, xmm2);
  x = _mm_add_ps(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(__m128*)_ps_coscof_p0;
  __m128 z = _mm_mul_ps(x,x);

  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, *(__m128*)_ps_coscof_p1);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, *(__m128*)_ps_coscof_p2);
  y = _mm_mul_ps(y, z);
  y = _mm_mul_ps(y, z);
  __m128 tmp = _mm_mul_ps(z, *(__m128*)_ps_0p5);
  y = _mm_sub_ps(y, tmp);
  y = _mm_add_ps(y, *(__m128*)_ps_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m128 y2 = *(__m128*)_ps_sincof_p0;
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, *(__m128*)_ps_sincof_p1);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, *(__m128*)_ps_sincof_p2);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_mul_ps(y2, x);
  y2 = _mm_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  y2 = _mm_and_ps(xmm3, y2); //, xmm3);
  y = _mm_andnot_ps(xmm3, y);
  y = _mm_add_ps(y,y2);
  /* update the sign */
  y = _mm_xor_ps(y, sign_bit);

  return y;
}

} //end of namespace etl

#endif //__SSE3__
