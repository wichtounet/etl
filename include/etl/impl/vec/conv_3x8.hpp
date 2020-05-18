//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::vec::detail {

#ifdef __AVX__

#ifndef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

/*!
 * \brief Vectorized implementation of the inner valid computation
 * of a 2D convolution with a 3x8 kernel, without stride nor
 * padding. This version uses AVX and stores twice the kernel inside the AVX
 * vector in order to compute two results at once.
 *
 * Since ETL uses padding for convolution, this handle the 3x3
 * kernel that is used a lot
 *
 * \param in The input matrix of dimensions (n1, n2)
 * \param n1 The first dimension  of the input
 * \param n2 The first dimension  of the input
 * \param kkk The kernel matrix of dimensions (m1, m2)
 * \param out The output matrix
 * \param beta The multiplicative for the previous values of out
 */
template <typename V, typename T>
void conv2_valid_flipped_micro_kernel_3x8([[maybe_unused]] const T* in,
                                          [[maybe_unused]] size_t n1,
                                          [[maybe_unused]] size_t n2,
                                          [[maybe_unused]] const T* kkk,
                                          [[maybe_unused]] T* out,
                                          [[maybe_unused]] T beta) {
    if constexpr (std::is_same_v<V, etl::avx_vec> && std::is_same_v<T, float>) {
        using vec_type = V;

        const size_t m1 = 3;
        const size_t m2 = 8;

        const size_t c1 = n1 - m1 + 1;
        const size_t c2 = n2 - m2 + 1;

        T double_k1[8] = {kkk[0], kkk[1], kkk[2], 0, kkk[0], kkk[1], kkk[2], 0};
        T double_k2[8] = {kkk[8], kkk[9], kkk[10], 0, kkk[8], kkk[9], kkk[10], 0};
        T double_k3[8] = {kkk[16], kkk[17], kkk[18], 0, kkk[16], kkk[17], kkk[18], 0};

        auto dk1 = vec_type::loadu(&double_k1[0]);
        auto dk2 = vec_type::loadu(&double_k2[0]);
        auto dk3 = vec_type::loadu(&double_k3[0]);

        auto k1 = vec_type::loadu(kkk + 0 * m2 + 0);
        auto k2 = vec_type::loadu(kkk + 1 * m2 + 0);
        auto k3 = vec_type::loadu(kkk + 2 * m2 + 0);

        if (beta == T(0)) {
            for (size_t i = 0; i < c1; ++i) {
                size_t j = 0;

                for (; j + 15 < c2; j += 16) {
                    // eight values
                    auto i11 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                    auto i12 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                    auto i13 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                    auto i14 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);
                    auto i15 = vec_type::loadu(in + (i + 0) * n2 + j + 4 + 0);
                    auto i16 = vec_type::loadu(in + (i + 0) * n2 + j + 5 + 0);
                    auto i17 = vec_type::loadu(in + (i + 0) * n2 + j + 6 + 0);
                    auto i18 = vec_type::loadu(in + (i + 0) * n2 + j + 7 + 0);

                    auto r1 = vec_type::mul(i11, dk1);
                    auto r2 = vec_type::mul(i12, dk1);
                    auto r3 = vec_type::mul(i13, dk1);
                    auto r4 = vec_type::mul(i14, dk1);
                    auto r5 = vec_type::mul(i15, dk1);
                    auto r6 = vec_type::mul(i16, dk1);
                    auto r7 = vec_type::mul(i17, dk1);
                    auto r8 = vec_type::mul(i18, dk1);

                    auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                    auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                    auto i23 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                    auto i24 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);
                    auto i25 = vec_type::loadu(in + (i + 1) * n2 + j + 4 + 0);
                    auto i26 = vec_type::loadu(in + (i + 1) * n2 + j + 5 + 0);
                    auto i27 = vec_type::loadu(in + (i + 1) * n2 + j + 6 + 0);
                    auto i28 = vec_type::loadu(in + (i + 1) * n2 + j + 7 + 0);

                    r1 = vec_type::fmadd(i21, dk2, r1);
                    r2 = vec_type::fmadd(i22, dk2, r2);
                    r3 = vec_type::fmadd(i23, dk2, r3);
                    r4 = vec_type::fmadd(i24, dk2, r4);
                    r5 = vec_type::fmadd(i25, dk2, r5);
                    r6 = vec_type::fmadd(i26, dk2, r5);
                    r7 = vec_type::fmadd(i27, dk2, r6);
                    r8 = vec_type::fmadd(i28, dk2, r7);

                    auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                    auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                    auto i33 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                    auto i34 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);
                    auto i35 = vec_type::loadu(in + (i + 2) * n2 + j + 4 + 0);
                    auto i36 = vec_type::loadu(in + (i + 2) * n2 + j + 5 + 0);
                    auto i37 = vec_type::loadu(in + (i + 2) * n2 + j + 6 + 0);
                    auto i38 = vec_type::loadu(in + (i + 2) * n2 + j + 7 + 0);

                    r1 = vec_type::fmadd(i31, dk3, r1);
                    r2 = vec_type::fmadd(i32, dk3, r2);
                    r3 = vec_type::fmadd(i33, dk3, r3);
                    r4 = vec_type::fmadd(i34, dk3, r4);
                    r5 = vec_type::fmadd(i35, dk3, r5);
                    r6 = vec_type::fmadd(i36, dk3, r5);
                    r7 = vec_type::fmadd(i37, dk3, r6);
                    r8 = vec_type::fmadd(i38, dk3, r7);

                    r1 = _mm256_hadd_ps(r1.value, r1.value);
                    r2 = _mm256_hadd_ps(r2.value, r2.value);
                    r3 = _mm256_hadd_ps(r3.value, r3.value);
                    r4 = _mm256_hadd_ps(r4.value, r4.value);
                    r5 = _mm256_hadd_ps(r5.value, r5.value);
                    r6 = _mm256_hadd_ps(r6.value, r6.value);
                    r7 = _mm256_hadd_ps(r7.value, r7.value);
                    r8 = _mm256_hadd_ps(r8.value, r8.value);

                    r1 = _mm256_hadd_ps(r1.value, r1.value);
                    r2 = _mm256_hadd_ps(r2.value, r2.value);
                    r3 = _mm256_hadd_ps(r3.value, r3.value);
                    r4 = _mm256_hadd_ps(r4.value, r4.value);
                    r5 = _mm256_hadd_ps(r5.value, r5.value);
                    r6 = _mm256_hadd_ps(r6.value, r6.value);
                    r7 = _mm256_hadd_ps(r7.value, r7.value);
                    r8 = _mm256_hadd_ps(r8.value, r8.value);

                    out[i * c2 + j + 0] = r1[0];
                    out[i * c2 + j + 4] = r1[4];

                    out[i * c2 + j + 1] = r2[0];
                    out[i * c2 + j + 5] = r2[4];

                    out[i * c2 + j + 2] = r3[0];
                    out[i * c2 + j + 6] = r3[4];

                    out[i * c2 + j + 3] = r4[0];
                    out[i * c2 + j + 7] = r4[4];

                    out[i * c2 + j + 0 + 8] = r5[0];
                    out[i * c2 + j + 4 + 8] = r5[4];

                    out[i * c2 + j + 1 + 8] = r6[0];
                    out[i * c2 + j + 5 + 8] = r6[4];

                    out[i * c2 + j + 2 + 8] = r7[0];
                    out[i * c2 + j + 6 + 8] = r7[4];

                    out[i * c2 + j + 3 + 8] = r8[0];
                    out[i * c2 + j + 7 + 8] = r8[4];
                }

                for (; j + 7 < c2; j += 8) {
                    // eight values
                    auto i11 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                    auto i12 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                    auto i13 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                    auto i14 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);

                    auto r1 = vec_type::mul(i11, dk1);
                    auto r2 = vec_type::mul(i12, dk1);
                    auto r3 = vec_type::mul(i13, dk1);
                    auto r4 = vec_type::mul(i14, dk1);

                    auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                    auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                    auto i23 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                    auto i24 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);

                    r1 = vec_type::fmadd(i21, dk2, r1);
                    r2 = vec_type::fmadd(i22, dk2, r2);
                    r3 = vec_type::fmadd(i23, dk2, r3);
                    r4 = vec_type::fmadd(i24, dk2, r4);

                    auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                    auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                    auto i33 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                    auto i34 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);

                    r1 = vec_type::fmadd(i31, dk3, r1);
                    r2 = vec_type::fmadd(i32, dk3, r2);
                    r3 = vec_type::fmadd(i33, dk3, r3);
                    r4 = vec_type::fmadd(i34, dk3, r4);

                    r1 = _mm256_hadd_ps(r1.value, r1.value);
                    r2 = _mm256_hadd_ps(r2.value, r2.value);
                    r3 = _mm256_hadd_ps(r3.value, r3.value);
                    r4 = _mm256_hadd_ps(r4.value, r4.value);

                    r1 = _mm256_hadd_ps(r1.value, r1.value);
                    r2 = _mm256_hadd_ps(r2.value, r2.value);
                    r3 = _mm256_hadd_ps(r3.value, r3.value);
                    r4 = _mm256_hadd_ps(r4.value, r4.value);

                    out[i * c2 + j + 0] = r1[0];
                    out[i * c2 + j + 4] = r1[4];

                    out[i * c2 + j + 1] = r2[0];
                    out[i * c2 + j + 5] = r2[4];

                    out[i * c2 + j + 2] = r3[0];
                    out[i * c2 + j + 6] = r3[4];

                    out[i * c2 + j + 3] = r4[0];
                    out[i * c2 + j + 7] = r4[4];
                }

                for (; j + 1 < c2; j += 2) {
                    auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                    auto i2 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);

                    auto r1 = vec_type::mul(i1, k1);
                    auto r2 = vec_type::mul(i2, k1);

                    auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                    auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);

                    r1 = vec_type::fmadd(i21, k2, r1);
                    r2 = vec_type::fmadd(i22, k2, r2);

                    auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                    auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);

                    r1 = vec_type::fmadd(i31, k3, r1);
                    r2 = vec_type::fmadd(i32, k3, r2);

                    out[i * c2 + j + 0] = vec_type::hadd(r1);
                    out[i * c2 + j + 1] = vec_type::hadd(r2);
                }

                for (; j < c2; j += 1) {
                    auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);

                    auto r1 = vec_type::mul(i1, k1);

                    auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);

                    r1 = vec_type::fmadd(i21, k2, r1);

                    auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);

                    r1 = vec_type::fmadd(i31, k3, r1);

                    out[i * c2 + j + 0] = vec_type::hadd(r1);
                }
            }
        } else {
            for (size_t i = 0; i < c1; ++i) {
                size_t j = 0;

                for (; j + 15 < c2; j += 16) {
                    // eight values
                    auto i11 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                    auto i12 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                    auto i13 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                    auto i14 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);
                    auto i15 = vec_type::loadu(in + (i + 0) * n2 + j + 4 + 0);
                    auto i16 = vec_type::loadu(in + (i + 0) * n2 + j + 5 + 0);
                    auto i17 = vec_type::loadu(in + (i + 0) * n2 + j + 6 + 0);
                    auto i18 = vec_type::loadu(in + (i + 0) * n2 + j + 7 + 0);

                    auto r1 = vec_type::mul(i11, dk1);
                    auto r2 = vec_type::mul(i12, dk1);
                    auto r3 = vec_type::mul(i13, dk1);
                    auto r4 = vec_type::mul(i14, dk1);
                    auto r5 = vec_type::mul(i15, dk1);
                    auto r6 = vec_type::mul(i16, dk1);
                    auto r7 = vec_type::mul(i17, dk1);
                    auto r8 = vec_type::mul(i18, dk1);

                    auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                    auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                    auto i23 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                    auto i24 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);
                    auto i25 = vec_type::loadu(in + (i + 1) * n2 + j + 4 + 0);
                    auto i26 = vec_type::loadu(in + (i + 1) * n2 + j + 5 + 0);
                    auto i27 = vec_type::loadu(in + (i + 1) * n2 + j + 6 + 0);
                    auto i28 = vec_type::loadu(in + (i + 1) * n2 + j + 7 + 0);

                    r1 = vec_type::fmadd(i21, dk2, r1);
                    r2 = vec_type::fmadd(i22, dk2, r2);
                    r3 = vec_type::fmadd(i23, dk2, r3);
                    r4 = vec_type::fmadd(i24, dk2, r4);
                    r5 = vec_type::fmadd(i25, dk2, r5);
                    r6 = vec_type::fmadd(i26, dk2, r5);
                    r7 = vec_type::fmadd(i27, dk2, r6);
                    r8 = vec_type::fmadd(i28, dk2, r7);

                    auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                    auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                    auto i33 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                    auto i34 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);
                    auto i35 = vec_type::loadu(in + (i + 2) * n2 + j + 4 + 0);
                    auto i36 = vec_type::loadu(in + (i + 2) * n2 + j + 5 + 0);
                    auto i37 = vec_type::loadu(in + (i + 2) * n2 + j + 6 + 0);
                    auto i38 = vec_type::loadu(in + (i + 2) * n2 + j + 7 + 0);

                    r1 = vec_type::fmadd(i31, dk3, r1);
                    r2 = vec_type::fmadd(i32, dk3, r2);
                    r3 = vec_type::fmadd(i33, dk3, r3);
                    r4 = vec_type::fmadd(i34, dk3, r4);
                    r5 = vec_type::fmadd(i35, dk3, r5);
                    r6 = vec_type::fmadd(i36, dk3, r5);
                    r7 = vec_type::fmadd(i37, dk3, r6);
                    r8 = vec_type::fmadd(i38, dk3, r7);

                    r1 = _mm256_hadd_ps(r1.value, r1.value);
                    r2 = _mm256_hadd_ps(r2.value, r2.value);
                    r3 = _mm256_hadd_ps(r3.value, r3.value);
                    r4 = _mm256_hadd_ps(r4.value, r4.value);
                    r5 = _mm256_hadd_ps(r5.value, r5.value);
                    r6 = _mm256_hadd_ps(r6.value, r6.value);
                    r7 = _mm256_hadd_ps(r7.value, r7.value);
                    r8 = _mm256_hadd_ps(r8.value, r8.value);

                    r1 = _mm256_hadd_ps(r1.value, r1.value);
                    r2 = _mm256_hadd_ps(r2.value, r2.value);
                    r3 = _mm256_hadd_ps(r3.value, r3.value);
                    r4 = _mm256_hadd_ps(r4.value, r4.value);
                    r5 = _mm256_hadd_ps(r5.value, r5.value);
                    r6 = _mm256_hadd_ps(r6.value, r6.value);
                    r7 = _mm256_hadd_ps(r7.value, r7.value);
                    r8 = _mm256_hadd_ps(r8.value, r8.value);

                    out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + r1[0];
                    out[i * c2 + j + 4] = beta * out[i * c2 + j + 4] + r1[4];

                    out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + r2[0];
                    out[i * c2 + j + 5] = beta * out[i * c2 + j + 5] + r2[4];

                    out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + r3[0];
                    out[i * c2 + j + 6] = beta * out[i * c2 + j + 6] + r3[4];

                    out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + r4[0];
                    out[i * c2 + j + 7] = beta * out[i * c2 + j + 7] + r4[4];

                    out[i * c2 + j + 0 + 8] = beta * out[i * c2 + j + 0 + 8] + r5[0];
                    out[i * c2 + j + 4 + 8] = beta * out[i * c2 + j + 4 + 8] + r5[4];

                    out[i * c2 + j + 1 + 8] = beta * out[i * c2 + j + 1 + 8] + r6[0];
                    out[i * c2 + j + 5 + 8] = beta * out[i * c2 + j + 5 + 8] + r6[4];

                    out[i * c2 + j + 2 + 8] = beta * out[i * c2 + j + 2 + 8] + r7[0];
                    out[i * c2 + j + 6 + 8] = beta * out[i * c2 + j + 6 + 8] + r7[4];

                    out[i * c2 + j + 3 + 8] = beta * out[i * c2 + j + 3 + 8] + r8[0];
                    out[i * c2 + j + 7 + 8] = beta * out[i * c2 + j + 7 + 8] + r8[4];
                }

                for (; j + 7 < c2; j += 8) {
                    // eight values
                    auto i11 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                    auto i12 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                    auto i13 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                    auto i14 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);

                    auto r1 = vec_type::mul(i11, dk1);
                    auto r2 = vec_type::mul(i12, dk1);
                    auto r3 = vec_type::mul(i13, dk1);
                    auto r4 = vec_type::mul(i14, dk1);

                    auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                    auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                    auto i23 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                    auto i24 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);

                    r1 = vec_type::fmadd(i21, dk2, r1);
                    r2 = vec_type::fmadd(i22, dk2, r2);
                    r3 = vec_type::fmadd(i23, dk2, r3);
                    r4 = vec_type::fmadd(i24, dk2, r4);

                    auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                    auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                    auto i33 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                    auto i34 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);

                    r1 = vec_type::fmadd(i31, dk3, r1);
                    r2 = vec_type::fmadd(i32, dk3, r2);
                    r3 = vec_type::fmadd(i33, dk3, r3);
                    r4 = vec_type::fmadd(i34, dk3, r4);

                    r1 = _mm256_hadd_ps(r1.value, r1.value);
                    r2 = _mm256_hadd_ps(r2.value, r2.value);
                    r3 = _mm256_hadd_ps(r3.value, r3.value);
                    r4 = _mm256_hadd_ps(r4.value, r4.value);

                    r1 = _mm256_hadd_ps(r1.value, r1.value);
                    r2 = _mm256_hadd_ps(r2.value, r2.value);
                    r3 = _mm256_hadd_ps(r3.value, r3.value);
                    r4 = _mm256_hadd_ps(r4.value, r4.value);

                    out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + r1[0];
                    out[i * c2 + j + 4] = beta * out[i * c2 + j + 4] + r1[4];

                    out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + r2[0];
                    out[i * c2 + j + 5] = beta * out[i * c2 + j + 5] + r2[4];

                    out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + r3[0];
                    out[i * c2 + j + 6] = beta * out[i * c2 + j + 6] + r3[4];

                    out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + r4[0];
                    out[i * c2 + j + 7] = beta * out[i * c2 + j + 7] + r4[4];
                }

                for (; j + 1 < c2; j += 2) {
                    auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                    auto i2 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);

                    auto r1 = vec_type::mul(i1, k1);
                    auto r2 = vec_type::mul(i2, k1);

                    auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                    auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);

                    r1 = vec_type::fmadd(i21, k2, r1);
                    r2 = vec_type::fmadd(i22, k2, r2);

                    auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                    auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);

                    r1 = vec_type::fmadd(i31, k3, r1);
                    r2 = vec_type::fmadd(i32, k3, r2);

                    out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                    out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
                }

                for (; j < c2; j += 1) {
                    auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);

                    auto r1 = vec_type::mul(i1, k1);

                    auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);

                    r1 = vec_type::fmadd(i21, k2, r1);

                    auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);

                    r1 = vec_type::fmadd(i31, k3, r1);

                    out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                }
            }
        }

    } else {
        cpp_unreachable("Should never get called");
    }
}

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

#else

/*!
 * \copydoc conv2_valid_flipped_micro_kernel_3x8
 */
template <typename V, typename T>
void conv2_valid_flipped_micro_kernel_3x8([[maybe_unused]] const T* in,
                                          [[maybe_unused]] size_t n1,
                                          [[maybe_unused]] size_t n2,
                                          [[maybe_unused]] const T* kkk,
                                          [[maybe_unused]] T* out,
                                          [[maybe_unused]] T beta) {
    cpp_unreachable("Should never get called");
}

#endif

} //end of namespace etl::impl::vec::detail
