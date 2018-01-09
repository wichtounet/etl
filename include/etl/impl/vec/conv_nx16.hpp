//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {
namespace impl {
namespace vec {
namespace detail {

#ifndef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

/*!
 * \brief Vectorized implementation of the inner valid computation
 * of a 2D convolution with a nx16 kernel, without stride nor
 * padding.
 *
 * \param in The input matrix of dimensions (n1, n2)
 * \param n1 The first dimension  of the input
 * \param n2 The first dimension  of the input
 * \param kkk The kernel matrix of dimensions (m1, m2)
 * \param m1 The first dimension  of the kernel
 * \param out The output matrix
 * \param beta The multiplicative for the previous values of out
 */
template <typename V, typename T>
void conv2_valid_flipped_micro_kernel_nx16(const T* in, size_t n1, size_t n2, const T* kkk, size_t m1, T* out, T beta) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t m2 = 16;

    const size_t c1 = n1 - m1 + 1;
    const size_t c2 = n2 - m2 + 1;

    if (beta == T(0)) {
        for (size_t i = 0; i < c1; ++i) {
            size_t j = 0;

            for (; j + 7 < c2; j += 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0 * vec_size);
                    auto k2 = vec_type::loadu(kkk + k * m2 + 1 * vec_size);

                    auto i11 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0 * vec_size);
                    auto i12 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0 * vec_size);
                    auto i13 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 0 * vec_size);
                    auto i14 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 0 * vec_size);
                    auto i15 = vec_type::loadu(in + (i + k) * n2 + j + 4 + 0 * vec_size);
                    auto i16 = vec_type::loadu(in + (i + k) * n2 + j + 5 + 0 * vec_size);
                    auto i17 = vec_type::loadu(in + (i + k) * n2 + j + 6 + 0 * vec_size);
                    auto i18 = vec_type::loadu(in + (i + k) * n2 + j + 7 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k1, r1);
                    r2 = vec_type::fmadd(i12, k1, r2);
                    r3 = vec_type::fmadd(i13, k1, r3);
                    r4 = vec_type::fmadd(i14, k1, r4);
                    r5 = vec_type::fmadd(i15, k1, r5);
                    r6 = vec_type::fmadd(i16, k1, r6);
                    r7 = vec_type::fmadd(i17, k1, r7);
                    r8 = vec_type::fmadd(i18, k1, r8);

                    auto i21 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 1 * vec_size);
                    auto i22 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 1 * vec_size);
                    auto i23 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 1 * vec_size);
                    auto i24 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 1 * vec_size);
                    auto i25 = vec_type::loadu(in + (i + k) * n2 + j + 4 + 1 * vec_size);
                    auto i26 = vec_type::loadu(in + (i + k) * n2 + j + 5 + 1 * vec_size);
                    auto i27 = vec_type::loadu(in + (i + k) * n2 + j + 6 + 1 * vec_size);
                    auto i28 = vec_type::loadu(in + (i + k) * n2 + j + 7 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k2, r1);
                    r2 = vec_type::fmadd(i22, k2, r2);
                    r3 = vec_type::fmadd(i23, k2, r3);
                    r4 = vec_type::fmadd(i24, k2, r4);
                    r5 = vec_type::fmadd(i25, k2, r5);
                    r6 = vec_type::fmadd(i26, k2, r6);
                    r7 = vec_type::fmadd(i27, k2, r7);
                    r8 = vec_type::fmadd(i28, k2, r8);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
                out[i * c2 + j + 2] = vec_type::hadd(r3);
                out[i * c2 + j + 3] = vec_type::hadd(r4);
                out[i * c2 + j + 4] = vec_type::hadd(r5);
                out[i * c2 + j + 5] = vec_type::hadd(r6);
                out[i * c2 + j + 6] = vec_type::hadd(r7);
                out[i * c2 + j + 7] = vec_type::hadd(r8);
            }

            for (; j + 3 < c2; j += 4) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();

                size_t k = 0;

                for (; k + 1 < m1; k += 2) {
                    auto k11 = vec_type::loadu(kkk + (k + 0) * m2 + 0);
                    auto k12 = vec_type::loadu(kkk + (k + 0) * m2 + 1 * vec_size);

                    auto k21 = vec_type::loadu(kkk + (k + 1) * m2 + 0);
                    auto k22 = vec_type::loadu(kkk + (k + 1) * m2 + 1 * vec_size);

                    auto i11 = vec_type::loadu(in + (i + k + 0) * n2 + j + 0 + 0 * vec_size);
                    auto i12 = vec_type::loadu(in + (i + k + 0) * n2 + j + 1 + 0 * vec_size);
                    auto i13 = vec_type::loadu(in + (i + k + 0) * n2 + j + 2 + 0 * vec_size);
                    auto i14 = vec_type::loadu(in + (i + k + 0) * n2 + j + 3 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k11, r1);
                    r2 = vec_type::fmadd(i12, k11, r2);
                    r3 = vec_type::fmadd(i13, k11, r3);
                    r4 = vec_type::fmadd(i14, k11, r4);

                    auto i21 = vec_type::loadu(in + (i + k + 0) * n2 + j + 0 + 1 * vec_size);
                    auto i22 = vec_type::loadu(in + (i + k + 0) * n2 + j + 1 + 1 * vec_size);
                    auto i23 = vec_type::loadu(in + (i + k + 0) * n2 + j + 2 + 1 * vec_size);
                    auto i24 = vec_type::loadu(in + (i + k + 0) * n2 + j + 3 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k12, r1);
                    r2 = vec_type::fmadd(i22, k12, r2);
                    r3 = vec_type::fmadd(i23, k12, r3);
                    r4 = vec_type::fmadd(i24, k12, r4);

                    i11 = vec_type::loadu(in + (i + k + 1) * n2 + j + 0 + 0 * vec_size);
                    i12 = vec_type::loadu(in + (i + k + 1) * n2 + j + 1 + 0 * vec_size);
                    i13 = vec_type::loadu(in + (i + k + 1) * n2 + j + 2 + 0 * vec_size);
                    i14 = vec_type::loadu(in + (i + k + 1) * n2 + j + 3 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k21, r1);
                    r2 = vec_type::fmadd(i12, k21, r2);
                    r3 = vec_type::fmadd(i13, k21, r3);
                    r4 = vec_type::fmadd(i14, k21, r4);

                    i21 = vec_type::loadu(in + (i + k + 1) * n2 + j + 0 + 1 * vec_size);
                    i22 = vec_type::loadu(in + (i + k + 1) * n2 + j + 1 + 1 * vec_size);
                    i23 = vec_type::loadu(in + (i + k + 1) * n2 + j + 2 + 1 * vec_size);
                    i24 = vec_type::loadu(in + (i + k + 1) * n2 + j + 3 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k22, r1);
                    r2 = vec_type::fmadd(i22, k22, r2);
                    r3 = vec_type::fmadd(i23, k22, r3);
                    r4 = vec_type::fmadd(i24, k22, r4);
                }

                for (; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);
                    auto k2 = vec_type::loadu(kkk + k * m2 + 1 * vec_size);

                    auto i11 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0 * vec_size);
                    auto i12 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0 * vec_size);
                    auto i13 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 0 * vec_size);
                    auto i14 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k1, r1);
                    r2 = vec_type::fmadd(i12, k1, r2);
                    r3 = vec_type::fmadd(i13, k1, r3);
                    r4 = vec_type::fmadd(i14, k1, r4);

                    auto i21 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 1 * vec_size);
                    auto i22 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 1 * vec_size);
                    auto i23 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 1 * vec_size);
                    auto i24 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k2, r1);
                    r2 = vec_type::fmadd(i22, k2, r2);
                    r3 = vec_type::fmadd(i23, k2, r3);
                    r4 = vec_type::fmadd(i24, k2, r4);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
                out[i * c2 + j + 2] = vec_type::hadd(r3);
                out[i * c2 + j + 3] = vec_type::hadd(r4);
            }

            for (; j + 1 < c2; j += 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0 * vec_size);
                    auto k2 = vec_type::loadu(kkk + k * m2 + 1 * vec_size);

                    auto i11 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0 * vec_size);
                    auto i12 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k1, r1);
                    r2 = vec_type::fmadd(i12, k1, r2);

                    auto i21 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 1 * vec_size);
                    auto i22 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k2, r1);
                    r2 = vec_type::fmadd(i22, k2, r2);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
            }

            if (j < c2) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0 * vec_size);
                    auto k2 = vec_type::loadu(kkk + k * m2 + 1 * vec_size);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0 * vec_size);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 1 * vec_size);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r1 = vec_type::fmadd(i2, k2, r1);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
            }
        }
    } else {
        for (size_t i = 0; i < c1; ++i) {
            size_t j = 0;

            for (; j + 7 < c2; j += 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0 * vec_size);
                    auto k2 = vec_type::loadu(kkk + k * m2 + 1 * vec_size);

                    auto i11 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0 * vec_size);
                    auto i12 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0 * vec_size);
                    auto i13 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 0 * vec_size);
                    auto i14 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 0 * vec_size);
                    auto i15 = vec_type::loadu(in + (i + k) * n2 + j + 4 + 0 * vec_size);
                    auto i16 = vec_type::loadu(in + (i + k) * n2 + j + 5 + 0 * vec_size);
                    auto i17 = vec_type::loadu(in + (i + k) * n2 + j + 6 + 0 * vec_size);
                    auto i18 = vec_type::loadu(in + (i + k) * n2 + j + 7 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k1, r1);
                    r2 = vec_type::fmadd(i12, k1, r2);
                    r3 = vec_type::fmadd(i13, k1, r3);
                    r4 = vec_type::fmadd(i14, k1, r4);
                    r5 = vec_type::fmadd(i15, k1, r5);
                    r6 = vec_type::fmadd(i16, k1, r6);
                    r7 = vec_type::fmadd(i17, k1, r7);
                    r8 = vec_type::fmadd(i18, k1, r8);

                    auto i21 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 1 * vec_size);
                    auto i22 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 1 * vec_size);
                    auto i23 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 1 * vec_size);
                    auto i24 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 1 * vec_size);
                    auto i25 = vec_type::loadu(in + (i + k) * n2 + j + 4 + 1 * vec_size);
                    auto i26 = vec_type::loadu(in + (i + k) * n2 + j + 5 + 1 * vec_size);
                    auto i27 = vec_type::loadu(in + (i + k) * n2 + j + 6 + 1 * vec_size);
                    auto i28 = vec_type::loadu(in + (i + k) * n2 + j + 7 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k2, r1);
                    r2 = vec_type::fmadd(i22, k2, r2);
                    r3 = vec_type::fmadd(i23, k2, r3);
                    r4 = vec_type::fmadd(i24, k2, r4);
                    r5 = vec_type::fmadd(i25, k2, r5);
                    r6 = vec_type::fmadd(i26, k2, r6);
                    r7 = vec_type::fmadd(i27, k2, r7);
                    r8 = vec_type::fmadd(i28, k2, r8);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + vec_type::hadd(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + vec_type::hadd(r4);
                out[i * c2 + j + 4] = beta * out[i * c2 + j + 4] + vec_type::hadd(r5);
                out[i * c2 + j + 5] = beta * out[i * c2 + j + 5] + vec_type::hadd(r6);
                out[i * c2 + j + 6] = beta * out[i * c2 + j + 6] + vec_type::hadd(r7);
                out[i * c2 + j + 7] = beta * out[i * c2 + j + 7] + vec_type::hadd(r8);
            }

            for (; j + 3 < c2; j += 4) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();

                size_t k = 0;

                for (; k + 1 < m1; k += 2) {
                    auto k11 = vec_type::loadu(kkk + (k + 0) * m2 + 0);
                    auto k12 = vec_type::loadu(kkk + (k + 0) * m2 + 1 * vec_size);

                    auto k21 = vec_type::loadu(kkk + (k + 1) * m2 + 0);
                    auto k22 = vec_type::loadu(kkk + (k + 1) * m2 + 1 * vec_size);

                    auto i11 = vec_type::loadu(in + (i + k + 0) * n2 + j + 0 + 0 * vec_size);
                    auto i12 = vec_type::loadu(in + (i + k + 0) * n2 + j + 1 + 0 * vec_size);
                    auto i13 = vec_type::loadu(in + (i + k + 0) * n2 + j + 2 + 0 * vec_size);
                    auto i14 = vec_type::loadu(in + (i + k + 0) * n2 + j + 3 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k11, r1);
                    r2 = vec_type::fmadd(i12, k11, r2);
                    r3 = vec_type::fmadd(i13, k11, r3);
                    r4 = vec_type::fmadd(i14, k11, r4);

                    auto i21 = vec_type::loadu(in + (i + k + 0) * n2 + j + 0 + 1 * vec_size);
                    auto i22 = vec_type::loadu(in + (i + k + 0) * n2 + j + 1 + 1 * vec_size);
                    auto i23 = vec_type::loadu(in + (i + k + 0) * n2 + j + 2 + 1 * vec_size);
                    auto i24 = vec_type::loadu(in + (i + k + 0) * n2 + j + 3 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k12, r1);
                    r2 = vec_type::fmadd(i22, k12, r2);
                    r3 = vec_type::fmadd(i23, k12, r3);
                    r4 = vec_type::fmadd(i24, k12, r4);

                    i11 = vec_type::loadu(in + (i + k + 1) * n2 + j + 0 + 0 * vec_size);
                    i12 = vec_type::loadu(in + (i + k + 1) * n2 + j + 1 + 0 * vec_size);
                    i13 = vec_type::loadu(in + (i + k + 1) * n2 + j + 2 + 0 * vec_size);
                    i14 = vec_type::loadu(in + (i + k + 1) * n2 + j + 3 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k21, r1);
                    r2 = vec_type::fmadd(i12, k21, r2);
                    r3 = vec_type::fmadd(i13, k21, r3);
                    r4 = vec_type::fmadd(i14, k21, r4);

                    i21 = vec_type::loadu(in + (i + k + 1) * n2 + j + 0 + 1 * vec_size);
                    i22 = vec_type::loadu(in + (i + k + 1) * n2 + j + 1 + 1 * vec_size);
                    i23 = vec_type::loadu(in + (i + k + 1) * n2 + j + 2 + 1 * vec_size);
                    i24 = vec_type::loadu(in + (i + k + 1) * n2 + j + 3 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k22, r1);
                    r2 = vec_type::fmadd(i22, k22, r2);
                    r3 = vec_type::fmadd(i23, k22, r3);
                    r4 = vec_type::fmadd(i24, k22, r4);
                }

                for (; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);
                    auto k2 = vec_type::loadu(kkk + k * m2 + 1 * vec_size);

                    auto i11 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0 * vec_size);
                    auto i12 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0 * vec_size);
                    auto i13 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 0 * vec_size);
                    auto i14 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k1, r1);
                    r2 = vec_type::fmadd(i12, k1, r2);
                    r3 = vec_type::fmadd(i13, k1, r3);
                    r4 = vec_type::fmadd(i14, k1, r4);

                    auto i21 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 1 * vec_size);
                    auto i22 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 1 * vec_size);
                    auto i23 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 1 * vec_size);
                    auto i24 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k2, r1);
                    r2 = vec_type::fmadd(i22, k2, r2);
                    r3 = vec_type::fmadd(i23, k2, r3);
                    r4 = vec_type::fmadd(i24, k2, r4);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + vec_type::hadd(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + vec_type::hadd(r4);
            }

            for (; j + 1 < c2; j += 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0 * vec_size);
                    auto k2 = vec_type::loadu(kkk + k * m2 + 1 * vec_size);

                    auto i11 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0 * vec_size);
                    auto i12 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0 * vec_size);

                    r1 = vec_type::fmadd(i11, k1, r1);
                    r2 = vec_type::fmadd(i12, k1, r2);

                    auto i21 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 1 * vec_size);
                    auto i22 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 1 * vec_size);

                    r1 = vec_type::fmadd(i21, k2, r1);
                    r2 = vec_type::fmadd(i22, k2, r2);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
            }

            if (j < c2) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0 * vec_size);
                    auto k2 = vec_type::loadu(kkk + k * m2 + 1 * vec_size);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0 * vec_size);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 1 * vec_size);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r1 = vec_type::fmadd(i2, k2, r1);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
            }
        }
    }
}

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

} //end of namespace detail
} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
