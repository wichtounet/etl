//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::vec::detail {

#ifndef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

/*!
 * \brief Vectorized implementation of the inner valid computation
 * of a 2D convolution with a 3x4 kernel, without stride nor
 * padding.
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
void conv2_valid_flipped_micro_kernel_3x4(const T* in, size_t n1, size_t n2, const T* kkk, T* out, T beta) {
    using vec_type = V;

    const size_t m1 = 3;
    const size_t m2 = 4;

    const size_t c1 = n1 - m1 + 1;
    const size_t c2 = n2 - m2 + 1;

    auto k1 = vec_type::loadu(kkk + 0 * m2 + 0);
    auto k2 = vec_type::loadu(kkk + 1 * m2 + 0);
    auto k3 = vec_type::loadu(kkk + 2 * m2 + 0);

    if (beta == T(0)) {
        for (size_t i = 0; i < c1; ++i) {
            size_t j = 0;

            for (; j + 7 < c2; j += 8) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                auto i3 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                auto i4 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);
                auto i5 = vec_type::loadu(in + (i + 0) * n2 + j + 4 + 0);
                auto i6 = vec_type::loadu(in + (i + 0) * n2 + j + 5 + 0);
                auto i7 = vec_type::loadu(in + (i + 0) * n2 + j + 6 + 0);
                auto i8 = vec_type::loadu(in + (i + 0) * n2 + j + 7 + 0);

                auto r1 = vec_type::mul(i1, k1);
                auto r2 = vec_type::mul(i2, k1);
                auto r3 = vec_type::mul(i3, k1);
                auto r4 = vec_type::mul(i4, k1);
                auto r5 = vec_type::mul(i5, k1);
                auto r6 = vec_type::mul(i6, k1);
                auto r7 = vec_type::mul(i7, k1);
                auto r8 = vec_type::mul(i8, k1);

                auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                auto i23 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                auto i24 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);
                auto i25 = vec_type::loadu(in + (i + 1) * n2 + j + 4 + 0);
                auto i26 = vec_type::loadu(in + (i + 1) * n2 + j + 5 + 0);
                auto i27 = vec_type::loadu(in + (i + 1) * n2 + j + 6 + 0);
                auto i28 = vec_type::loadu(in + (i + 1) * n2 + j + 7 + 0);

                r1 = vec_type::fmadd(i21, k2, r1);
                r2 = vec_type::fmadd(i22, k2, r2);
                r3 = vec_type::fmadd(i23, k2, r3);
                r4 = vec_type::fmadd(i24, k2, r4);
                r5 = vec_type::fmadd(i25, k2, r5);
                r6 = vec_type::fmadd(i26, k2, r6);
                r7 = vec_type::fmadd(i27, k2, r7);
                r8 = vec_type::fmadd(i28, k2, r8);

                auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                auto i33 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                auto i34 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);
                auto i35 = vec_type::loadu(in + (i + 2) * n2 + j + 4 + 0);
                auto i36 = vec_type::loadu(in + (i + 2) * n2 + j + 5 + 0);
                auto i37 = vec_type::loadu(in + (i + 2) * n2 + j + 6 + 0);
                auto i38 = vec_type::loadu(in + (i + 2) * n2 + j + 7 + 0);

                r1 = vec_type::fmadd(i31, k3, r1);
                r2 = vec_type::fmadd(i32, k3, r2);
                r3 = vec_type::fmadd(i33, k3, r3);
                r4 = vec_type::fmadd(i34, k3, r4);
                r5 = vec_type::fmadd(i35, k3, r5);
                r6 = vec_type::fmadd(i36, k3, r6);
                r7 = vec_type::fmadd(i37, k3, r7);
                r8 = vec_type::fmadd(i38, k3, r8);

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
                auto i11 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i12 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                auto i13 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                auto i14 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);

                auto r1 = vec_type::mul(i11, k1);
                auto r2 = vec_type::mul(i12, k1);
                auto r3 = vec_type::mul(i13, k1);
                auto r4 = vec_type::mul(i14, k1);

                auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                auto i23 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                auto i24 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i21, k2, r1);
                r2 = vec_type::fmadd(i22, k2, r2);
                r3 = vec_type::fmadd(i23, k2, r3);
                r4 = vec_type::fmadd(i24, k2, r4);

                auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                auto i33 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                auto i34 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i31, k3, r1);
                r2 = vec_type::fmadd(i32, k3, r2);
                r3 = vec_type::fmadd(i33, k3, r3);
                r4 = vec_type::fmadd(i34, k3, r4);

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
                out[i * c2 + j + 2] = vec_type::hadd(r3);
                out[i * c2 + j + 3] = vec_type::hadd(r4);
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

            if (j < c2) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                auto i3 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);

                auto r1 = vec_type::mul(i1, k1);
                r1      = vec_type::fmadd(i2, k2, r1);
                r1      = vec_type::fmadd(i3, k3, r1);

                out[i * c2 + j + 0] = vec_type::hadd(r1);
            }
        }
    } else {
        for (size_t i = 0; i < c1; ++i) {
            size_t j = 0;

            for (; j + 7 < c2; j += 8) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                auto i3 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                auto i4 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);
                auto i5 = vec_type::loadu(in + (i + 0) * n2 + j + 4 + 0);
                auto i6 = vec_type::loadu(in + (i + 0) * n2 + j + 5 + 0);
                auto i7 = vec_type::loadu(in + (i + 0) * n2 + j + 6 + 0);
                auto i8 = vec_type::loadu(in + (i + 0) * n2 + j + 7 + 0);

                auto r1 = vec_type::mul(i1, k1);
                auto r2 = vec_type::mul(i2, k1);
                auto r3 = vec_type::mul(i3, k1);
                auto r4 = vec_type::mul(i4, k1);
                auto r5 = vec_type::mul(i5, k1);
                auto r6 = vec_type::mul(i6, k1);
                auto r7 = vec_type::mul(i7, k1);
                auto r8 = vec_type::mul(i8, k1);

                auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                auto i23 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                auto i24 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);
                auto i25 = vec_type::loadu(in + (i + 1) * n2 + j + 4 + 0);
                auto i26 = vec_type::loadu(in + (i + 1) * n2 + j + 5 + 0);
                auto i27 = vec_type::loadu(in + (i + 1) * n2 + j + 6 + 0);
                auto i28 = vec_type::loadu(in + (i + 1) * n2 + j + 7 + 0);

                r1 = vec_type::fmadd(i21, k2, r1);
                r2 = vec_type::fmadd(i22, k2, r2);
                r3 = vec_type::fmadd(i23, k2, r3);
                r4 = vec_type::fmadd(i24, k2, r4);
                r5 = vec_type::fmadd(i25, k2, r5);
                r6 = vec_type::fmadd(i26, k2, r6);
                r7 = vec_type::fmadd(i27, k2, r7);
                r8 = vec_type::fmadd(i28, k2, r8);

                auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                auto i33 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                auto i34 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);
                auto i35 = vec_type::loadu(in + (i + 2) * n2 + j + 4 + 0);
                auto i36 = vec_type::loadu(in + (i + 2) * n2 + j + 5 + 0);
                auto i37 = vec_type::loadu(in + (i + 2) * n2 + j + 6 + 0);
                auto i38 = vec_type::loadu(in + (i + 2) * n2 + j + 7 + 0);

                r1 = vec_type::fmadd(i31, k3, r1);
                r2 = vec_type::fmadd(i32, k3, r2);
                r3 = vec_type::fmadd(i33, k3, r3);
                r4 = vec_type::fmadd(i34, k3, r4);
                r5 = vec_type::fmadd(i35, k3, r5);
                r6 = vec_type::fmadd(i36, k3, r6);
                r7 = vec_type::fmadd(i37, k3, r7);
                r8 = vec_type::fmadd(i38, k3, r8);

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
                auto i11 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i12 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                auto i13 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                auto i14 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);

                auto r1 = vec_type::mul(i11, k1);
                auto r2 = vec_type::mul(i12, k1);
                auto r3 = vec_type::mul(i13, k1);
                auto r4 = vec_type::mul(i14, k1);

                auto i21 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                auto i22 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                auto i23 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                auto i24 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i21, k2, r1);
                r2 = vec_type::fmadd(i22, k2, r2);
                r3 = vec_type::fmadd(i23, k2, r3);
                r4 = vec_type::fmadd(i24, k2, r4);

                auto i31 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                auto i32 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                auto i33 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                auto i34 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i31, k3, r1);
                r2 = vec_type::fmadd(i32, k3, r2);
                r3 = vec_type::fmadd(i33, k3, r3);
                r4 = vec_type::fmadd(i34, k3, r4);

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + vec_type::hadd(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + vec_type::hadd(r4);
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

            if (j < c2) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                auto i3 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);

                auto r1 = vec_type::mul(i1, k1);
                r1      = vec_type::fmadd(i2, k2, r1);
                r1      = vec_type::fmadd(i3, k3, r1);

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
            }
        }
    }
}

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

} //end of namespace etl::impl::vec::detail
