//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
 * of a 2D convolution with a 5x8 kernel, without stride nor
 * padding.
 *
 * Since ETL uses padding for convolution, this handle the 5x5
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
void conv2_valid_flipped_micro_kernel_5x8(const T* in, size_t n1, size_t n2, const T* kkk, T* out, T beta) {
    using vec_type = V;

    // The padder kernel sizes
    const size_t m1 = 5;
    const size_t m2 = 8;

    // The output size
    const size_t c1 = n1 - m1 + 1;
    const size_t c2 = n2 - m2 + 1;

    // Load the kernel at once in memory
    auto k1 = vec_type::loadu(kkk + 0 * m2 + 0);
    auto k2 = vec_type::loadu(kkk + 1 * m2 + 0);
    auto k3 = vec_type::loadu(kkk + 2 * m2 + 0);
    auto k4 = vec_type::loadu(kkk + 3 * m2 + 0);
    auto k5 = vec_type::loadu(kkk + 4 * m2 + 0);

    if (beta == T(0)) {
        for (size_t i = 0; i < c1 - 0; ++i) {
            size_t j = 0;

            for (; j + 3 < c2 - 0; j += 4) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                auto i3 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                auto i4 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);

                auto r1 = vec_type::mul(i1, k1);
                auto r2 = vec_type::mul(i2, k1);
                auto r3 = vec_type::mul(i3, k1);
                auto r4 = vec_type::mul(i4, k1);

                i1 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                i3 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                i4 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i1, k2, r1);
                r2 = vec_type::fmadd(i2, k2, r2);
                r3 = vec_type::fmadd(i3, k2, r3);
                r4 = vec_type::fmadd(i4, k2, r4);

                i1 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                i3 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                i4 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i1, k3, r1);
                r2 = vec_type::fmadd(i2, k3, r2);
                r3 = vec_type::fmadd(i3, k3, r3);
                r4 = vec_type::fmadd(i4, k3, r4);

                i1 = vec_type::loadu(in + (i + 3) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 3) * n2 + j + 1 + 0);
                i3 = vec_type::loadu(in + (i + 3) * n2 + j + 2 + 0);
                i4 = vec_type::loadu(in + (i + 3) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i1, k4, r1);
                r2 = vec_type::fmadd(i2, k4, r2);
                r3 = vec_type::fmadd(i3, k4, r3);
                r4 = vec_type::fmadd(i4, k4, r4);

                i1 = vec_type::loadu(in + (i + 4) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 4) * n2 + j + 1 + 0);
                i3 = vec_type::loadu(in + (i + 4) * n2 + j + 2 + 0);
                i4 = vec_type::loadu(in + (i + 4) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i1, k5, r1);
                r2 = vec_type::fmadd(i2, k5, r2);
                r3 = vec_type::fmadd(i3, k5, r3);
                r4 = vec_type::fmadd(i4, k5, r4);

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
                out[i * c2 + j + 2] = vec_type::hadd(r3);
                out[i * c2 + j + 3] = vec_type::hadd(r4);
            }

            for (; j + 1 < c2 - 0; j += 2) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);

                auto r1 = vec_type::mul(i1, k1);
                auto r2 = vec_type::mul(i2, k1);

                i1 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);

                r1 = vec_type::fmadd(i1, k2, r1);
                r2 = vec_type::fmadd(i2, k2, r2);

                i1 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);

                r1 = vec_type::fmadd(i1, k3, r1);
                r2 = vec_type::fmadd(i2, k3, r2);

                i1 = vec_type::loadu(in + (i + 3) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 3) * n2 + j + 1 + 0);

                r1 = vec_type::fmadd(i1, k4, r1);
                r2 = vec_type::fmadd(i2, k4, r2);

                i1 = vec_type::loadu(in + (i + 4) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 4) * n2 + j + 1 + 0);

                r1 = vec_type::fmadd(i1, k5, r1);
                r2 = vec_type::fmadd(i2, k5, r2);

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
            }

            if (j < c2 - 0) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                auto i3 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                auto i4 = vec_type::loadu(in + (i + 3) * n2 + j + 0 + 0);
                auto i5 = vec_type::loadu(in + (i + 4) * n2 + j + 0 + 0);

                auto r1 = vec_type::mul(i1, k1);
                auto r2 = vec_type::mul(i2, k2);
                auto r3 = vec_type::mul(i3, k3);
                auto r4 = vec_type::mul(i4, k4);
                auto r5 = vec_type::mul(i5, k5);

                out[i * c2 + j + 0] = vec_type::hadd(r1) + vec_type::hadd(r2) + vec_type::hadd(r3) + vec_type::hadd(r4) + vec_type::hadd(r5);
            }
        }
    } else {
        for (size_t i = 0; i < c1 - 0; ++i) {
            size_t j = 0;

            for (; j + 3 < c2 - 0; j += 4) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);
                auto i3 = vec_type::loadu(in + (i + 0) * n2 + j + 2 + 0);
                auto i4 = vec_type::loadu(in + (i + 0) * n2 + j + 3 + 0);

                auto r1 = vec_type::mul(i1, k1);
                auto r2 = vec_type::mul(i2, k1);
                auto r3 = vec_type::mul(i3, k1);
                auto r4 = vec_type::mul(i4, k1);

                i1 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);
                i3 = vec_type::loadu(in + (i + 1) * n2 + j + 2 + 0);
                i4 = vec_type::loadu(in + (i + 1) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i1, k2, r1);
                r2 = vec_type::fmadd(i2, k2, r2);
                r3 = vec_type::fmadd(i3, k2, r3);
                r4 = vec_type::fmadd(i4, k2, r4);

                i1 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);
                i3 = vec_type::loadu(in + (i + 2) * n2 + j + 2 + 0);
                i4 = vec_type::loadu(in + (i + 2) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i1, k3, r1);
                r2 = vec_type::fmadd(i2, k3, r2);
                r3 = vec_type::fmadd(i3, k3, r3);
                r4 = vec_type::fmadd(i4, k3, r4);

                i1 = vec_type::loadu(in + (i + 3) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 3) * n2 + j + 1 + 0);
                i3 = vec_type::loadu(in + (i + 3) * n2 + j + 2 + 0);
                i4 = vec_type::loadu(in + (i + 3) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i1, k4, r1);
                r2 = vec_type::fmadd(i2, k4, r2);
                r3 = vec_type::fmadd(i3, k4, r3);
                r4 = vec_type::fmadd(i4, k4, r4);

                i1 = vec_type::loadu(in + (i + 4) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 4) * n2 + j + 1 + 0);
                i3 = vec_type::loadu(in + (i + 4) * n2 + j + 2 + 0);
                i4 = vec_type::loadu(in + (i + 4) * n2 + j + 3 + 0);

                r1 = vec_type::fmadd(i1, k5, r1);
                r2 = vec_type::fmadd(i2, k5, r2);
                r3 = vec_type::fmadd(i3, k5, r3);
                r4 = vec_type::fmadd(i4, k5, r4);

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + vec_type::hadd(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + vec_type::hadd(r4);
            }

            for (; j + 1 < c2 - 0; j += 2) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 0) * n2 + j + 1 + 0);

                auto r1 = vec_type::mul(i1, k1);
                auto r2 = vec_type::mul(i2, k1);

                i1 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 1) * n2 + j + 1 + 0);

                r1 = vec_type::fmadd(i1, k2, r1);
                r2 = vec_type::fmadd(i2, k2, r2);

                i1 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 2) * n2 + j + 1 + 0);

                r1 = vec_type::fmadd(i1, k3, r1);
                r2 = vec_type::fmadd(i2, k3, r2);

                i1 = vec_type::loadu(in + (i + 3) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 3) * n2 + j + 1 + 0);

                r1 = vec_type::fmadd(i1, k4, r1);
                r2 = vec_type::fmadd(i2, k4, r2);

                i1 = vec_type::loadu(in + (i + 4) * n2 + j + 0 + 0);
                i2 = vec_type::loadu(in + (i + 4) * n2 + j + 1 + 0);

                r1 = vec_type::fmadd(i1, k5, r1);
                r2 = vec_type::fmadd(i2, k5, r2);

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
            }

            if (j < c2 - 0) {
                auto i1 = vec_type::loadu(in + (i + 0) * n2 + j + 0 + 0);
                auto i2 = vec_type::loadu(in + (i + 1) * n2 + j + 0 + 0);
                auto i3 = vec_type::loadu(in + (i + 2) * n2 + j + 0 + 0);
                auto i4 = vec_type::loadu(in + (i + 3) * n2 + j + 0 + 0);
                auto i5 = vec_type::loadu(in + (i + 4) * n2 + j + 0 + 0);

                auto r1 = vec_type::mul(i1, k1);
                auto r2 = vec_type::mul(i2, k2);
                auto r3 = vec_type::mul(i3, k3);
                auto r4 = vec_type::mul(i4, k4);
                auto r5 = vec_type::mul(i5, k5);

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1) + vec_type::hadd(r2) + vec_type::hadd(r3) + vec_type::hadd(r4) + vec_type::hadd(r5);
            }
        }
    }
}

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

} //end of namespace etl::impl::vec::detail
