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
 * of a 2D convolution with a 8x8 kernel, without stride nor
 * padding.
 *
 * \param in The input matrix of dimensions (n1, n2)
 * \param n1 The first dimension  of the input
 * \param n2 The first dimension  of the input
 * \param kkk The kernel matrix of dimensions (m1, m2)
 * \param out The output matrix
 * \param beta The multiplicative for the previous values of out
 */
template <typename V, typename T>
void conv2_valid_flipped_micro_kernel_8x8(const T* in, size_t n1, size_t n2, const T* kkk, T* out, T beta) {
    using vec_type = V;

    static constexpr size_t m1 = 8;
    static constexpr size_t m2 = 8;

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
                    auto k1 = vec_type::loadu(kkk + k * m2);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1);
                    auto i3 = vec_type::loadu(in + (i + k) * n2 + j + 2);
                    auto i4 = vec_type::loadu(in + (i + k) * n2 + j + 3);
                    auto i5 = vec_type::loadu(in + (i + k) * n2 + j + 4);
                    auto i6 = vec_type::loadu(in + (i + k) * n2 + j + 5);
                    auto i7 = vec_type::loadu(in + (i + k) * n2 + j + 6);
                    auto i8 = vec_type::loadu(in + (i + k) * n2 + j + 7);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                    r3 = vec_type::fmadd(i3, k1, r3);
                    r4 = vec_type::fmadd(i4, k1, r4);
                    r5 = vec_type::fmadd(i5, k1, r5);
                    r6 = vec_type::fmadd(i6, k1, r6);
                    r7 = vec_type::fmadd(i7, k1, r7);
                    r8 = vec_type::fmadd(i8, k1, r8);
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

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1);
                    auto i3 = vec_type::loadu(in + (i + k) * n2 + j + 2);
                    auto i4 = vec_type::loadu(in + (i + k) * n2 + j + 3);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                    r3 = vec_type::fmadd(i3, k1, r3);
                    r4 = vec_type::fmadd(i4, k1, r4);
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
                    auto k1 = vec_type::loadu(kkk + k * m2);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
            }

            if (j < c2) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2);
                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0);
                    r1      = vec_type::fmadd(i1, k1, r1);
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
                    auto k1 = vec_type::loadu(kkk + k * m2);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1);
                    auto i3 = vec_type::loadu(in + (i + k) * n2 + j + 2);
                    auto i4 = vec_type::loadu(in + (i + k) * n2 + j + 3);
                    auto i5 = vec_type::loadu(in + (i + k) * n2 + j + 4);
                    auto i6 = vec_type::loadu(in + (i + k) * n2 + j + 5);
                    auto i7 = vec_type::loadu(in + (i + k) * n2 + j + 6);
                    auto i8 = vec_type::loadu(in + (i + k) * n2 + j + 7);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                    r3 = vec_type::fmadd(i3, k1, r3);
                    r4 = vec_type::fmadd(i4, k1, r4);
                    r5 = vec_type::fmadd(i5, k1, r5);
                    r6 = vec_type::fmadd(i6, k1, r6);
                    r7 = vec_type::fmadd(i7, k1, r7);
                    r8 = vec_type::fmadd(i8, k1, r8);
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

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1);
                    auto i3 = vec_type::loadu(in + (i + k) * n2 + j + 2);
                    auto i4 = vec_type::loadu(in + (i + k) * n2 + j + 3);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                    r3 = vec_type::fmadd(i3, k1, r3);
                    r4 = vec_type::fmadd(i4, k1, r4);
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
                    auto k1 = vec_type::loadu(kkk + k * m2);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
            }

            if (j < c2) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2);
                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0);
                    r1      = vec_type::fmadd(i1, k1, r1);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
            }
        }
    }
}

#ifndef __clang__
#pragma GCC diagnostic pop
#endif

} //end of namespace etl::impl::vec::detail
