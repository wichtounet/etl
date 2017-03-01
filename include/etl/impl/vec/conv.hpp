//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/common/conv.hpp"

/*
 * Performance notes:
 *
 * Several things are suboptimal in this implementation
 * a) Ideally, the valid_micro_kernel should directly use input.loadu instead of
 * going through the memory and vec_type::loadu(mem), but compilers do not seem
 * to be able to go through layers of sub view for memory_start
 * b) Ideally the micro kernel should use conv(i, j) directly instead of using
 * the pointer. But there is a small overhead when using this technique :s
 */

namespace etl {

namespace impl {

namespace vec {

namespace detail {

template <typename T>
constexpr bool prefer_sse(const size_t n) {
    return !avx_enabled || (sse3_enabled && (std::is_same<T, float>::value
                                                 ? (n % 4 < n % 8)
                                                 : (n % 2 < n % 4)));
}

template <typename I, typename C>
void pad_2d_input(const I& in, C& out, size_t p1, size_t p2) {
    in.ensure_cpu_up_to_date();

    auto in_m  = in.memory_start();
    auto out_m = out.memory_start();

    for (size_t i = 0; i < etl::dim<0>(in); ++i) {
        direct_copy_n(in_m + i * etl::dim<1>(in), out_m + (i + p1) * etl::dim<1>(out) + p2, etl::dim<1>(in));
    }

    out.invalidate_gpu();
}

template <typename I, typename K, typename C>
inline void conv2_valid_flipped_border(const I& input, const K& kernel, C&& conv, size_t i, size_t j, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta) {
    using T = value_t<I>;

    const size_t n1 = etl::dim<0>(input);
    const size_t n2 = etl::dim<1>(input);

    const size_t m1 = etl::dim<0>(kernel);
    const size_t m2 = etl::dim<1>(kernel);

    T temp = T(0);

    const size_t s_i = i * s1;
    const size_t s_j = j * s2;

    for (size_t k = 0; k < m1; ++k) {
        for (size_t l = 0; l < m2; ++l) {
            if (s_i + k >= p1 && (s_i + k) - p1 < n1 && s_j + l >= p2 && (s_j + l) - p2 < n2) {
                const size_t i_i = (s_i + k) - p1;
                const size_t i_j = (s_j + l) - p2;

                temp += input(i_i, i_j) * kernel(k, l);
            }
        }
    }

    if (beta == T(0)) {
        conv(i, j) = temp;
    } else {
        conv(i, j) = beta * conv(i, j) + temp;
    }
}

#ifndef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
#endif

// Computation of a 3x3 kernel
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

template <typename V, typename T>
void conv2_valid_flipped_micro_kernel_nx8(const T* in, size_t n1, size_t n2, const T* kkk, size_t m1, T* out, T beta) {
    using vec_type = V;

    const size_t m2 = 8;

    const size_t c1 = n1 - m1 + 1;
    const size_t c2 = n2 - m2 + 1;

    if (beta == T(0)) {
        for (size_t i = 0; i < c1 - 0; ++i) {
            size_t j = 0;

            for (; j + 7 < c2 - 0; j += 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0);
                    auto i3 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 0);
                    auto i4 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 0);
                    auto i5 = vec_type::loadu(in + (i + k) * n2 + j + 4 + 0);
                    auto i6 = vec_type::loadu(in + (i + k) * n2 + j + 5 + 0);
                    auto i7 = vec_type::loadu(in + (i + k) * n2 + j + 6 + 0);
                    auto i8 = vec_type::loadu(in + (i + k) * n2 + j + 7 + 0);

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

            for (; j + 1 < c2 - 0; j += 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
            }

            if (j < c2 - 0) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);
                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    r1      = vec_type::fmadd(i1, k1, r1);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
            }
        }
    } else {
        for (size_t i = 0; i < c1 - 0; ++i) {
            size_t j = 0;

            for (; j + 7 < c2 - 0; j += 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0);
                    auto i3 = vec_type::loadu(in + (i + k) * n2 + j + 2 + 0);
                    auto i4 = vec_type::loadu(in + (i + k) * n2 + j + 3 + 0);
                    auto i5 = vec_type::loadu(in + (i + k) * n2 + j + 4 + 0);
                    auto i6 = vec_type::loadu(in + (i + k) * n2 + j + 5 + 0);
                    auto i7 = vec_type::loadu(in + (i + k) * n2 + j + 6 + 0);
                    auto i8 = vec_type::loadu(in + (i + k) * n2 + j + 7 + 0);

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

            for (; j + 1 < c2 - 0; j += 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
            }

            if (j < c2 - 0) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);
                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    r1      = vec_type::fmadd(i1, k1, r1);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
            }
        }
    }
}

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

template <typename V, typename T>
void conv2_valid_flipped_micro_kernel_5x8(const T* in, size_t n1, size_t n2, const T* kkk, T* out, T beta) {
    using vec_type = V;

    const size_t m1 = 5;
    const size_t m2 = 8;

    const size_t c1 = n1 - m1 + 1;
    const size_t c2 = n2 - m2 + 1;

    if (beta == T(0)) {
        for (size_t i = 0; i < c1 - 0; ++i) {
            size_t j = 0;

            auto k1 = vec_type::loadu(kkk + 0 * m2 + 0);
            auto k2 = vec_type::loadu(kkk + 1 * m2 + 0);
            auto k3 = vec_type::loadu(kkk + 2 * m2 + 0);
            auto k4 = vec_type::loadu(kkk + 3 * m2 + 0);
            auto k5 = vec_type::loadu(kkk + 4 * m2 + 0);

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
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
            }

            if (j < c2 - 0) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);
                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    r1      = vec_type::fmadd(i1, k1, r1);
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
            }
        }
    } else {
        for (size_t i = 0; i < c1 - 0; ++i) {
            size_t j = 0;

            auto k1 = vec_type::loadu(kkk + 0 * m2 + 0);
            auto k2 = vec_type::loadu(kkk + 1 * m2 + 0);
            auto k3 = vec_type::loadu(kkk + 2 * m2 + 0);
            auto k4 = vec_type::loadu(kkk + 3 * m2 + 0);
            auto k5 = vec_type::loadu(kkk + 4 * m2 + 0);

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
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);

                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
                    auto i2 = vec_type::loadu(in + (i + k) * n2 + j + 1 + 0);

                    r1 = vec_type::fmadd(i1, k1, r1);
                    r2 = vec_type::fmadd(i2, k1, r2);
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
            }

            if (j < c2 - 0) {
                auto r1 = vec_type::template zero<T>();

                for (size_t k = 0; k < m1; ++k) {
                    auto k1 = vec_type::loadu(kkk + k * m2 + 0);
                    auto i1 = vec_type::loadu(in + (i + k) * n2 + j + 0 + 0);
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

template <typename V, typename T>
void conv2_valid_flipped_inner_kernel(const T* in, size_t n1, size_t n2, const T* kkk, size_t m1, size_t m2, T* out, size_t s1, size_t s2, size_t p1, size_t p2, T beta) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t c1 = (n1 - m1 + 2 * p1) / s1 + 1;
    const size_t c2 = (n2 - m2 + 2 * p2) / s2 + 1;

    if (cpp_likely(!p1 && !p2 && s1 == 1 && s2 == 1)) {
        if (vec_size == 4 && m1 == 3 && m2 == 4) {
            conv2_valid_flipped_micro_kernel_3x4<V>(in, n1, n2, kkk, out, beta);
            return;
        } else if (vec_size == 8 && m1 == 5 && m2 == 8) {
            conv2_valid_flipped_micro_kernel_5x8<V>(in, n1, n2, kkk, out, beta);
            return;
        } else if (vec_size == 8 && m1 == 8 && m2 == 8) {
            conv2_valid_flipped_micro_kernel_8x8<V>(in, n1, n2, kkk, out, beta);
            return;
        } else if (vec_size == 8 && m2 == 8) {
            conv2_valid_flipped_micro_kernel_nx8<V>(in, n1, n2, kkk, m1, out, beta);
            return;
        } else if (vec_size == 8 && m2 == 16) {
            conv2_valid_flipped_micro_kernel_nx16<V>(in, n1, n2, kkk, m1, out, beta);
            return;
        }
    }

    if (beta == T(0)) {
        for (size_t i = p1; i < c1 - p1; ++i) {
            size_t j = p2;

            for (; j + 7 < c2 - p2; j += 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);

                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        auto i2 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 1) * s2 - p2) + l);
                        auto i3 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 2) * s2 - p2) + l);
                        auto i4 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 3) * s2 - p2) + l);
                        auto i5 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 4) * s2 - p2) + l);
                        auto i6 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 5) * s2 - p2) + l);
                        auto i7 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 6) * s2 - p2) + l);
                        auto i8 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 7) * s2 - p2) + l);

                        r1 = vec_type::fmadd(i1, k1, r1);
                        r2 = vec_type::fmadd(i2, k1, r2);
                        r3 = vec_type::fmadd(i3, k1, r3);
                        r4 = vec_type::fmadd(i4, k1, r4);
                        r5 = vec_type::fmadd(i5, k1, r5);
                        r6 = vec_type::fmadd(i6, k1, r6);
                        r7 = vec_type::fmadd(i7, k1, r7);
                        r8 = vec_type::fmadd(i8, k1, r8);
                    }
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

            for (; j + 1 < c2 - p2; j += 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);

                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        auto i2 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 1) * s2 - p2) + l);

                        r1 = vec_type::fmadd(i1, k1, r1);
                        r2 = vec_type::fmadd(i2, k1, r2);
                    }
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
            }

            if (j < c2 - p2) {
                auto r1 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);
                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        r1      = vec_type::fmadd(i1, k1, r1);
                    }
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
            }
        }
    } else {
        for (size_t i = p1; i < c1 - p1; ++i) {
            size_t j = p2;

            for (; j + 7 < c2 - p2; j += 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);

                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        auto i2 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 1) * s2 - p2) + l);
                        auto i3 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 2) * s2 - p2) + l);
                        auto i4 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 3) * s2 - p2) + l);
                        auto i5 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 4) * s2 - p2) + l);
                        auto i6 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 5) * s2 - p2) + l);
                        auto i7 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 6) * s2 - p2) + l);
                        auto i8 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 7) * s2 - p2) + l);

                        r1 = vec_type::fmadd(i1, k1, r1);
                        r2 = vec_type::fmadd(i2, k1, r2);
                        r3 = vec_type::fmadd(i3, k1, r3);
                        r4 = vec_type::fmadd(i4, k1, r4);
                        r5 = vec_type::fmadd(i5, k1, r5);
                        r6 = vec_type::fmadd(i6, k1, r6);
                        r7 = vec_type::fmadd(i7, k1, r7);
                        r8 = vec_type::fmadd(i8, k1, r8);
                    }
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

            for (; j + 1 < c2 - p2; j += 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);

                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        auto i2 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 1) * s2 - p2) + l);

                        r1 = vec_type::fmadd(i1, k1, r1);
                        r2 = vec_type::fmadd(i2, k1, r2);
                    }
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
            }

            if (j < c2 - p2) {
                auto r1 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);
                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        r1      = vec_type::fmadd(i1, k1, r1);
                    }
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
            }
        }
    }

    if (!padding_impl && m2 % vec_size != 0) {
        size_t rem = m2 % vec_size;
        for (size_t i = p1; i < c1 - p1; ++i) {
            for (size_t j = p2; j < c2 - p2; ++j) {
                T temp = 0.0;

                const size_t i_i = i * s1 - p1;
                const size_t i_j = j * s2 - p2;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = m2 - rem; l < m2; ++l) {
                        temp += in[i_i + k * n2 + i_j + l] * kkk[k * m2 + l];
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    }
}

template <typename V, typename I, typename K, typename C>
void conv2_valid_flipped_micro_kernel(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta) {
    using T        = value_t<I>;

    const size_t n1 = etl::dim<0>(input);
    const size_t n2 = etl::dim<1>(input);

    const size_t m1 = etl::dim<0>(kernel);
    const size_t m2 = etl::dim<1>(kernel);

    const size_t c1 = (n1 - m1 + 2 * p1) / s1 + 1;
    const size_t c2 = (n2 - m2 + 2 * p2) / s2 + 1;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (beta != T(0)) {
        conv.ensure_cpu_up_to_date();
    }

    auto* kkk = kernel.memory_start();
    auto* in  = input.memory_start();
    auto* out = conv.memory_start();

    conv2_valid_flipped_inner_kernel<V>(in, n1, n2, kkk, m1, m2, out, s1, s2, p1, p2, beta);

    conv.invalidate_gpu();

    if (cpp_unlikely(p1 || p2)) {
        for (size_t i = 0; i < p1; ++i) {
            for (size_t j = 0; j < c2; ++j) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t i = c1 - p1; i < c1; ++i) {
            for (size_t j = 0; j < c2; ++j) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t j = 0; j < p2; ++j) {
            for (size_t i = p1; i < c1 - p1; ++i) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t j = c2 - p2; j < c2; ++j) {
            for (size_t i = p1; i < c1 - p1; ++i) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }
    }
}

template <typename V, typename I, typename K, typename C>
void conv2_valid_micro_kernel(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta) {
    etl::dyn_matrix<value_t<I>, 2> kernel_reverse(etl::dim<0>(kernel), etl::dim<1>(kernel));

    std::reverse_copy(kernel.begin(), kernel.end(), kernel_reverse.begin());

    conv2_valid_flipped_micro_kernel<V>(input, kernel_reverse, conv, s1, s2, p1, p2, beta);
}

#ifdef __AVX__
using safe_avx_vec = avx_vec;
#else
using safe_avx_vec = no_vec;
#endif

#ifdef __SSE3__
using safe_sse_vec = sse_vec;
#else
using safe_sse_vec = no_vec;
#endif

} // end of namespace detail

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t k2 = etl::dim<1>(kernel);

    if (cpp_unlikely(p1 || p2)) {
        const size_t n1 = etl::dim<0>(input);
        const size_t n2 = etl::dim<1>(input);

        const size_t o1 = n1 + 2 * p1;
        const size_t o2 = n2 + 2 * p2;

        if (o1 * o2 * sizeof(T) < max_workspace) {
            etl::dyn_matrix<T, 2> padded_matrix(o1, o2, T(0));

            detail::pad_2d_input(input, padded_matrix, p1, p2);

            conv2_valid_flipped(padded_matrix, kernel, conv, s1, s2, 0, 0);

            return;
        }
    }

    if (cpp_unlikely(s1 > 1 || s2 > 1)) {
        const size_t n1 = etl::dim<0>(input);
        const size_t n2 = etl::dim<1>(input);

        const size_t c1 = etl::dim<0>(conv);
        const size_t c2 = etl::dim<1>(conv);

        const size_t k1 = etl::dim<0>(kernel);
        const size_t k2 = etl::dim<1>(kernel);

        etl::dyn_matrix<T> tmp_result(n1 - k1 + 1, n2 - k2 + 1);

        conv2_valid_flipped(input, kernel, tmp_result, 1, 1, 0, 0);

        // Strided copy of the large result into the small result
        for (size_t i = 0; i < c1; ++i) {
            for (size_t j = 0; j < c2; ++j) {
                conv(i, j) = tmp_result(i * s1, j * s2);
            }
        }

        return;
    }

    if (padding_impl) {
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if (k2 < SS || k2 % AS > 0) {
            const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input  = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right(kernel, pad);

            if (detail::prefer_sse<T>(k2 + pad)) {
                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input, padded_kernel, conv, s1, s2, p1, p2, T(0));
            } else {
                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input, padded_kernel, conv, s1, s2, p1, p2, T(0));
            }

            return;
        }
    }

    if (detail::prefer_sse<T>(k2)) {
        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input, kernel, conv, s1, s2, p1, p2, T(0));
    } else {
        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input, kernel, conv, s1, s2, p1, p2, T(0));
    }
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t k2 = etl::dim<1>(kernel);

    if (cpp_unlikely(p1 || p2)) {
        const size_t n1 = etl::dim<0>(input);
        const size_t n2 = etl::dim<1>(input);

        const size_t o1 = n1 + 2 * p1;
        const size_t o2 = n2 + 2 * p2;

        if (o1 * o2 * sizeof(T) < max_workspace) {
            etl::dyn_matrix<T, 2> padded_matrix(o1, o2, T(0));

            detail::pad_2d_input(input, padded_matrix, p1, p2);

            conv2_valid(padded_matrix, kernel, conv, s1, s2, 0, 0);

            return;
        }
    }

    if (cpp_unlikely(s1 > 1 || s2 > 1)) {
        const size_t n1 = etl::dim<0>(input);
        const size_t n2 = etl::dim<1>(input);

        const size_t c1 = etl::dim<0>(conv);
        const size_t c2 = etl::dim<1>(conv);

        const size_t k1 = etl::dim<0>(kernel);

        etl::dyn_matrix<T> tmp_result(n1 - k1 + 1, n2 - k2 + 1);

        conv2_valid(input, kernel, tmp_result, 1, 1, 0, 0);

        // Strided copy of the large result into the small result
        for (size_t i = 0; i < c1; ++i) {
            for (size_t j = 0; j < c2; ++j) {
                conv(i, j) = tmp_result(i * s1, j * s2);
            }
        }

        return;
    }

    if (padding_impl) {
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if (k2 < SS || k2 % AS > 0) {
            const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input  = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right_flip(kernel, pad);

            if (detail::prefer_sse<T>(k2 + pad)) {
                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input, padded_kernel, conv, s1, s2, p1, p2, T(0));
            } else {
                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input, padded_kernel, conv, s1, s2, p1, p2, T(0));
            }

            return;
        }
    }

    if (detail::prefer_sse<T>(k2)) {
        detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input, kernel, conv, s1, s2, p1, p2, T(0));
    } else {
        detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input, kernel, conv, s1, s2, p1, p2, T(0));
    }
}

/*!
 * \brief Vectorized implementation of a 1D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv1_valid(const I& input, const K& kernel, C&& conv, size_t first, size_t last) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using vec_type = V;
    using T        = value_t<I>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t n = etl::size(input);
    const size_t m = etl::size(kernel);

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    auto llast = std::min(n - m + 1, last);

    auto kernel_reverse = aligned_allocate_auto<T>(m);

    std::reverse_copy(kernel.begin(), kernel.end(), kernel_reverse.get());

    size_t j = first;

    for (; j + 7 < llast; j += 8) {
        const size_t j1 = j;
        const size_t j2 = j + 1;
        const size_t j3 = j + 2;
        const size_t j4 = j + 3;
        const size_t j5 = j + 4;
        const size_t j6 = j + 5;
        const size_t j7 = j + 6;
        const size_t j8 = j + 7;

        auto r11 = vec_type::template zero<T>();
        auto r21 = vec_type::template zero<T>();
        auto r31 = vec_type::template zero<T>();
        auto r41 = vec_type::template zero<T>();
        auto r51 = vec_type::template zero<T>();
        auto r61 = vec_type::template zero<T>();
        auto r71 = vec_type::template zero<T>();
        auto r81 = vec_type::template zero<T>();

        size_t l = 0;

        for (; l + vec_size - 1 < m; l += vec_size) {
            auto i11 = input.template loadu<vec_type>(j1 + l);
            auto i21 = input.template loadu<vec_type>(j2 + l);
            auto i31 = input.template loadu<vec_type>(j3 + l);
            auto i41 = input.template loadu<vec_type>(j4 + l);
            auto i51 = input.template loadu<vec_type>(j5 + l);
            auto i61 = input.template loadu<vec_type>(j6 + l);
            auto i71 = input.template loadu<vec_type>(j7 + l);
            auto i81 = input.template loadu<vec_type>(j8 + l);

            auto k1 = vec_type::load(kernel_reverse.get() + l);

            r11 = vec_type::fmadd(i11, k1, r11);
            r21 = vec_type::fmadd(i21, k1, r21);
            r31 = vec_type::fmadd(i31, k1, r31);
            r41 = vec_type::fmadd(i41, k1, r41);
            r51 = vec_type::fmadd(i51, k1, r51);
            r61 = vec_type::fmadd(i61, k1, r61);
            r71 = vec_type::fmadd(i71, k1, r71);
            r81 = vec_type::fmadd(i81, k1, r81);
        }

        auto p11 = vec_type::hadd(r11);
        auto p21 = vec_type::hadd(r21);
        auto p31 = vec_type::hadd(r31);
        auto p41 = vec_type::hadd(r41);
        auto p51 = vec_type::hadd(r51);
        auto p61 = vec_type::hadd(r61);
        auto p71 = vec_type::hadd(r71);
        auto p81 = vec_type::hadd(r81);

        for (; l < m; ++l) {
            p11 += input[j1 + l] * kernel_reverse[l];
            p21 += input[j2 + l] * kernel_reverse[l];
            p31 += input[j3 + l] * kernel_reverse[l];
            p41 += input[j4 + l] * kernel_reverse[l];
            p51 += input[j5 + l] * kernel_reverse[l];
            p61 += input[j6 + l] * kernel_reverse[l];
            p71 += input[j7 + l] * kernel_reverse[l];
            p81 += input[j8 + l] * kernel_reverse[l];
        }

        conv[j1] = p11;
        conv[j2] = p21;
        conv[j3] = p31;
        conv[j4] = p41;
        conv[j5] = p51;
        conv[j6] = p61;
        conv[j7] = p71;
        conv[j8] = p81;
    }

    for (; j + 3 < llast; j += 4) {
        const size_t j1 = j;
        const size_t j2 = j + 1;
        const size_t j3 = j + 2;
        const size_t j4 = j + 3;

        auto r11 = vec_type::template zero<T>();
        auto r12 = vec_type::template zero<T>();

        auto r21 = vec_type::template zero<T>();
        auto r22 = vec_type::template zero<T>();

        auto r31 = vec_type::template zero<T>();
        auto r32 = vec_type::template zero<T>();

        auto r41 = vec_type::template zero<T>();
        auto r42 = vec_type::template zero<T>();

        size_t l = 0;

        for (; l + (vec_size * 2) - 1 < m; l += 2 * vec_size) {
            auto k1 = vec_type::load(kernel_reverse.get() + l + vec_size * 0);
            auto k2 = vec_type::load(kernel_reverse.get() + l + vec_size * 1);

            auto i11 = input.template loadu<vec_type>(j1 + l + vec_size * 0);
            auto i12 = input.template loadu<vec_type>(j1 + l + vec_size * 1);

            auto i21 = input.template loadu<vec_type>(j2 + l + vec_size * 0);
            auto i22 = input.template loadu<vec_type>(j2 + l + vec_size * 1);

            auto i31 = input.template loadu<vec_type>(j3 + l + vec_size * 0);
            auto i32 = input.template loadu<vec_type>(j3 + l + vec_size * 1);

            auto i41 = input.template loadu<vec_type>(j4 + l + vec_size * 0);
            auto i42 = input.template loadu<vec_type>(j4 + l + vec_size * 1);

            r11 = vec_type::fmadd(i11, k1, r11);
            r12 = vec_type::fmadd(i12, k2, r12);

            r21 = vec_type::fmadd(i21, k1, r21);
            r22 = vec_type::fmadd(i22, k2, r22);

            r31 = vec_type::fmadd(i31, k1, r31);
            r32 = vec_type::fmadd(i32, k2, r32);

            r41 = vec_type::fmadd(i41, k1, r41);
            r42 = vec_type::fmadd(i42, k2, r42);
        }

        for (; l + vec_size - 1 < m; l += vec_size) {
            auto i11 = input.template loadu<vec_type>(j1 + l);
            auto i21 = input.template loadu<vec_type>(j2 + l);
            auto i31 = input.template loadu<vec_type>(j3 + l);
            auto i41 = input.template loadu<vec_type>(j4 + l);

            auto k1 = vec_type::load(kernel_reverse.get() + l);

            r11 = vec_type::fmadd(i11, k1, r11);
            r21 = vec_type::fmadd(i21, k1, r21);
            r31 = vec_type::fmadd(i31, k1, r31);
            r41 = vec_type::fmadd(i41, k1, r41);
        }

        auto p11 = vec_type::hadd(vec_type::add(r11, r12));
        auto p12 = T(0);

        auto p21 = vec_type::hadd(vec_type::add(r21, r22));
        auto p22 = T(0);

        auto p31 = vec_type::hadd(vec_type::add(r31, r32));
        auto p32 = T(0);

        auto p41 = vec_type::hadd(vec_type::add(r41, r42));
        auto p42 = T(0);

        for (; l + 1 < m; l += 2) {
            p11 += input[j1 + l + 0] * kernel_reverse[l + 0];
            p12 += input[j1 + l + 1] * kernel_reverse[l + 1];

            p21 += input[j2 + l + 0] * kernel_reverse[l + 0];
            p22 += input[j2 + l + 1] * kernel_reverse[l + 1];

            p31 += input[j3 + l + 0] * kernel_reverse[l + 0];
            p32 += input[j3 + l + 1] * kernel_reverse[l + 1];

            p41 += input[j4 + l + 0] * kernel_reverse[l + 0];
            p42 += input[j4 + l + 1] * kernel_reverse[l + 1];
        }

        if (l < m) {
            p11 += input[j1 + l] * kernel_reverse[l];
            p21 += input[j2 + l] * kernel_reverse[l];
            p31 += input[j3 + l] * kernel_reverse[l];
            p31 += input[j4 + l] * kernel_reverse[l];
        }

        conv[j1] = p11 + p12;
        conv[j2] = p21 + p22;
        conv[j3] = p31 + p32;
        conv[j4] = p41 + p42;
    }

    for (; j + 1 < llast; j += 2) {
        const size_t j1 = j;
        const size_t j2 = j + 1;

        auto r11 = vec_type::template zero<T>();
        auto r12 = vec_type::template zero<T>();

        auto r21 = vec_type::template zero<T>();
        auto r22 = vec_type::template zero<T>();

        size_t l = 0;

        for (; l + (vec_size * 2) - 1 < m; l += 2 * vec_size) {
            auto k1 = vec_type::load(kernel_reverse.get() + l + vec_size * 0);
            auto k2 = vec_type::load(kernel_reverse.get() + l + vec_size * 1);

            auto i11 = input.template loadu<vec_type>(j1 + l + vec_size * 0);
            auto i12 = input.template loadu<vec_type>(j1 + l + vec_size * 1);

            auto i21 = input.template loadu<vec_type>(j2 + l + vec_size * 0);
            auto i22 = input.template loadu<vec_type>(j2 + l + vec_size * 1);

            r11 = vec_type::fmadd(i11, k1, r11);
            r12 = vec_type::fmadd(i12, k2, r12);

            r21 = vec_type::fmadd(i21, k1, r21);
            r22 = vec_type::fmadd(i22, k2, r22);
        }

        for (; l + vec_size - 1 < m; l += vec_size) {
            auto i11 = input.template loadu<vec_type>(j1 + l);

            auto i21 = input.template loadu<vec_type>(j2 + l);

            auto k1 = vec_type::load(kernel_reverse.get() + l);

            r11 = vec_type::fmadd(i11, k1, r11);
            r21 = vec_type::fmadd(i21, k1, r21);
        }

        auto p11 = vec_type::hadd(vec_type::add(r11, r12));
        auto p12 = T(0);

        auto p21 = vec_type::hadd(vec_type::add(r21, r22));
        auto p22 = T(0);

        for (; l + 1 < m; l += 2) {
            p11 += input[j1 + l + 0] * kernel_reverse[l + 0];
            p12 += input[j1 + l + 1] * kernel_reverse[l + 1];

            p21 += input[j2 + l + 0] * kernel_reverse[l + 0];
            p22 += input[j2 + l + 1] * kernel_reverse[l + 1];
        }

        if (l < m) {
            p11 += input[j1 + l] * kernel_reverse[l];
            p21 += input[j2 + l] * kernel_reverse[l];
        }

        conv[j1] = p11 + p12;
        conv[j2] = p21 + p22;
    }

    if (j < llast) {
        const size_t j1 = j;

        auto r11 = vec_type::template zero<T>();
        auto r12 = vec_type::template zero<T>();

        size_t l = 0;

        for (; l + (vec_size * 2) - 1 < m; l += 2 * vec_size) {
            auto k1 = vec_type::load(kernel_reverse.get() + l + vec_size * 0);
            auto k2 = vec_type::load(kernel_reverse.get() + l + vec_size * 1);

            auto i11 = input.template loadu<vec_type>(j1 + l + vec_size * 0);
            auto i12 = input.template loadu<vec_type>(j1 + l + vec_size * 1);

            r11 = vec_type::fmadd(i11, k1, r11);
            r12 = vec_type::fmadd(i12, k2, r12);
        }

        for (; l + vec_size - 1 < m; l += vec_size) {
            auto i11 = input.template loadu<vec_type>(j1 + l);
            auto k1  = vec_type::load(kernel_reverse.get() + l);

            r11 = vec_type::fmadd(i11, k1, r11);
        }

        auto p11 = vec_type::hadd(vec_type::add(r11, r12));
        auto p12 = T(0);

        for (; l + 1 < m; l += 2) {
            p11 += input[j1 + l + 0] * kernel_reverse[l + 0];
            p12 += input[j1 + l + 1] * kernel_reverse[l + 1];
        }

        if (l < m) {
            p11 += input[j1 + l] * kernel_reverse[l];
        }

        conv[j1] = p11 + p12;
    }

    conv.invalidate_gpu();
}

/*!
 * \brief Vectorized implementation of a 1D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_valid(const I& input, const K& kernel, C&& conv, size_t first, size_t last) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    conv1_valid<default_vec>(input, kernel, conv, first, last);
}

/*!
 * \brief Vectorized implementation of a 1D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_full(const I& input, const K& kernel, C&& conv, size_t first, size_t last) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    size_t left = size(kernel) - 1;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    auto* out      = conv.memory_start();
    const auto* in = input.memory_start();
    const auto* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_full_kernel(in, size(input), k, size(kernel), out, first, last);
    etl::impl::common::right_full_kernel(in, size(input), k, size(kernel), out, first, last);

    conv1_valid<default_vec>(input, kernel, memory_slice(conv, left, size(conv)), first, last);
}

/*!
 * \brief Vectorized implementation of a 1D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_same(const I& input, const K& kernel, C&& conv, size_t first, size_t last) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    size_t left = (size(kernel) - 1) / 2;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    auto* out      = conv.memory_start();
    const auto* in = input.memory_start();
    const auto* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_same_kernel(in, size(input), k, size(kernel), out, first, last);
    etl::impl::common::right_same_kernel(in, size(input), k, size(kernel), out, first, last);

    conv1_valid<default_vec>(input, kernel, memory_slice(conv, left, size(conv)), first, last);
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv2_same_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T        = value_t<I>;
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t n1 = etl::dim<0>(input);
    const size_t n2 = etl::dim<1>(input);

    const size_t k1 = etl::dim<0>(kernel);
    const size_t k2 = etl::dim<1>(kernel);

    const size_t c1 = etl::dim<0>(conv);
    const size_t c2 = etl::dim<1>(conv);

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    for (size_t i = 0; i < c1; ++i) {
        size_t k_lo = std::max<int>(0, i - (k1 - 1) / 2);
        size_t k_hi = std::min<int>(n1 - 1, i + k1 / 2) + 1;

        for (size_t j = 0; j < c2; ++j) {
            size_t l_lo = std::max<int>(0, j - (k2 - 1) / 2);
            size_t l_hi = std::min<int>(n2 - 1, j + k2 / 2) + 1;

            auto r1 = vec_type::template zero<T>();
            auto r2 = vec_type::template zero<T>();

            auto temp1 = T(0);
            auto temp2 = T(0);

            for (size_t k = k_lo; k < k_hi; ++k) {
                const auto idx1 = k1 - 1 - i + k - k1 / 2;

                size_t l = l_lo;

                for (; l + 2 * vec_size - 1 < l_hi; l += 2 * vec_size) {
                    const auto idx2 = k2 - 1 - j + l - k2 / 2;

                    auto i1 = input.template loadu<vec_type>(k * n2 + l + vec_size * 0);
                    auto i2 = input.template loadu<vec_type>(k * n2 + l + vec_size * 1);

                    auto sk1 = kernel.template loadu<vec_type>(idx1 * k2 + idx2 + vec_size * 0);
                    auto sk2 = kernel.template loadu<vec_type>(idx1 * k2 + idx2 + vec_size * 1);

                    r1 = vec_type::fmadd(sk1, i1, r1);
                    r2 = vec_type::fmadd(sk2, i2, r2);
                }

                for (; l + vec_size - 1 < l_hi; l += vec_size) {
                    const auto idx2 = k2 - 1 - j + l - k2 / 2;

                    auto i1  = input.template loadu<vec_type>(k * n2 + l);
                    auto sk1 = kernel.template loadu<vec_type>(idx1 * k2 + idx2 + vec_size * 0);
                    r1       = vec_type::fmadd(sk1, i1, r1);
                }

                for (; l + 1 < l_hi; l += 2) {
                    const auto idx2 = k2 - 1 - j + l - k2 / 2;

                    temp1 += input(k, l + 0) * kernel(idx1, idx2 + 0);
                    temp2 += input(k, l + 1) * kernel(idx1, idx2 + 1);
                }

                if (l < l_hi) {
                    const auto idx2 = k2 - 1 - j + l - k2 / 2;

                    temp1 += input(k, l) * kernel(idx1, idx2);
                }
            }

            conv(i, j) = vec_type::hadd(r1) + vec_type::hadd(r2) + temp1 + temp2;
        }
    }

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K, with the
 * flipped kernels of K.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    conv2_same_flipped<default_vec>(input, kernel, conv);
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    const size_t k1 = etl::dim<0>(kernel);
    const size_t k2 = etl::dim<1>(kernel);

    etl::dyn_matrix<T, 2> kernel_reverse(k1, k2);

    std::reverse_copy(kernel.memory_start(), kernel.memory_start() + k1 * k2, kernel_reverse.memory_start());

    conv2_same_flipped<V>(input, kernel_reverse, conv);

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    conv2_same<default_vec>(input, kernel, conv);
}

/*!
 * \brief VEC implementation of a 2D 'same' convolution C = I * K, with multiple kernels
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_multi(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    const size_t Kn = etl::dim<0>(kernel);

    auto batch_fun_k = [&input, &kernel, &conv](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_same<default_vec>(input, kernel(k), conv(k));
        }
    };

    dispatch_1d_any(select_parallel(Kn, 2), batch_fun_k, 0, Kn);
}

/*!
 * \brief VEC implementation of a 2D 'same' convolution C = I * K, with multiple kernels, with kernels flipped.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    const size_t Kn = etl::dim<0>(kernel);

    auto batch_fun_k = [&input, &kernel, &conv](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_same_flipped<default_vec>(input, kernel(k), conv(k));
        }
    };

    dispatch_1d_any(select_parallel(Kn, 2), batch_fun_k, 0, Kn);
}

// TODO This need to be make much faster

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv2_full_flipped(const I& input, const K& kernel, C&& conv, value_t<I> beta) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T        = value_t<I>;
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t n1 = etl::dim<0>(input);
    const size_t n2 = etl::dim<1>(input);

    const size_t m1 = etl::dim<0>(kernel);
    const size_t m2 = etl::dim<1>(kernel);

    const size_t c1 = etl::dim<0>(conv);
    const size_t c2 = etl::dim<1>(conv);

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (beta != T(0)) {
        conv.ensure_cpu_up_to_date();
    }

    if (beta == T(0)) {
        for (size_t i = 0; i < c1; ++i) {
            const auto k_lo = std::max<int>(0, i - m1 + 1);
            const auto k_hi = std::min(n1 - 1, i) + 1;

            for (size_t j = 0; j < c2; ++j) {
                const auto l_lo = std::max<int>(0, j - m2 + 1);
                const auto l_hi = std::min(n2 - 1, j) + 1;

                auto r1    = vec_type::template zero<T>();
                auto temp1 = T(0);

                for (size_t k = k_lo; k < k_hi; ++k) {
                    const auto idx1 = m1 - 1 - i + k;

                    size_t l = l_lo;

                    for (; l + vec_size - 1 < l_hi; l += vec_size) {
                        const auto idx2 = m2 - 1 - j + l;

                        auto i1  = input.template loadu<vec_type>(k * n2 + l);
                        auto sk1 = kernel.template loadu<vec_type>(idx1 * m2 + idx2 + vec_size * 0);
                        r1       = vec_type::fmadd(sk1, i1, r1);
                    }

                    for (; l < l_hi; ++l) {
                        const auto idx2 = m2 - 1 - j + l;
                        temp1 += input(k, l) * kernel(idx1, idx2);
                    }
                }

                conv(i, j) = vec_type::hadd(r1) + temp1;
            }
        }
    } else {
        for (size_t i = 0; i < c1; ++i) {
            const auto k_lo = std::max<int>(0, i - m1 + 1);
            const auto k_hi = std::min(n1 - 1, i) + 1;

            for (size_t j = 0; j < c2; ++j) {
                const auto l_lo = std::max<int>(0, j - m2 + 1);
                const auto l_hi = std::min(n2 - 1, j) + 1;

                auto r1    = vec_type::template zero<T>();
                auto temp1 = T(0);

                for (size_t k = k_lo; k < k_hi; ++k) {
                    const auto idx1 = m1 - 1 - i + k;

                    size_t l = l_lo;

                    for (; l + vec_size - 1 < l_hi; l += vec_size) {
                        const auto idx2 = m2 - 1 - j + l;

                        auto i1  = input.template loadu<vec_type>(k * n2 + l);
                        auto sk1 = kernel.template loadu<vec_type>(idx1 * m2 + idx2 + vec_size * 0);
                        r1       = vec_type::fmadd(sk1, i1, r1);
                    }

                    for (; l < l_hi; ++l) {
                        const auto idx2 = m2 - 1 - j + l;
                        temp1 += input(k, l) * kernel(idx1, idx2);
                    }
                }

                conv(i, j) = beta * conv(i, j) + vec_type::hadd(r1) + temp1;
            }
        }
    }

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv2_full(const I& input, const K& kernel, C&& conv, value_t<I> beta) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t k1 = etl::dim<0>(kernel);
    const size_t k2 = etl::dim<1>(kernel);

    etl::dyn_matrix<T, 2> kernel_reverse(k1, k2);

    std::reverse_copy(kernel.memory_start(), kernel.memory_start() + k1 * k2, kernel_reverse.memory_start());

    conv2_full_flipped<V>(input, kernel_reverse, conv, beta);
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;
    conv2_full<default_vec>(input, kernel, conv, T(0.0));
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;
    conv2_full_flipped<default_vec>(input, kernel, conv, T(0));
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K, with multiple
 * kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t KK = etl::dim<0>(kernel);

    auto batch_fun_k = [&input, &kernel, &conv](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_full<default_vec>(input, kernel(k), conv(k), T(0));
        }
    };

    dispatch_1d_any(select_parallel(KK, 2), batch_fun_k, 0, KK);
}

/*!
 * \brief AVX implementation of a 2D 'full' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t KK = etl::dim<0>(kernel);

    auto batch_fun_k = [&input, &kernel, &conv](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_full_flipped<default_vec>(input, kernel(k), conv(k), T(0));
        }
    };

    dispatch_1d_any(select_parallel(KK, 2), batch_fun_k, 0, KK);
}

/*!
 * \brief SSE implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename KK, typename CC>
void conv4_full(const I& input, const KK& kernel, CC&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t N = etl::dim<0>(input);
    const size_t K = etl::dim<0>(kernel);
    const size_t C = etl::dim<1>(kernel);

    const size_t k1 = etl::dim<2>(kernel);
    const size_t k2 = etl::dim<3>(kernel);

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (C > 0) {
        etl::dyn_matrix<T, 4> prepared_k(K, C, k1, k2);

        std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

        prepared_k.deep_fflip_inplace();

        if (N > C) {
            auto batch_fun_n = [&](const size_t first, const size_t last) {
                if (last - first) {
                    for (size_t i = first; i < last; ++i) {
                        // k = 0
                        for (size_t c = 0; c < C; ++c) {
                            conv2_full_flipped<V>(input(i)(0), prepared_k(0)(c), conv(i)(c), T(0));
                        }

                        for (size_t k = 1; k < K; ++k) {
                            for (size_t c = 0; c < C; ++c) {
                                conv2_full_flipped<V>(input(i)(k), prepared_k(k)(c), conv(i)(c), T(1));
                            }
                        }
                    }
                }
            };

            dispatch_1d_any(select_parallel(N, 2), batch_fun_n, 0, N);
        } else {
            auto batch_fun_c = [&](const size_t first, const size_t last) {
                if (last - first) {
                    for (size_t i = 0; i < N; ++i) {
                        // k = 0
                        for (size_t c = first; c < last; ++c) {
                            conv2_full_flipped<V>(input(i)(0), prepared_k(0)(c), conv(i)(c), T(0));
                        }

                        for (size_t k = 1; k < K; ++k) {
                            for (size_t c = first; c < last; ++c) {
                                conv2_full_flipped<V>(input(i)(k), prepared_k(k)(c), conv(i)(c), T(1));
                            }
                        }
                    }
                }
            };

            dispatch_1d_any(select_parallel(C, 2), batch_fun_c, 0, C);
        }
    }

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_full(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    if(avx_enabled && sse3_enabled){
        const size_t k2 = etl::dim<3>(kernel);

        if (detail::prefer_sse<value_t<I>>(k2)) {
            return conv4_full<detail::safe_avx_vec>(input, kernel, conv);
        } else {
            return conv4_full<detail::safe_sse_vec>(input, kernel, conv);
        }
    } else {
        return conv4_full<default_vec>(input, kernel, conv);
    }
}

/*!
 * \brief SSE implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename KK, typename CC>
void conv4_full_flipped(const I& input, const KK& kernel, CC&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t N = etl::dim<0>(input);
    const size_t K = etl::dim<0>(kernel);
    const size_t C = etl::dim<1>(kernel);

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (C > 0) {
        if (N > C) {
            auto batch_fun_n = [&](const size_t first, const size_t last) {
                if (last - first) {
                    for (size_t i = first; i < last; ++i) {
                        // k = 0
                        for (size_t c = 0; c < C; ++c) {
                            conv2_full_flipped<V>(input(i)(0), kernel(0)(c), conv(i)(c), T(0));
                        }

                        for (size_t k = 1; k < K; ++k) {
                            for (size_t c = 0; c < C; ++c) {
                                conv2_full_flipped<V>(input(i)(k), kernel(k)(c), conv(i)(c), T(1));
                            }
                        }
                    }
                }
            };

            dispatch_1d_any(select_parallel(N, 2), batch_fun_n, 0, N);
        } else {
            auto batch_fun_c = [&](const size_t first, const size_t last) {
                if (last - first) {
                    for (size_t i = 0; i < N; ++i) {
                        // k = 0
                        for (size_t c = first; c < last; ++c) {
                            conv2_full_flipped<V>(input(i)(0), kernel(0)(c), conv(i)(c), T(0));
                        }

                        for (size_t k = 1; k < K; ++k) {
                            for (size_t c = first; c < last; ++c) {
                                conv2_full_flipped<V>(input(i)(k), kernel(k)(c), conv(i)(c), T(1));
                            }
                        }
                    }
                }
            };

            dispatch_1d_any(select_parallel(C, 2), batch_fun_c, 0, C);
        }
    }

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_full_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    if(avx_enabled && sse3_enabled){
        const size_t k2 = etl::dim<3>(kernel);

        if (detail::prefer_sse<value_t<I>>(k2)) {
            return conv4_full_flipped<detail::safe_avx_vec>(input, kernel, conv);
        } else {
            return conv4_full_flipped<detail::safe_sse_vec>(input, kernel, conv);
        }
    } else {
        return conv4_full_flipped<default_vec>(input, kernel, conv);
    }
}

/*!
 * \brief SSE implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename KK, typename C>
void conv2_valid_multi(const I& input, const KK& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t K  = etl::dim<0>(kernel);
    const size_t k2 = etl::dim<2>(kernel);

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (padding_impl) {
        static constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        static constexpr size_t SS = AS / 2;

        if (k2 < SS || k2 % AS > 0) {
            const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input  = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

            // TODO Test if it is better to do the padding of the kernel inside each thread

            if (detail::prefer_sse<T>(k2 + pad)) {
                auto fun_k = [&](const size_t first, const size_t last) {
                    for (size_t k = first; k < last; ++k) {
                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input, padded_kernel(k), conv(k), s1, s2, p1, p2, 0.0);
                    }
                };

                dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
            } else {
                auto fun_k = [&](const size_t first, const size_t last) {
                    for (size_t k = first; k < last; ++k) {
                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input, padded_kernel(k), conv(k), s1, s2, p1, p2, 0.0);
                    }
                };

                dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
            }

            return;
        }
    }

    if (detail::prefer_sse<T>(kernel.dim(2))) {
        auto fun_k = [&](const size_t first, const size_t last) {
            for (size_t k = first; k < last; ++k) {
                detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input, kernel(k), conv(k), s1, s2, p1, p2, 0.0);
            }
        };

        dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
    } else {
        auto fun_k = [&](const size_t first, const size_t last) {
            for (size_t k = first; k < last; ++k) {
                detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input, kernel(k), conv(k), s1, s2, p1, p2, 0.0);
            }
        };

        dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
    }

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename KK, typename C>
void conv2_valid_multi_flipped(const I& input, const KK& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t K  = etl::dim<0>(kernel);
    const size_t k2 = etl::dim<2>(kernel);

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (padding_impl) {
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if (k2 < SS || k2 % AS > 0) {
            const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input  = common::pad_right(input, pad);
            auto padded_kernel = common::pad_right_multi(kernel, pad);

            // TODO Test if it is better to do the padding of the kernel inside each thread

            if (detail::prefer_sse<T>(k2 + pad)) {
                auto fun_k = [&](const size_t first, const size_t last) {
                    for (size_t k = first; k < last; ++k) {
                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input, padded_kernel(k), conv(k), s1, s2, p1, p2, T(0));
                    }
                };

                dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
            } else {
                auto fun_k = [&](const size_t first, const size_t last) {
                    for (size_t k = first; k < last; ++k) {
                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input, padded_kernel(k), conv(k), s1, s2, p1, p2, T(0));
                    }
                };

                dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
            }

            return;
        }
    }

    if (detail::prefer_sse<T>(k2)) {
        auto fun_k = [&](const size_t first, const size_t last) {
            for (size_t k = first; k < last; ++k) {
                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input, kernel(k), conv(k), s1, s2, p1, p2, T(0));
            }
        };

        dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
    } else {
        auto fun_k = [&](const size_t first, const size_t last) {
            for (size_t k = first; k < last; ++k) {
                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input, kernel(k), conv(k), s1, s2, p1, p2, T(0));
            }
        };

        dispatch_1d_any(select_parallel(K, 2), fun_k, 0, K);
    }

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename KK, typename C>
void conv2_valid_multi_multi(const I& input, const KK& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t k2 = etl::dim<2>(kernel);
    const size_t K  = etl::dim<0>(kernel);
    const size_t N  = etl::dim<0>(input);
    const size_t KN = K * N;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (padding_impl) {
        static constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        static constexpr size_t SS = AS / 2;

        if (k2 < SS || k2 % AS > 0) {
            const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input  = common::pad_right_multi(input, pad);
            auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

            if (detail::prefer_sse<T>(k2 + pad)) {
                auto fun_kn = [&](const size_t first, const size_t last) {
                    for (size_t kn = first; kn < last; ++kn) {
                        size_t k = kn / N;
                        size_t n = kn % N;

                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(n), padded_kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                    }
                };

                dispatch_1d_any(select_parallel(KN, 2), fun_kn, 0, KN);
            } else {
                auto fun_kn = [&](const size_t first, const size_t last) {
                    for (size_t kn = first; kn < last; ++kn) {
                        size_t k = kn / N;
                        size_t n = kn % N;

                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(n), padded_kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                    }
                };

                dispatch_1d_any(select_parallel(KN, 2), fun_kn, 0, KN);
            }

            return;
        }
    }

    if (detail::prefer_sse<T>(kernel.dim(2))) {
        auto fun_kn = [&](const size_t first, const size_t last) {
            for (size_t kn = first; kn < last; ++kn) {
                size_t k = kn / N;
                size_t n = kn % N;

                detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input(n), kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
            }
        };

        dispatch_1d_any(select_parallel(KN, 2), fun_kn, 0, KN);
    } else {
        auto fun_kn = [&](const size_t first, const size_t last) {
            for (size_t kn = first; kn < last; ++kn) {
                size_t k = kn / N;
                size_t n = kn % N;

                detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input(n), kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
            }
        };

        dispatch_1d_any(select_parallel(KN, 2), fun_kn, 0, KN);
    }

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding (left and right)
 * \param p2 The second dimension padding (top and bottom)
 */
template <typename I, typename KK, typename C>
void conv2_valid_multi_multi_flipped(const I& input, const KK& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t k2 = etl::dim<2>(kernel);
    const size_t K  = etl::dim<0>(kernel);
    const size_t N  = etl::dim<0>(input);
    const size_t KN = K * N;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (padding_impl) {
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if (k2 < SS || k2 % AS > 0) {
            const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto padded_input  = common::pad_right_multi(input, pad);
            auto padded_kernel = common::pad_right_multi(kernel, pad);

            // TODO Test if it is better to do the padding of the kernel inside each thread

            if (detail::prefer_sse<T>(k2 + pad)) {
                auto fun_kn = [&](const size_t first, const size_t last) {
                    for (size_t kn = first; kn < last; ++kn) {
                        size_t k = kn / N;
                        size_t n = kn % N;

                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(n), padded_kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                    }
                };

                dispatch_1d_any(select_parallel(KN, 2), fun_kn, 0, KN);
            } else {
                auto fun_kn = [&](const size_t first, const size_t last) {
                    for (size_t kn = first; kn < last; ++kn) {
                        size_t k = kn / N;
                        size_t n = kn % N;

                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(n), padded_kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                    }
                };

                dispatch_1d_any(select_parallel(KN, 2), fun_kn, 0, KN);
            }

            return;
        }
    }

    if (detail::prefer_sse<T>(kernel.dim(2))) {
        auto fun_kn = [&](const size_t first, const size_t last) {
            for (size_t kn = first; kn < last; ++kn) {
                size_t k = kn / N;
                size_t n = kn % N;

                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input(n), kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
            }
        };

        dispatch_1d_any(select_parallel(KN, 2), fun_kn, 0, KN);
    } else {
        auto fun_kn = [&](const size_t first, const size_t last) {
            for (size_t kn = first; kn < last; ++kn) {
                size_t k = kn / N;
                size_t n = kn % N;

                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input(n), kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
            }
        };

        dispatch_1d_any(select_parallel(KN, 2), fun_kn, 0, KN);
    }

    conv.invalidate_gpu();
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_valid(const I& input, const KK& kernel, CC&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    if (etl::dim<1>(kernel) > 0) {
        const size_t N = etl::dim<0>(input);  // The number of images
        const size_t K = etl::dim<0>(kernel); // The number of kernels
        const size_t C = etl::dim<1>(input);  // The number of channels

        const size_t k2 = etl::dim<3>(kernel);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        conv = 0;

        if (padding_impl) {
            static constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
            static constexpr size_t SS = AS / 2;

            if (k2 < SS || k2 % AS > 0) {
                const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

                auto padded_input  = common::pad_right_multi(input, pad);
                auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

                if (detail::prefer_sse<T>(k2 + pad)) {
                    auto fun_nk = [&](const size_t first, const size_t last) {
                        for (size_t nk = first; nk < last; ++nk) {
                            const size_t i = nk / K;
                            const size_t k = nk % K;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2, p1, p2, T(0));

                            for (size_t c = 1; c < C; ++c) {
                                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1, s2, p1, p2, T(1));
                            }
                        }
                    };

                    dispatch_1d_any(select_parallel(K * N, 4), fun_nk, 0, K * N);
                } else {
                    auto fun_nk = [&](const size_t first, const size_t last) {
                        for (size_t nk = first; nk < last; ++nk) {
                            const size_t i = nk / K;
                            const size_t k = nk % K;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2, p1, p2, T(0));

                            for (size_t c = 1; c < C; ++c) {
                                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1, s2, p1, p2, T(1));
                            }
                        }
                    };

                    dispatch_1d_any(select_parallel(K * N, 4), fun_nk, 0, K * N);
                }

                return;
            }
        }

        if (detail::prefer_sse<T>(k2)) {
            auto fun_nk = [&](const size_t first, const size_t last) {
                for (size_t nk = first; nk < last; ++nk) {
                    const size_t i = nk / K;
                    const size_t k = nk % K;

                    detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, p1, p2, T(0));

                    for (size_t c = 1; c < C; ++c) {
                        detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, p1, p2, T(1));
                    }
                }
            };

            dispatch_1d_any(select_parallel(K * N, 4), fun_nk, 0, K * N);
        } else {
            auto fun_nk = [&](const size_t first, const size_t last) {
                for (size_t nk = first; nk < last; ++nk) {
                    const size_t i = nk / K;
                    const size_t k = nk % K;

                    detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, p1, p2, T(0));

                    for (size_t c = 1; c < C; ++c) {
                        detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, p1, p2, T(1));
                    }
                }
            };

            dispatch_1d_any(select_parallel(K * N, 4), fun_nk, 0, K * N);
        }

        conv.invalidate_gpu();
    }
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_valid_flipped(const I& input, const KK& kernel, CC&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    if (etl::dim<0>(kernel) > 0) {
        const size_t N = etl::dim<0>(input);  // The number of images
        const size_t K = etl::dim<0>(kernel); // The number of kernels
        const size_t C = etl::dim<1>(input);  // The number of channels

        const size_t k2 = etl::dim<3>(kernel);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        if (padding_impl) {
            static constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
            static constexpr size_t SS = AS / 2;

            if (k2 < SS || k2 % AS > 0) {
                const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

                auto padded_input  = common::pad_right_multi(input, pad);
                auto padded_kernel = common::pad_right_multi(kernel, pad);

                if (detail::prefer_sse<T>(k2 + pad)) {
                    auto fun_nk = [&](const size_t first, const size_t last) {
                        for (size_t nk = first; nk < last; ++nk) {
                            const size_t i = nk / K;
                            const size_t k = nk % K;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2, p1, p2, T(0));

                            for (size_t c = 1; c < C; ++c) {
                                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1, s2, p1, p2, T(1));
                            }
                        }
                    };

                    dispatch_1d_any(select_parallel(K * N, 4), fun_nk, 0, K * N);
                } else {
                    auto fun_nk = [&](const size_t first, const size_t last) {
                        for (size_t nk = first; nk < last; ++nk) {
                            const size_t i = nk / K;
                            const size_t k = nk % K;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2, p1, p2, T(0));

                            for (size_t c = 1; c < C; ++c) {
                                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1, s2, p1, p2, T(1));
                            }
                        }
                    };

                    dispatch_1d_any(select_parallel(K * N, 4), fun_nk, 0, K * N);
                }

                return;
            }
        }

        if (detail::prefer_sse<T>(k2)) {
            auto fun_nk = [&](const size_t first, const size_t last) {
                for (size_t nk = first; nk < last; ++nk) {
                    const size_t i = nk / K;
                    const size_t k = nk % K;

                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, p1, p2, T(0));

                    for (size_t c = 1; c < C; ++c) {
                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, p1, p2, T(1));
                    }
                }
            };

            dispatch_1d_any(select_parallel(K * N, 4), fun_nk, 0, K * N);
        } else {
            auto fun_nk = [&](const size_t first, const size_t last) {
                for (size_t nk = first; nk < last; ++nk) {
                    const size_t i = nk / K;
                    const size_t k = nk % K;

                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, p1, p2, T(0));

                    for (size_t c = 1; c < C; ++c) {
                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, p1, p2, T(1));
                    }
                }
            };

            dispatch_1d_any(select_parallel(K * N, 4), fun_nk, 0, K * N);
        }

        conv.invalidate_gpu();
    }
}

/*!
 * \brief AVX implementation of a 4D 'valid' convolution C = I * K, where the output are considered to be kernels
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_valid_filter(const I& input, const KK& kernel, CC&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    if (input.dim(0) > 0) {
        const size_t N = input.dim(0);  // The number of images
        const size_t C = input.dim(1);  // The number of channels
        const size_t K = kernel.dim(1); // The number of kernels

        const size_t k2 = kernel.dim(3);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        if (padding_impl) {
            static constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
            static constexpr size_t SS = AS / 2;

            if (k2 < SS || k2 % AS > 0) {
                const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

                auto padded_input  = common::pad_right_multi(input, pad);
                auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

                if (detail::prefer_sse<T>(k2 + pad)) {
                    auto fun_kc = [&](const size_t first, const size_t last) {
                        //i = 0
                        for (size_t kc = first; kc < last; ++kc) {
                            const size_t k = kc / C;
                            const size_t c = kc % C;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2, p1, p2, T(0));
                        }

                        for (size_t i = 1; i < N; ++i) {
                            for (size_t kc = first; kc < last; ++kc) {
                                const size_t k = kc / C;
                                const size_t c = kc % C;

                                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1, s2, p1, p2, T(1));
                            }
                        }
                    };

                    dispatch_1d_any(select_parallel(K * C, 4), fun_kc, 0, K * C);
                } else {
                    auto fun_kc = [&](const size_t first, const size_t last) {
                        //i = 0
                        for (size_t kc = first; kc < last; ++kc) {
                            const size_t k = kc / C;
                            const size_t c = kc % C;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2, p1, p2, T(0));
                        }

                        for (size_t i = 1; i < N; ++i) {
                            for (size_t kc = first; kc < last; ++kc) {
                                const size_t k = kc / C;
                                const size_t c = kc % C;

                                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1, s2, p1, p2, T(1));
                            }
                        }
                    };

                    dispatch_1d_any(select_parallel(K * C, 4), fun_kc, 0, K * C);
                }

                return;
            }
        }

        if (detail::prefer_sse<T>(k2)) {
            auto fun_kc = [&](const size_t first, const size_t last) {
                //i = 0
                for (size_t kc = first; kc < last; ++kc) {
                    const size_t k = kc / C;
                    const size_t c = kc % C;

                    detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, p1, p2, T(0));
                }

                for (size_t i = 1; i < N; ++i) {
                    for (size_t kc = first; kc < last; ++kc) {
                        const size_t k = kc / C;
                        const size_t c = kc % C;

                        detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, p1, p2, T(1));
                    }
                }
            };

            dispatch_1d_any(select_parallel(K * C, 4), fun_kc, 0, K * C);
        } else {
            auto fun_kc = [&](const size_t first, const size_t last) {
                //i = 0
                for (size_t kc = first; kc < last; ++kc) {
                    const size_t k = kc / C;
                    const size_t c = kc % C;

                    detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, p1, p2, T(0));
                }

                for (size_t i = 1; i < N; ++i) {
                    for (size_t kc = first; kc < last; ++kc) {
                        const size_t k = kc / C;
                        const size_t c = kc % C;

                        detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, p1, p2, T(1));
                    }
                }
            };

            dispatch_1d_any(select_parallel(K * C, 4), fun_kc, 0, K * C);
        }

        conv.invalidate_gpu();
    }
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K, where the output
 * are considered to be kernels, with flipped weights
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_valid_filter_flipped(const I& input, const KK& kernel, CC&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    if (input.dim(0) > 0) {
        const size_t N = input.dim(0);  // The number of images
        const size_t C = input.dim(1);  // The number of channels
        const size_t K = kernel.dim(1); // The number of kernels

        const size_t k2 = kernel.dim(3);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        if (padding_impl) {
            constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
            constexpr size_t SS = AS / 2;

            if (k2 < SS || k2 % AS > 0) {
                const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

                auto padded_input  = common::pad_right_multi(input, pad);
                auto padded_kernel = common::pad_right_multi(kernel, pad);

                if (detail::prefer_sse<T>(k2 + pad)) {
                    auto fun_kc = [&](const size_t first, const size_t last) {
                        //i = 0
                        for (size_t kc = first; kc < last; ++kc) {
                            const size_t k = kc / C;
                            const size_t c = kc % C;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2, p1, p2, T(0));
                        }

                        for (size_t i = 1; i < N; ++i) {
                            for (size_t kc = first; kc < last; ++kc) {
                                const size_t k = kc / C;
                                const size_t c = kc % C;

                                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1, s2, p1, p2, T(1));
                            }
                        }
                    };

                    dispatch_1d_any(select_parallel(K * C, 4), fun_kc, 0, K * C);
                } else {
                    auto fun_kc = [&](const size_t first, const size_t last) {
                        //i = 0
                        for (size_t kc = first; kc < last; ++kc) {
                            const size_t k = kc / C;
                            const size_t c = kc % C;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2, p1, p2, T(0));
                        }

                        for (size_t i = 1; i < N; ++i) {
                            for (size_t kc = first; kc < last; ++kc) {
                                const size_t k = kc / C;
                                const size_t c = kc % C;

                                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1, s2, p1, p2, T(1));
                            }
                        }
                    };

                    dispatch_1d_any(select_parallel(K * C, 4), fun_kc, 0, K * C);
                }

                return;
            }
        }

        if (detail::prefer_sse<T>(k2)) {
            auto fun_kc = [&](const size_t first, const size_t last) {
                //i = 0
                for (size_t kc = first; kc < last; ++kc) {
                    const size_t k = kc / C;
                    const size_t c = kc % C;

                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, p1, p2, T(0));
                }

                for (size_t i = 1; i < N; ++i) {
                    for (size_t kc = first; kc < last; ++kc) {
                        const size_t k = kc / C;
                        const size_t c = kc % C;

                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, p1, p2, T(1));
                    }
                }
            };

            dispatch_1d_any(select_parallel(K * C, 4), fun_kc, 0, K * C);
        } else {
            auto fun_kc = [&](const size_t first, const size_t last) {
                //i = 0
                for (size_t kc = first; kc < last; ++kc) {
                    const size_t k = kc / C;
                    const size_t c = kc % C;

                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, p1, p2, T(0));
                }

                for (size_t i = 1; i < N; ++i) {
                    for (size_t kc = first; kc < last; ++kc) {
                        const size_t k = kc / C;
                        const size_t c = kc % C;

                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, p1, p2, T(1));
                    }
                }
            };

            dispatch_1d_any(select_parallel(K * C, 4), fun_kc, 0, K * C);
        }

        conv.invalidate_gpu();
    }
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
