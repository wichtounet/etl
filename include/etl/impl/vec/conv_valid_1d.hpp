//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/common/conv.hpp"

namespace etl::impl::vec {

/*!
 * \brief Vectorized implementation of a 1D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv1_valid_impl(const I& input, const K& kernel, C&& conv, size_t first, size_t last) {
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
void conv1_valid(const I& input, const K& kernel, C&& conv, [[maybe_unused]] size_t first, [[maybe_unused]] size_t last) {
    if constexpr (conv1_possible<vector_mode, I, K, C>) {
        conv1_valid_impl<default_vec>(input, kernel, conv, first, last);
    } else {
        cpp_unreachable("Invalid call to vec::conv_1_valid");
    }
}

} //end of namespace etl::impl::vec
