//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

namespace vec {

namespace detail {

template <typename V, typename I, typename K, typename C>
void conv2_valid_flipped(const I& input, const K& kernel, C&& conv) {
    using T = value_t<I>;
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr size_t unroll = 8;

    const auto n1 = etl::dim<0>(input);
    const auto n2 = etl::dim<1>(input);

    const auto k1 = etl::dim<0>(kernel);
    const auto k2 = etl::dim<1>(kernel);

    const auto c1 = etl::dim<0>(conv);
    const auto c2 = etl::dim<1>(conv);

    const auto R = std::min(k1, c1); // Max number of kernels per line of input

    conv = T(0);

    // Primary steps
    for(size_t i = 0; i < k1 - 1; ++i){
        const auto M = std::min(i + 1, R);

        for(size_t m = 0; m < M; ++m){
            const auto k_i = i - m;

            size_t j = 0;

            for(; j + unroll - 1 < c2; j += unroll){
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                size_t k = 0;
                for(; k + vec_size - 1 < k2; k += vec_size){
                    auto b1 = kernel.template loadu<vec_type>(k_i * k2 + k);

                    auto a1 = input.template loadu<vec_type>(i * n2 + (j + 0) + k);
                    auto a2 = input.template loadu<vec_type>(i * n2 + (j + 1) + k);
                    auto a3 = input.template loadu<vec_type>(i * n2 + (j + 2) + k);
                    auto a4 = input.template loadu<vec_type>(i * n2 + (j + 3) + k);
                    auto a5 = input.template loadu<vec_type>(i * n2 + (j + 4) + k);
                    auto a6 = input.template loadu<vec_type>(i * n2 + (j + 5) + k);
                    auto a7 = input.template loadu<vec_type>(i * n2 + (j + 6) + k);
                    auto a8 = input.template loadu<vec_type>(i * n2 + (j + 7) + k);

                    auto t1 = vec_type::template mul<false>(a1, b1);
                    auto t2 = vec_type::template mul<false>(a2, b1);
                    auto t3 = vec_type::template mul<false>(a3, b1);
                    auto t4 = vec_type::template mul<false>(a4, b1);
                    auto t5 = vec_type::template mul<false>(a5, b1);
                    auto t6 = vec_type::template mul<false>(a6, b1);
                    auto t7 = vec_type::template mul<false>(a7, b1);
                    auto t8 = vec_type::template mul<false>(a8, b1);

                    r1 = vec_type::add(r1, t1);
                    r2 = vec_type::add(r2, t2);
                    r3 = vec_type::add(r3, t3);
                    r4 = vec_type::add(r4, t4);
                    r5 = vec_type::add(r5, t5);
                    r6 = vec_type::add(r6, t6);
                    r7 = vec_type::add(r7, t7);
                    r8 = vec_type::add(r8, t8);
                }

                T v1 = vec_type::hadd(r1);
                T v2 = vec_type::hadd(r2);
                T v3 = vec_type::hadd(r3);
                T v4 = vec_type::hadd(r4);
                T v5 = vec_type::hadd(r5);
                T v6 = vec_type::hadd(r6);
                T v7 = vec_type::hadd(r7);
                T v8 = vec_type::hadd(r8);

                for(; !padding_impl && k < k2; ++k){
                    v1 += input(i, (j + 0) + k) * kernel(k_i, k);
                    v2 += input(i, (j + 1) + k) * kernel(k_i, k);
                    v3 += input(i, (j + 2) + k) * kernel(k_i, k);
                    v4 += input(i, (j + 3) + k) * kernel(k_i, k);
                    v5 += input(i, (j + 4) + k) * kernel(k_i, k);
                    v6 += input(i, (j + 5) + k) * kernel(k_i, k);
                    v7 += input(i, (j + 6) + k) * kernel(k_i, k);
                    v8 += input(i, (j + 7) + k) * kernel(k_i, k);
                }

                conv(m, j+0) += v1;
                conv(m, j+1) += v2;
                conv(m, j+2) += v3;
                conv(m, j+3) += v4;
                conv(m, j+4) += v5;
                conv(m, j+5) += v6;
                conv(m, j+6) += v7;
                conv(m, j+7) += v8;
            }

            for(; j < c2;  ++j){
                auto r1 = vec_type::template zero<T>();

                size_t k = 0;
                for(; k + vec_size - 1 < k2; k += vec_size){
                    auto a1 = input.template loadu<vec_type>(i * n2 + j + k);
                    auto b1 = kernel.template loadu<vec_type>(k_i * k2 + k);

                    auto t1 = vec_type::template mul<false>(a1, b1);
                    r1 = vec_type::add(r1, t1);
                }

                T value = vec_type::hadd(r1);

                for(; !padding_impl && k < k2; ++k){
                    value += input(i, j + k) * kernel(k_i, k);
                }

                conv(m,j) += value;
            }
        }
    }

    // Main steps
    for(size_t i = k1 - 1; i < c1; ++i){
        const auto M = R;

        for(size_t m = 0; m < M; ++m){
            const auto c_i = i - m;

            size_t j = 0;

            for(; j + unroll - 1 < c2; j += unroll){
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                size_t k = 0;
                for(; k + vec_size - 1 < k2; k += vec_size){
                    auto b1 = kernel.template loadu<vec_type>(m * k2 + k);

                    auto a1 = input.template loadu<vec_type>(i * n2 + (j + 0) + k);
                    auto a2 = input.template loadu<vec_type>(i * n2 + (j + 1) + k);
                    auto a3 = input.template loadu<vec_type>(i * n2 + (j + 2) + k);
                    auto a4 = input.template loadu<vec_type>(i * n2 + (j + 3) + k);
                    auto a5 = input.template loadu<vec_type>(i * n2 + (j + 4) + k);
                    auto a6 = input.template loadu<vec_type>(i * n2 + (j + 5) + k);
                    auto a7 = input.template loadu<vec_type>(i * n2 + (j + 6) + k);
                    auto a8 = input.template loadu<vec_type>(i * n2 + (j + 7) + k);

                    auto t1 = vec_type::template mul<false>(a1, b1);
                    auto t2 = vec_type::template mul<false>(a2, b1);
                    auto t3 = vec_type::template mul<false>(a3, b1);
                    auto t4 = vec_type::template mul<false>(a4, b1);
                    auto t5 = vec_type::template mul<false>(a5, b1);
                    auto t6 = vec_type::template mul<false>(a6, b1);
                    auto t7 = vec_type::template mul<false>(a7, b1);
                    auto t8 = vec_type::template mul<false>(a8, b1);

                    r1 = vec_type::add(r1, t1);
                    r2 = vec_type::add(r2, t2);
                    r3 = vec_type::add(r3, t3);
                    r4 = vec_type::add(r4, t4);
                    r5 = vec_type::add(r5, t5);
                    r6 = vec_type::add(r6, t6);
                    r7 = vec_type::add(r7, t7);
                    r8 = vec_type::add(r8, t8);
                }

                T v1 = vec_type::hadd(r1);
                T v2 = vec_type::hadd(r2);
                T v3 = vec_type::hadd(r3);
                T v4 = vec_type::hadd(r4);
                T v5 = vec_type::hadd(r5);
                T v6 = vec_type::hadd(r6);
                T v7 = vec_type::hadd(r7);
                T v8 = vec_type::hadd(r8);

                for(; !padding_impl && k < k2; ++k){
                    v1 += input(i, (j + 0) + k) * kernel(m, k);
                    v2 += input(i, (j + 1) + k) * kernel(m, k);
                    v3 += input(i, (j + 2) + k) * kernel(m, k);
                    v4 += input(i, (j + 3) + k) * kernel(m, k);
                    v5 += input(i, (j + 4) + k) * kernel(m, k);
                    v6 += input(i, (j + 5) + k) * kernel(m, k);
                    v7 += input(i, (j + 6) + k) * kernel(m, k);
                    v8 += input(i, (j + 7) + k) * kernel(m, k);
                }

                conv(c_i, j+0) += v1;
                conv(c_i, j+1) += v2;
                conv(c_i, j+2) += v3;
                conv(c_i, j+3) += v4;
                conv(c_i, j+4) += v5;
                conv(c_i, j+5) += v6;
                conv(c_i, j+6) += v7;
                conv(c_i, j+7) += v8;
            }

            for(; j < c2; ++j){
                auto r1 = vec_type::template zero<T>();

                size_t k = 0;
                for(; k + vec_size - 1 < k2; k += vec_size){
                    auto a1 = input.template loadu<vec_type>(i * n2 + j + k);
                    auto b1 = kernel.template loadu<vec_type>(m * k2 + k);

                    auto t1 = vec_type::template mul<false>(a1, b1);
                    r1 = vec_type::add(r1, t1);
                }

                T value = vec_type::hadd(r1);

                for(; !padding_impl && k < k2; ++k){
                    value += input(i, j + k) * kernel(m, k);
                }

                conv(c_i, j) += value;
            }
        }
    }

    // Secondary steps
    for(size_t i = c1; i < n1; ++i){
        auto M = std::min(n1 - i, R);

        for(size_t m = 0; m < M; ++m){
            const auto c_i = m + i - k1 + 1;
            const auto k_i = M - m - c1 + i;

            size_t j = 0;

            for(; j + unroll - 1 < c2; j += unroll){
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                size_t k = 0;
                for(; k + vec_size - 1 < k2; k += vec_size){
                    auto b1 = kernel.template loadu<vec_type>(k_i * k2 + k);

                    auto a1 = input.template loadu<vec_type>(i * n2 + (j + 0) + k);
                    auto a2 = input.template loadu<vec_type>(i * n2 + (j + 1) + k);
                    auto a3 = input.template loadu<vec_type>(i * n2 + (j + 2) + k);
                    auto a4 = input.template loadu<vec_type>(i * n2 + (j + 3) + k);
                    auto a5 = input.template loadu<vec_type>(i * n2 + (j + 4) + k);
                    auto a6 = input.template loadu<vec_type>(i * n2 + (j + 5) + k);
                    auto a7 = input.template loadu<vec_type>(i * n2 + (j + 6) + k);
                    auto a8 = input.template loadu<vec_type>(i * n2 + (j + 7) + k);

                    auto t1 = vec_type::template mul<false>(a1, b1);
                    auto t2 = vec_type::template mul<false>(a2, b1);
                    auto t3 = vec_type::template mul<false>(a3, b1);
                    auto t4 = vec_type::template mul<false>(a4, b1);
                    auto t5 = vec_type::template mul<false>(a5, b1);
                    auto t6 = vec_type::template mul<false>(a6, b1);
                    auto t7 = vec_type::template mul<false>(a7, b1);
                    auto t8 = vec_type::template mul<false>(a8, b1);

                    r1 = vec_type::add(r1, t1);
                    r2 = vec_type::add(r2, t2);
                    r3 = vec_type::add(r3, t3);
                    r4 = vec_type::add(r4, t4);
                    r5 = vec_type::add(r5, t5);
                    r6 = vec_type::add(r6, t6);
                    r7 = vec_type::add(r7, t7);
                    r8 = vec_type::add(r8, t8);
                }

                T v1 = vec_type::hadd(r1);
                T v2 = vec_type::hadd(r2);
                T v3 = vec_type::hadd(r3);
                T v4 = vec_type::hadd(r4);
                T v5 = vec_type::hadd(r5);
                T v6 = vec_type::hadd(r6);
                T v7 = vec_type::hadd(r7);
                T v8 = vec_type::hadd(r8);

                for(; !padding_impl && k < k2; ++k){
                    v1 += input(i, (j + 0) + k) * kernel(k_i, k);
                    v2 += input(i, (j + 1) + k) * kernel(k_i, k);
                    v3 += input(i, (j + 2) + k) * kernel(k_i, k);
                    v4 += input(i, (j + 3) + k) * kernel(k_i, k);
                    v5 += input(i, (j + 4) + k) * kernel(k_i, k);
                    v6 += input(i, (j + 5) + k) * kernel(k_i, k);
                    v7 += input(i, (j + 6) + k) * kernel(k_i, k);
                    v8 += input(i, (j + 7) + k) * kernel(k_i, k);
                }

                conv(c_i, j+0) += v1;
                conv(c_i, j+1) += v2;
                conv(c_i, j+2) += v3;
                conv(c_i, j+3) += v4;
                conv(c_i, j+4) += v5;
                conv(c_i, j+5) += v6;
                conv(c_i, j+6) += v7;
                conv(c_i, j+7) += v8;
            }

            for(; j < c2;  ++j){
                auto r1 = vec_type::template zero<T>();

                size_t k = 0;
                for(; k + vec_size - 1 < k2; k += vec_size){
                    auto a1 = input.template loadu<vec_type>(i * n2 + j + k);
                    auto b1 = kernel.template loadu<vec_type>(k_i * k2 + k);

                    auto t1 = vec_type::template mul<false>(a1, b1);
                    r1 = vec_type::add(r1, t1);
                }

                T value = vec_type::hadd(r1);

                for(; !padding_impl && k < k2; ++k){
                    value += input(i, j + k) * kernel(k_i, k);
                }

                conv(c_i, j) += value;
            }
        }
    }
}

template<typename T>
constexpr bool prefer_sse(const size_t n){
    return
           !avx_enabled
        || (
                sse3_enabled
            &&  (std::is_same<T, float>::value
                    ? (n % 4 < n % 8)
                    : (n % 2 < n % 4))
           );
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
    using T = value_t<I>;

    const auto n1 = etl::dim<0>(input);
    const auto n2 = etl::dim<1>(input);

    const auto k1 = etl::dim<0>(kernel);
    const auto k2 = etl::dim<1>(kernel);

    const auto c1 = etl::dim<0>(conv);
    const auto c2 = etl::dim<1>(conv);

    if(cpp_unlikely(p1 || p2)){
        const auto o1 = n1 + 2 * p1;
        const auto o2 = n2 + 2 * p2;

        etl::dyn_matrix<T> padded_matrix(o1, o2);
        padded_matrix = T(0);

        for (size_t i = 0; i < n1; ++i) {
            for(size_t j = 0; j < n2; ++j){
                padded_matrix[(i + p1) * o2 + p2 + j] = input(i, j);
            }
        }

        conv2_valid_flipped(padded_matrix, kernel, conv, s1, s2, 0, 0);

        return;
    }

    if(cpp_unlikely(s1 > 1 || s2 > 1)){
        etl::dyn_matrix<T> tmp_result(n1 - k1 + 1, n2 - k2 + 1);

        conv2_valid_flipped(input, kernel, tmp_result, 1, 1, 0, 0);

        // Strided copy of the large result into the small result
        for (std::size_t i = 0; i < c1; ++i) {
            for (std::size_t j = 0; j < c2; ++j) {
                conv(i, j) = tmp_result(i * s1, j * s2);
            }
        }

        return;
    }

    constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
    constexpr size_t SS = AS / 2;

    if(padding_impl && k2 % SS > 0){
        const auto pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

        etl::dyn_matrix<T, 2> padded_kernel(k1, k2 + pad);
        etl::dyn_matrix<T, 2> padded_input(n1, n2 + pad);

        padded_kernel = 0;
        padded_input = 0;

        for(size_t i = 0; i < k1; ++i){
            direct_copy_n(kernel.memory_start() + i * k2, padded_kernel.memory_start() + i * padded_kernel.dim(1), k2);
        }

        for(size_t i = 0; i < n1; ++i){
            direct_copy_n(input.memory_start() + i * n2, padded_input.memory_start() + i * padded_input.dim(1), n2);
        }

        if(detail::prefer_sse<T>(k2 + pad)){
            detail::conv2_valid_flipped<detail::safe_sse_vec>(padded_input, padded_kernel, conv);
        } else {
            detail::conv2_valid_flipped<detail::safe_avx_vec>(padded_input, padded_kernel, conv);
        }

        return;
    }

    if(detail::prefer_sse<T>(k2)){
        detail::conv2_valid_flipped<detail::safe_sse_vec>(input, kernel, conv);
    } else {
        detail::conv2_valid_flipped<detail::safe_avx_vec>(input, kernel, conv);
    }
}

template <typename V, typename I, typename K, typename C>
void conv1_valid(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    using vec_type = V;
    using T        = value_t<I>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;
    static constexpr bool Cx         = is_complex_t<T>::value;

    const size_t m = etl::size(kernel);

    auto llast = std::min(etl::size(conv), last);

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

            r11 = vec_type::template fmadd<Cx>(i11, k1, r11);
            r21 = vec_type::template fmadd<Cx>(i21, k1, r21);
            r31 = vec_type::template fmadd<Cx>(i31, k1, r31);
            r41 = vec_type::template fmadd<Cx>(i41, k1, r41);
            r51 = vec_type::template fmadd<Cx>(i51, k1, r51);
            r61 = vec_type::template fmadd<Cx>(i61, k1, r61);
            r71 = vec_type::template fmadd<Cx>(i71, k1, r71);
            r81 = vec_type::template fmadd<Cx>(i81, k1, r81);
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

        for (; l + (vec_size * 2)- 1 < m; l += 2 * vec_size) {
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

            r11 = vec_type::template fmadd<Cx>(i11, k1, r11);
            r12 = vec_type::template fmadd<Cx>(i12, k2, r12);

            r21 = vec_type::template fmadd<Cx>(i21, k1, r21);
            r22 = vec_type::template fmadd<Cx>(i22, k2, r22);

            r31 = vec_type::template fmadd<Cx>(i31, k1, r31);
            r32 = vec_type::template fmadd<Cx>(i32, k2, r32);

            r41 = vec_type::template fmadd<Cx>(i41, k1, r41);
            r42 = vec_type::template fmadd<Cx>(i42, k2, r42);
        }

        for (; l + vec_size - 1 < m; l += vec_size) {
            auto i11 = input.template loadu<vec_type>(j1 + l);
            auto i21 = input.template loadu<vec_type>(j2 + l);
            auto i31 = input.template loadu<vec_type>(j3 + l);
            auto i41 = input.template loadu<vec_type>(j4 + l);

            auto k1 = vec_type::load(kernel_reverse.get() + l);

            r11 = vec_type::template fmadd<Cx>(i11, k1, r11);
            r21 = vec_type::template fmadd<Cx>(i21, k1, r21);
            r31 = vec_type::template fmadd<Cx>(i31, k1, r31);
            r41 = vec_type::template fmadd<Cx>(i41, k1, r41);
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

        for (; l + (vec_size * 2)- 1 < m; l += 2 * vec_size) {
            auto k1 = vec_type::load(kernel_reverse.get() + l + vec_size * 0);
            auto k2 = vec_type::load(kernel_reverse.get() + l + vec_size * 1);

            auto i11 = input.template loadu<vec_type>(j1 + l + vec_size * 0);
            auto i12 = input.template loadu<vec_type>(j1 + l + vec_size * 1);

            auto i21 = input.template loadu<vec_type>(j2 + l + vec_size * 0);
            auto i22 = input.template loadu<vec_type>(j2 + l + vec_size * 1);

            r11 = vec_type::template fmadd<Cx>(i11, k1, r11);
            r12 = vec_type::template fmadd<Cx>(i12, k2, r12);

            r21 = vec_type::template fmadd<Cx>(i21, k1, r21);
            r22 = vec_type::template fmadd<Cx>(i22, k2, r22);
        }

        for (; l + vec_size - 1 < m; l += vec_size) {
            auto i11 = input.template loadu<vec_type>(j1 + l);

            auto i21 = input.template loadu<vec_type>(j2 + l);

            auto k1 = vec_type::load(kernel_reverse.get() + l);

            r11 = vec_type::template fmadd<Cx>(i11, k1, r11);
            r21 = vec_type::template fmadd<Cx>(i21, k1, r21);
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

        for (; l + (vec_size * 2)- 1 < m; l += 2 * vec_size) {
            auto k1 = vec_type::load(kernel_reverse.get() + l + vec_size * 0);
            auto k2 = vec_type::load(kernel_reverse.get() + l + vec_size * 1);

            auto i11 = input.template loadu<vec_type>(j1 + l + vec_size * 0);
            auto i12 = input.template loadu<vec_type>(j1 + l + vec_size * 1);

            r11 = vec_type::template fmadd<Cx>(i11, k1, r11);
            r12 = vec_type::template fmadd<Cx>(i12, k2, r12);
        }

        for (; l + vec_size - 1 < m; l += vec_size) {
            auto i11 = input.template loadu<vec_type>(j1 + l);
            auto k1 = vec_type::load(kernel_reverse.get() + l);

            r11 = vec_type::template fmadd<Cx>(i11, k1, r11);
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
}

template <typename I, typename K, typename C>
void conv1_valid(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    conv1_valid<default_vec>(input, kernel, conv, first, last);
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
