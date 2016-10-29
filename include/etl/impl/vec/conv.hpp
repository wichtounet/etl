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

                for(; padding_impl && k < k2; ++k){
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

                for(; padding_impl && k < k2; ++k){
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

                for(; padding_impl && k < k2; ++k){
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

                for(; padding_impl && k < k2; ++k){
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

                for(; padding_impl && k < k2; ++k){
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

                for(; padding_impl && k < k2; ++k){
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
          std::is_same<T, float>::value
        ? (n % 4 < n % 8)
        : (n % 2 < n % 4);
}

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

    if(padding_impl && kernel.dim(1) % 4 > 0){
        const auto pad = 4 - kernel.dim(1) % 4;
        etl::dyn_matrix<T, 2> padded_kernel(kernel.dim(0), kernel.dim(1) + pad);
        etl::dyn_matrix<T, 2> padded_input(input.dim(0), input.dim(1) + pad);

        padded_kernel = 0;
        padded_input = 0;

        for(size_t i = 0; i < kernel.dim(0); ++i){
            direct_copy_n(kernel.memory_start() + i * kernel.dim(1), padded_kernel.memory_start() + i * padded_kernel.dim(1), kernel.dim(1));
        }

        for(size_t i = 0; i < input.dim(0); ++i){
            direct_copy_n(input.memory_start() + i * input.dim(1), padded_input.memory_start() + i * padded_input.dim(1), input.dim(1));
        }

        if(detail::prefer_sse<T>(k2)){
            detail::conv2_valid_flipped<sse_vec>(padded_input, padded_kernel, conv);
        } else {
            detail::conv2_valid_flipped<avx_vec>(padded_input, padded_kernel, conv);
        }

        return;
    }

    if(detail::prefer_sse<T>(k2)){
        detail::conv2_valid_flipped<sse_vec>(input, kernel, conv);
    } else {
        detail::conv2_valid_flipped<avx_vec>(input, kernel, conv);
    }
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
