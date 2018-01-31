//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::vec {

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

    size_t left = (etl::size(kernel) - 1) / 2;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    auto* out      = conv.memory_start();
    const auto* in = input.memory_start();
    const auto* k  = kernel.memory_start();

    //Process not-'valid' parts of the convolution (left and right)
    etl::impl::common::left_same_kernel(in, etl::size(input), k, etl::size(kernel), out, first, last);
    etl::impl::common::right_same_kernel(in, etl::size(input), k, etl::size(kernel), out, first, last);

    conv1_valid_impl<default_vec>(input, kernel, memory_slice(conv, left, etl::size(conv)), first, last);
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv2_same_flipped_impl(const I& input, const K& kernel, C&& conv) {
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
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv2_same_impl(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    const size_t k1 = etl::dim<0>(kernel);
    const size_t k2 = etl::dim<1>(kernel);

    etl::dyn_matrix<T, 2> kernel_reverse(k1, k2);

    std::reverse_copy(kernel.memory_start(), kernel.memory_start() + k1 * k2, kernel_reverse.memory_start());

    conv2_same_flipped_impl<V>(input, kernel_reverse, conv);

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
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_same_flipped(const I& input, const K& kernel, C&& conv) {
    conv2_same_flipped_impl<default_vec>(input, kernel, conv);
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K, with the
 * flipped kernels of K.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_disable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_same_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv2_same_flipped");
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    conv2_same_impl<default_vec>(input, kernel, conv);
}

/*!
 * \brief SSE implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_disable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv2_same");
}

/*!
 * \brief VEC implementation of a 2D 'same' convolution C = I * K, with multiple kernels
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_same_multi(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    const size_t Kn = etl::dim<0>(kernel);

    auto batch_fun_k = [&input, &kernel, &conv](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_same_impl<default_vec>(input, kernel(k), conv(k));
        }
    };

    engine_dispatch_1d(batch_fun_k, 0, Kn, 2UL);
}

/*!
 * \brief VEC implementation of a 2D 'same' convolution C = I * K, with multiple kernels, with kernels flipped.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_same_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    const size_t Kn = etl::dim<0>(kernel);

    auto batch_fun_k = [&input, &kernel, &conv](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_same_flipped_impl<default_vec>(input, kernel(k), conv(k));
        }
    };

    engine_dispatch_1d(batch_fun_k, 0, Kn, 2UL);
}

/*!
 * \brief VEC implementation of a 2D 'same' convolution C = I * K, with multiple kernels, with kernels flipped.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_disable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_same_multi(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv2_same_multi");
}

/*!
 * \brief VEC implementation of a 2D 'same' convolution C = I * K, with multiple kernels, with kernels flipped.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_disable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_same_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv2_same_multi_flipped");
}

} //end of namespace etl::impl::vec
