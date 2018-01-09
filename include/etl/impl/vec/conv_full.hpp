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

    conv1_valid_impl<default_vec>(input, kernel, memory_slice(conv, left, size(conv)), first, last);
}

// TODO This need to be make much faster

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename K, typename C>
void conv2_full_flipped_impl(const I& input, const K& kernel, C&& conv, value_t<I> beta) {
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

    auto* in = input.memory_start();
    auto* kkk = kernel.memory_start();
    auto* out = conv.memory_start();

    if (beta == T(0)) {
        for (size_t i = 0; i < c1; ++i) {
            const auto k_lo = std::max<int>(0, i - m1 + 1);
            const auto k_hi = std::min(n1 - 1, i) + 1;

            const int k1 = int(m1) - 1 - int(i);

            for (size_t j = 0; j < c2; ++j) {
                const auto l_lo = std::max<int>(0, j - m2 + 1);
                const auto l_hi = std::min(n2 - 1, j) + 1;

                const int k2 = int(m2) - 1 - int(j);

                auto r1    = vec_type::template zero<T>();
                auto temp1 = T(0);

                for (size_t k = k_lo; k < k_hi; ++k) {
                    size_t l = l_lo;

                    for (; l + vec_size - 1 < l_hi; l += vec_size) {
                        auto i1  = vec_type::loadu(in + k * n2 + l);
                        auto sk1 = vec_type::loadu(kkk + (k1 + k) * m2 + (k2 + l) + vec_size * 0);
                        r1       = vec_type::fmadd(sk1, i1, r1);
                    }

                    for (; l < l_hi; ++l) {
                        temp1 += in[k * n2 + l] * kkk[(k1 + k) * m2 + k2 + l];
                    }
                }

                out[i * c2 +j] = vec_type::hadd(r1) + temp1;
            }
        }
    } else {
        for (size_t i = 0; i < c1; ++i) {
            const auto k_lo = std::max<int>(0, i - m1 + 1);
            const auto k_hi = std::min(n1 - 1, i) + 1;

            const int k1 = int(m1) - 1 - int(i);

            for (size_t j = 0; j < c2; ++j) {
                const auto l_lo = std::max<int>(0, j - m2 + 1);
                const auto l_hi = std::min(n2 - 1, j) + 1;

                const int k2 = int(m2) - 1 - int(j);

                auto r1    = vec_type::template zero<T>();
                auto temp1 = T(0);

                for (size_t k = k_lo; k < k_hi; ++k) {
                    size_t l = l_lo;

                    for (; l + vec_size - 1 < l_hi; l += vec_size) {
                        auto i1  = vec_type::loadu(in + k * n2 + l);
                        auto sk1 = vec_type::loadu(kkk + (k1 + k) * m2 + (k2 + l) + vec_size * 0);
                        r1       = vec_type::fmadd(sk1, i1, r1);
                    }

                    for (; l < l_hi; ++l) {
                        temp1 += in[k * n2 + l] * kkk[(k1 + k) * m2 + k2 + l];
                    }
                }

                out[i * c2 +j] = beta * out[i * c2 +j] + vec_type::hadd(r1) + temp1;
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
void conv2_full_impl(const I& input, const K& kernel, C&& conv, value_t<I> beta) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t k1 = etl::dim<0>(kernel);
    const size_t k2 = etl::dim<1>(kernel);

    etl::dyn_matrix<T, 2> kernel_reverse(k1, k2);

    std::reverse_copy(kernel.memory_start(), kernel.memory_start() + k1 * k2, kernel_reverse.memory_start());

    conv2_full_flipped_impl<V>(input, kernel_reverse, conv, beta);
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    using T = value_t<I>;
    conv2_full_impl<default_vec>(input, kernel, conv, T(0.0));
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_disable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv2_full");
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_full_flipped(const I& input, const K& kernel, C&& conv) {
    using T = value_t<I>;
    conv2_full_flipped_impl<default_vec>(input, kernel, conv, T(0));
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_disable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_full_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv2_full_flipped");
}

/*!
 * \brief SSE implementation of a 2D 'full' convolution C = I * K, with multiple
 * kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_full_multi(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t KK = etl::dim<0>(kernel);

    auto batch_fun_k = [&input, &kernel, &conv](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_full_impl<default_vec>(input, kernel(k), conv(k), T(0));
        }
    };

    engine_dispatch_1d(batch_fun_k, 0, KK, 2UL);
}

/*!
 * \brief AVX implementation of a 2D 'full' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_full_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t KK = etl::dim<0>(kernel);

    auto batch_fun_k = [&input, &kernel, &conv](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_full_flipped_impl<default_vec>(input, kernel(k), conv(k), T(0));
        }
    };

    engine_dispatch_1d(batch_fun_k, 0, KK, 2UL);
}

/*!
 * \brief AVX implementation of a 2D 'full' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_disable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_full_multi(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv2_full_multi");
}

/*!
 * \brief AVX implementation of a 2D 'full' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_disable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv2_full_multi_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv2_full_multi_flipped");
}

/*!
 * \brief SSE implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename V, typename I, typename KK, typename CC>
void conv4_full_impl(const I& input, const KK& kernel, CC&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t N = etl::dim<0>(input);
    const size_t K = etl::dim<0>(kernel);
    const size_t C = etl::dim<1>(kernel);

    const size_t k1 = etl::dim<2>(kernel);
    const size_t k2 = etl::dim<3>(kernel);

    if (K > 0) {
        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        etl::dyn_matrix<T, 4> prepared_k(K, C, k1, k2);

        std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

        prepared_k.deep_fflip_inplace();

        auto batch_fun_nc = [&](const size_t first, const size_t last) {
            for (size_t nc = first; nc < last; ++nc) {
                const size_t i = nc / C;
                const size_t c = nc % C;

                // k = 0
                conv2_full_flipped_impl<V>(input(i)(0), prepared_k(0)(c), conv(i)(c), T(0));

                for (size_t k = 1; k < K; ++k) {
                    conv2_full_flipped_impl<V>(input(i)(k), prepared_k(k)(c), conv(i)(c), T(1));
                }
            }
        };

        engine_dispatch_1d_serial(batch_fun_nc, 0, N * C, 2UL);

        conv.invalidate_gpu();
    }
}

/*!
 * \brief Optimized implementation of a 4D 'full' convolution C = I * K for small kernels.
 *
 * This returns true if it was able to perform the optimized
 * convolution false otherwise
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
bool conv4_full_flipped_small(const I& input, const KK& kernel, CC&& conv) {
    using T = value_t<I>;

    const size_t N = etl::dim<0>(input);
    const size_t K = etl::dim<0>(kernel);
    const size_t C = etl::dim<1>(kernel);

    const size_t k1 = etl::dim<2>(kernel);
    const size_t k2 = etl::dim<3>(kernel);

    // Handle small square kernels
    if (k1 != k2 || !(k2 == 3 || k2 == 5)) {
        return false;
    }

    const size_t pad = k2 - 1;
    const size_t full_pad = pad * 2;

    if /*constexpr*/ (padding_impl) {
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if (k2 < SS || k2 % AS > 0) {
            const size_t valid_pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            // Prepare the double padding of the input (padding for full->valid and padding for valid impl)

            etl::dyn_matrix<T, 4> p_input(etl::dim<0>(input), etl::dim<1>(input), full_pad + etl::dim<2>(input), full_pad + etl::dim<3>(input) + valid_pad);

            p_input = 0;

            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                for (size_t j = 0; j < etl::dim<1>(input); ++j) {
                    for (size_t k = 0; k < etl::dim<2>(input); ++k) {
                        direct_copy_n(input(i)(j)(k).memory_start(), p_input(i)(j)(pad + k).memory_start() + pad, etl::dim<3>(input));
                    }
                }
            }

            // Pad the kernel for fast implementation
            auto p_kernel = common::pad_right_multi(kernel, valid_pad);

            if (detail::prefer_sse<T>(k2 + valid_pad)) {
                auto batch_fun_nc = [&](const size_t first, const size_t last) {
                    if (last - first) {
                        for (size_t nc = first; nc < last; ++nc) {
                            const size_t i = nc / C;
                            const size_t c = nc % C;

                            // k = 0
                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(p_input(i)(0), p_kernel(0)(c), conv(i)(c), 1, 1, 0, 0, T(0));

                            for (size_t k = 1; k < K; ++k) {
                                detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(p_input(i)(k), p_kernel(k)(c), conv(i)(c), 1, 1, 0, 0, T(1));
                            }
                        }
                    }
                };

                engine_dispatch_1d(batch_fun_nc, 0, N * C, 2UL);
            } else {
                auto batch_fun_nc = [&](const size_t first, const size_t last) {
                    if (last - first) {
                        for (size_t nc = first; nc < last; ++nc) {
                            const size_t i = nc / C;
                            const size_t c = nc % C;

                            // k = 0
                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(p_input(i)(0), p_kernel(0)(c), conv(i)(c), 1, 1, 0, 0, T(0));

                            for (size_t k = 1; k < K; ++k) {
                                detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(p_input(i)(k), p_kernel(k)(c), conv(i)(c), 1, 1, 0, 0, T(1));
                            }
                        }
                    }
                };

                engine_dispatch_1d(batch_fun_nc, 0, N * C, 2UL);
            }

            return true;
        }
    }

    // No padding implementation => Defer directly to conv2_valid_flipped

    // Pad the input for valid convolution

    etl::dyn_matrix<T, 4> p_input(etl::dim<0>(input), etl::dim<1>(input), full_pad + etl::dim<2>(input), full_pad + etl::dim<3>(input));

    p_input = 0;

    for (size_t i = 0; i < etl::dim<0>(input); ++i) {
        for (size_t j = 0; j < etl::dim<1>(input); ++j) {
            for (size_t k = 0; k < etl::dim<2>(input); ++k) {
                direct_copy_n(input(i)(j)(k).memory_start(), p_input(i)(j)(pad + k).memory_start() + pad, etl::dim<3>(input));
            }
        }
    }

    if (detail::prefer_sse<T>(k2)) {
        auto batch_fun_nc = [&](const size_t first, const size_t last) {
            if (last - first) {
                for (size_t nc = first; nc < last; ++nc) {
                    const size_t i = nc / C;
                    const size_t c = nc % C;

                    // k = 0
                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(p_input(i)(0), kernel(0)(c), conv(i)(c), 1, 1, 0, 0, T(0));

                    for (size_t k = 1; k < K; ++k) {
                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(p_input(i)(k), kernel(k)(c), conv(i)(c), 1, 1, 0, 0, T(1));
                    }
                }
            }
        };

        engine_dispatch_1d(batch_fun_nc, 0, N * C, 2UL);
    } else {
        auto batch_fun_nc = [&](const size_t first, const size_t last) {
            if (last - first) {
                for (size_t nc = first; nc < last; ++nc) {
                    const size_t i = nc / C;
                    const size_t c = nc % C;

                    // k = 0
                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(p_input(i)(0), kernel(0)(c), conv(i)(c), 1, 1, 0, 0, T(0));

                    for (size_t k = 1; k < K; ++k) {
                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(p_input(i)(k), kernel(k)(c), conv(i)(c), 1, 1, 0, 0, T(1));
                    }
                }
            }
        };

        engine_dispatch_1d(batch_fun_nc, 0, N * C, 2UL);
    }

    return true;
}

/*!
 * \brief Optimized implementation of a 4D 'full' convolution C = I * K with padding.
 *
 * This returns true if the method was able to perform the
 * convolution, false otherwise.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
bool conv4_full_flipped_padding(const I& input, const KK& kernel, CC&& conv) {
    using T = value_t<I>;

    const size_t N = etl::dim<0>(input);
    const size_t K = etl::dim<0>(kernel);
    const size_t C = etl::dim<1>(kernel);

    const size_t k2 = etl::dim<3>(kernel);

    // Disabled for now because slower in fact than non-padded
    if /*constexpr*/ (padding_impl && false) {
        constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
        constexpr size_t SS = AS / 2;

        if (k2 < SS || k2 % AS > 0) {
            const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

            auto p_kernel = common::pad_right_multi(kernel, pad);

            if (detail::prefer_sse<T>(k2 + pad)) {
                auto batch_fun_nc = [&](const size_t first, const size_t last) {
                    if (last - first) {
                        etl::dyn_matrix<T, 2> padded_conv(etl::dim<2>(conv), pad + etl::dim<3>(conv));

                        for (size_t nc = first; nc < last; ++nc) {
                            const size_t i = nc / C;
                            const size_t c = nc % C;

                            // k = 0
                            conv2_full_flipped_impl<detail::safe_sse_vec>(input(i)(0), p_kernel(0)(c), padded_conv, T(0));

                            for (size_t k = 1; k < K; ++k) {
                                conv2_full_flipped_impl<detail::safe_sse_vec>(input(i)(k), p_kernel(k)(c), padded_conv, T(1));
                            }

                            // Copy back the results

                            for (size_t k = 0; k < etl::dim<2>(conv); ++k) {
                                for (size_t l = 0; l < etl::dim<3>(conv); ++l) {
                                    conv(i, c, k, l) = padded_conv(k, pad + l);
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d(batch_fun_nc, 0, N * C, 2UL);
            } else {
                auto batch_fun_nc = [&](const size_t first, const size_t last) {
                    if (last - first) {
                        etl::dyn_matrix<T, 2> padded_conv(etl::dim<2>(conv), pad + etl::dim<2>(conv));

                        for (size_t nc = first; nc < last; ++nc) {
                            const size_t i = nc / C;
                            const size_t c = nc % C;

                            // k = 0
                            conv2_full_flipped_impl<detail::safe_avx_vec>(input(i)(0), p_kernel(0)(c), padded_conv, T(0));

                            for (size_t k = 1; k < K; ++k) {
                                conv2_full_flipped_impl<detail::safe_avx_vec>(input(i)(k), p_kernel(k)(c), padded_conv, T(1));
                            }

                            // Copy back the results

                            for (size_t k = 0; k < etl::dim<2>(conv); ++k) {
                                for (size_t l = 0; l < etl::dim<3>(conv); ++l) {
                                    conv(i, c, k, l) = padded_conv(k, pad + l);
                                }
                            }
                        }
                    }
                };

                engine_dispatch_1d(batch_fun_nc, 0, N * C, 2UL);
            }

            return true;
        }
    }

    return false;
}

/*!
 * \brief SSE implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C, cpp_enable_iff(conv2_possible<vector_mode, I, K, C>)>
void conv4_full(const I& input, const K& kernel, C&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    if /*constexpr*/ (avx_enabled && sse3_enabled) {
        const size_t k2 = etl::dim<3>(kernel);

        if (detail::prefer_sse<value_t<I>>(k2)) {
            return conv4_full_impl<detail::safe_avx_vec>(input, kernel, conv);
        } else {
            return conv4_full_impl<detail::safe_sse_vec>(input, kernel, conv);
        }
    } else {
        return conv4_full_impl<default_vec>(input, kernel, conv);
    }
}

/*!
 * \brief Vectorized implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC, cpp_enable_iff(conv2_possible<vector_mode, I, KK, CC>)>
void conv4_full_flipped(const I& input, const KK& kernel, CC&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t N = etl::dim<0>(input);
    const size_t K = etl::dim<0>(kernel);
    const size_t C = etl::dim<1>(kernel);

    if (K > 0) {
        const size_t k2 = etl::dim<3>(kernel);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        // 1. Try optimized algorithms for small kernels

        if (conv4_full_flipped_small(input, kernel, conv)) {
            conv.invalidate_gpu();
            return;
        }

        // 2. Try padding implementation

        if (conv4_full_flipped_padding(input, kernel, conv)) {
            conv.invalidate_gpu();
            return;
        }

        // 3. General algorithms

        if (detail::prefer_sse<T>(k2)) {
            auto batch_fun_nc = [&](const size_t first, const size_t last) {
                if (last - first) {
                    for (size_t nc = first; nc < last; ++nc) {
                        const size_t i = nc / C;
                        const size_t c = nc % C;

                        // k = 0
                        conv2_full_flipped_impl<detail::safe_sse_vec>(input(i)(0), kernel(0)(c), conv(i)(c), T(0));

                        for (size_t k = 1; k < K; ++k) {
                            conv2_full_flipped_impl<detail::safe_sse_vec>(input(i)(k), kernel(k)(c), conv(i)(c), T(1));
                        }
                    }
                }
            };

            engine_dispatch_1d(batch_fun_nc, 0, N * C, 2UL);
        } else {
            auto batch_fun_nc = [&](const size_t first, const size_t last) {
                if (last - first) {
                    for (size_t nc = first; nc < last; ++nc) {
                        const size_t i = nc / C;
                        const size_t c = nc % C;

                        // k = 0
                        conv2_full_flipped_impl<detail::safe_avx_vec>(input(i)(0), kernel(0)(c), conv(i)(c), T(0));

                        for (size_t k = 1; k < K; ++k) {
                            conv2_full_flipped_impl<detail::safe_avx_vec>(input(i)(k), kernel(k)(c), conv(i)(c), T(1));
                        }
                    }
                }
            };

            engine_dispatch_1d(batch_fun_nc, 0, N * C, 2UL);
        }

        conv.invalidate_gpu();
    }
}

/*!
 * \brief Vectorized implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC, cpp_disable_iff(conv2_possible<vector_mode, I, KK, CC>)>
void conv4_full(const I& input, const KK& kernel, CC&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv4_full");
}

/*!
 * \brief Vectorized implementation of a 4D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC, cpp_disable_iff(conv2_possible<vector_mode, I, KK, CC>)>
void conv4_full_flipped(const I& input, const KK& kernel, CC&& conv) {
    cpp_unused(input);
    cpp_unused(kernel);
    cpp_unused(conv);

    cpp_unreachable("Invalid call to vec::conv4_full_flipped");
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
