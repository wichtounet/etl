//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

    conv1_valid<default_vec>(input, kernel, memory_slice(conv, left, size(conv)), first, last);
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

    if (K > 0) {
        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        etl::dyn_matrix<T, 4> prepared_k(K, C, k1, k2);

        std::copy(kernel.memory_start(), kernel.memory_end(), prepared_k.memory_start());

        prepared_k.deep_fflip_inplace();

        auto batch_fun_nc = [&](const size_t first, const size_t last) {
            if (last - first) {
                SERIAL_SECTION {
                    for (size_t nc = first; nc < last; ++nc) {
                        const size_t i = nc / C;
                        const size_t c = nc % C;

                        // k = 0
                        conv2_full_flipped<V>(input(i)(0), prepared_k(0)(c), conv(i)(c), T(0));

                        for (size_t k = 1; k < K; ++k) {
                            conv2_full_flipped<V>(input(i)(k), prepared_k(k)(c), conv(i)(c), T(1));
                        }
                    }
                }
            }
        };

        if(etl::is_parallel){
            dispatch_1d_any(select_parallel(N * C, 2), batch_fun_nc, 0, N * C);
        } else {
            batch_fun_nc(0, N * C);
        }

        conv.invalidate_gpu();
    }
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
template <typename I, typename KK, typename CC>
void conv4_full_flipped(const I& input, const KK& kernel, CC&& conv) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I>;

    const size_t N = etl::dim<0>(input);
    const size_t K = etl::dim<0>(kernel);
    const size_t C = etl::dim<1>(kernel);

    if (K > 0) {
        const size_t k1 = etl::dim<2>(kernel);
        const size_t k2 = etl::dim<3>(kernel);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();


        // Disabled for now because slower in fact than non-padded
        if (padding_impl && false) {
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
                                conv2_full_flipped<detail::safe_sse_vec>(input(i)(0), p_kernel(0)(c), padded_conv, T(0));

                                for (size_t k = 1; k < K; ++k) {
                                    conv2_full_flipped<detail::safe_sse_vec>(input(i)(k), p_kernel(k)(c), padded_conv, T(1));
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

                    if (etl::is_parallel) {
                        dispatch_1d_any(select_parallel(N * C, 2), batch_fun_nc, 0, N * C);
                    } else {
                        batch_fun_nc(0, N * C);
                    }

                } else {
                    auto batch_fun_nc = [&](const size_t first, const size_t last) {
                        if (last - first) {
                            etl::dyn_matrix<T, 2> padded_conv(etl::dim<2>(conv), pad + etl::dim<2>(conv));

                            for (size_t nc = first; nc < last; ++nc) {
                                const size_t i = nc / C;
                                const size_t c = nc % C;

                                // k = 0
                                conv2_full_flipped<detail::safe_avx_vec>(input(i)(0), p_kernel(0)(c), padded_conv, T(0));

                                for (size_t k = 1; k < K; ++k) {
                                    conv2_full_flipped<detail::safe_avx_vec>(input(i)(k), p_kernel(k)(c), padded_conv, T(1));
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

                    if (etl::is_parallel) {
                        dispatch_1d_any(select_parallel(N * C, 2), batch_fun_nc, 0, N * C);
                    } else {
                        batch_fun_nc(0, N * C);
                    }
                }

                conv.invalidate_gpu();

                return;
            }
        }

        if (detail::prefer_sse<T>(k2)) {
            auto batch_fun_nc = [&](const size_t first, const size_t last) {
                if (last - first) {
                    for (size_t nc = first; nc < last; ++nc) {
                        const size_t i = nc / C;
                        const size_t c = nc % C;

                        // k = 0
                        conv2_full_flipped<detail::safe_sse_vec>(input(i)(0), kernel(0)(c), conv(i)(c), T(0));

                        for (size_t k = 1; k < K; ++k) {
                            conv2_full_flipped<detail::safe_sse_vec>(input(i)(k), kernel(k)(c), conv(i)(c), T(1));
                        }
                    }
                }
            };

            if (etl::is_parallel) {
                dispatch_1d_any(select_parallel(N * C, 2), batch_fun_nc, 0, N * C);
            } else {
                batch_fun_nc(0, N * C);
            }
        } else {
            auto batch_fun_nc = [&](const size_t first, const size_t last) {
                if (last - first) {
                    for (size_t nc = first; nc < last; ++nc) {
                        const size_t i = nc / C;
                        const size_t c = nc % C;

                        // k = 0
                        conv2_full_flipped<detail::safe_avx_vec>(input(i)(0), kernel(0)(c), conv(i)(c), T(0));

                        for (size_t k = 1; k < K; ++k) {
                            conv2_full_flipped<detail::safe_avx_vec>(input(i)(k), kernel(k)(c), conv(i)(c), T(1));
                        }
                    }
                }
            };

            if (etl::is_parallel) {
                dispatch_1d_any(select_parallel(N * C, 2), batch_fun_nc, 0, N * C);
            } else {
                batch_fun_nc(0, N * C);
            }
        }

        conv.invalidate_gpu();
    }
}

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl

#include "etl/impl/vec/conv_same.hpp"
