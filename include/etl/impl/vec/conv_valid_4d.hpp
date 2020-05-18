//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/common/conv.hpp"
#include "etl/impl/vec/conv_valid_kernels.hpp"

namespace etl::impl::vec {

/*!
 * \brief Decide if padding is used for the given kernel dimensions and the
 * input padding.
 *
 * \param k1 The first dimension of the kernel.
 * \param k2 The second dimension of the kernel.
 *
 * \return true if padding is to be used, false otherwise.
 */
template <typename T>
constexpr bool need_padding([[maybe_unused]] size_t k1, size_t k2, size_t p1, size_t p2) {
    constexpr bool single = std::is_same_v<T, float>;
    constexpr size_t AS   = single ? 8 : 4;
    constexpr size_t SS   = AS / 2;

    return k2 < SS || k2 % AS > 0 || p1 || p2;
}

/*!
 * \brief Select the amount of padding for the second dimension of the kernel
 * based on the dimensions of the kernel.
 *
 * \param k1 The first dimension of the kernel.
 * \param k2 The second dimension of the kernel.
 *
 * \return The amount of padding for the second dimension of the kernel and input.
 */
template <typename T>
constexpr size_t select_pad(size_t k1, size_t k2) {
    constexpr bool single = std::is_same_v<T, float>;
    constexpr size_t AS   = single ? 8 : 4;
    constexpr size_t SS   = AS / 2;

    if (k2 < SS || k2 % AS > 0) {
        // Very special version for 3x3 with AVX
        if (single && avx_enabled && k1 == 3 && k2 == 3) {
            return AS - 3;
        }

        return k2 < SS ? SS - k2 % SS : AS - k2 % AS;
    } else {
        return 0;
    }
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_valid([[maybe_unused]] const I& input,
                 [[maybe_unused]] const KK& kernel,
                 [[maybe_unused]] CC&& conv,
                 [[maybe_unused]] size_t s1,
                 [[maybe_unused]] size_t s2,
                 [[maybe_unused]] size_t p1,
                 [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, CC>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        if (etl::dim<1>(kernel) > 0) {
            const size_t N = etl::dim<0>(input);  // The number of images
            const size_t K = etl::dim<0>(kernel); // The number of kernels
            const size_t C = etl::dim<1>(input);  // The number of channels

            const size_t k1 = etl::dim<3>(kernel);
            const size_t k2 = etl::dim<3>(kernel);

            input.ensure_cpu_up_to_date();
            kernel.ensure_cpu_up_to_date();

            if constexpr (padding_impl) {
                if (need_padding<T>(k1, k2, p1, p2)) {
                    const size_t pad = select_pad<T>(k1, k2);

                    if (cpp_likely(p1 == 0 && p2 == 0)) {
                        cpp_assert(pad, "Invalid configuration, need_padding shoud not return true");

                        auto padded_input  = common::pad_right_multi(input, pad);
                        auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2,
                                                                                                   p1, p2, T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        } else {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2,
                                                                                                   p1, p2, T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        }
                    } else if (pad) {
                        auto padded_input  = common::pad_right_multi_double(input, pad, p1, p2);
                        auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2,
                                                                                                   0, 0, T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        } else {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2,
                                                                                                   0, 0, T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        }
                    } else {
                        cpp_assert(!pad, "Invalid padding configuration");

                        auto padded_input = common::pad_right_multi_double(input, 0, p1, p2);

                        if (detail::prefer_sse<T>(k2)) {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, 0, 0,
                                                                                                   T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        } else {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, 0, 0,
                                                                                                   T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        }
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

                engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
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

                engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
            }

            conv.invalidate_gpu();
        }
    } else {
        cpp_unreachable("Invalid call to vec::conv2_full");
    }
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_valid_flipped([[maybe_unused]] const I& input,
                         [[maybe_unused]] const KK& kernel,
                         [[maybe_unused]] CC&& conv,
                         [[maybe_unused]] size_t s1,
                         [[maybe_unused]] size_t s2,
                         [[maybe_unused]] size_t p1,
                         [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, CC>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        if (etl::dim<0>(kernel) > 0) {
            const size_t N = etl::dim<0>(input);  // The number of images
            const size_t K = etl::dim<0>(kernel); // The number of kernels
            const size_t C = etl::dim<1>(input);  // The number of channels

            const size_t k1 = etl::dim<2>(kernel);
            const size_t k2 = etl::dim<3>(kernel);

            input.ensure_cpu_up_to_date();
            kernel.ensure_cpu_up_to_date();

            // TODO Performance can be improved further by doing the
            // padding of the kernel inside the thread for small kernel (3x3, 5x5)

            if constexpr (padding_impl) {
                if (need_padding<T>(k1, k2, p1, p2)) {
                    const size_t pad = select_pad<T>(k1, k2);

                    if (cpp_likely(p1 == 0 && p2 == 0)) {
                        cpp_assert(pad, "Invalid configuration, need_padding shoud not return true");

                        auto padded_input  = common::pad_right_multi(input, pad);
                        auto padded_kernel = common::pad_right_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2,
                                                                                                   p1, p2, T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        } else {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2,
                                                                                                   p1, p2, T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        }
                    } else if (pad) {
                        auto padded_input  = common::pad_right_multi_double(input, pad, p1, p2);
                        auto padded_kernel = common::pad_right_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2,
                                                                                                   0, 0, T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        } else {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(k)(0), conv(i)(k), s1, s2,
                                                                                                   0, 0, T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(k)(c), conv(i)(k), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        }
                    } else {
                        cpp_assert(!pad, "Invalid padding configuration");

                        auto padded_input = common::pad_right_multi_double(input, 0, p1, p2);

                        if (detail::prefer_sse<T>(k2)) {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, 0, 0,
                                                                                                   T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        } else {
                            auto fun_nk = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / K;
                                    const size_t k = nk % K;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, 0, 0,
                                                                                                   T(0));

                                    for (size_t c = 1; c < C; ++c) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
                        }
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

                engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
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

                engine_dispatch_1d(fun_nk, 0, N * K, 4UL);
            }

            conv.invalidate_gpu();
        }
    } else {
        cpp_unreachable("Invalid call to vec::conv2_full");
    }
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_valid_back([[maybe_unused]] const I& input,
                      [[maybe_unused]] const KK& kernel,
                      [[maybe_unused]] CC&& conv,
                      [[maybe_unused]] size_t s1,
                      [[maybe_unused]] size_t s2,
                      [[maybe_unused]] size_t p1,
                      [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, CC>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        if (etl::dim<1>(kernel) > 0) {
            const auto N = etl::dim<0>(input);
            const auto C = etl::dim<1>(kernel);
            const auto K = etl::dim<1>(input);

            const size_t k1 = etl::dim<2>(kernel);
            const size_t k2 = etl::dim<3>(kernel);

            input.ensure_cpu_up_to_date();
            kernel.ensure_cpu_up_to_date();

            conv = 0;

            if constexpr (padding_impl) {
                if (need_padding<T>(k1, k2, p1, p2)) {
                    const size_t pad = select_pad<T>(k1, k2);

                    if (cpp_likely(p1 == 0 && p2 == 0)) {
                        cpp_assert(pad, "Invalid configuration, need_padding shoud not return true");

                        auto padded_input  = common::pad_right_multi(input, pad);
                        auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(0)(c), conv(i)(c), s1, s2,
                                                                                                   p1, p2, T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(k), padded_kernel(k)(c), conv(i)(c), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        } else {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(0)(c), conv(i)(c), s1, s2,
                                                                                                   p1, p2, T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(k), padded_kernel(k)(c), conv(i)(c), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        }
                    } else if (pad) {
                        auto padded_input  = common::pad_right_multi_double(input, pad, p1, p2);
                        auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(0)(c), conv(i)(c), s1, s2,
                                                                                                   0, 0, T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(k), padded_kernel(k)(c), conv(i)(c), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        } else {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(0)(c), conv(i)(c), s1, s2,
                                                                                                   0, 0, T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(k), padded_kernel(k)(c), conv(i)(c), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        }
                    } else {
                        cpp_assert(!pad, "Invalid padding configuration");

                        auto padded_input = common::pad_right_multi_double(input, 0, p1, p2);

                        if (detail::prefer_sse<T>(k2)) {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, 0, 0,
                                                                                                   T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        } else {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, 0, 0,
                                                                                                   T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        }
                    }

                    return;
                }
            }

            if (detail::prefer_sse<T>(k2)) {
                auto fun_nc = [&](const size_t first, const size_t last) {
                    for (size_t nk = first; nk < last; ++nk) {
                        const size_t i = nk / C;
                        const size_t c = nk % C;

                        detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, p1, p2, T(0));

                        for (size_t k = 1; k < K; ++k) {
                            detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, p1, p2, T(1));
                        }
                    }
                };

                engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
            } else {
                auto fun_nc = [&](const size_t first, const size_t last) {
                    for (size_t nk = first; nk < last; ++nk) {
                        const size_t i = nk / C;
                        const size_t c = nk % C;

                        detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, p1, p2, T(0));

                        for (size_t k = 1; k < K; ++k) {
                            detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, p1, p2, T(1));
                        }
                    }
                };

                engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
            }

            conv.invalidate_gpu();
        }
    } else {
        cpp_unreachable("Invalid call to vec::conv2_full");
    }
}

/*!
 * \brief SSE implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename KK, typename CC>
void conv4_valid_back_flipped([[maybe_unused]] const I& input,
                              [[maybe_unused]] const KK& kernel,
                              [[maybe_unused]] CC&& conv,
                              [[maybe_unused]] size_t s1,
                              [[maybe_unused]] size_t s2,
                              [[maybe_unused]] size_t p1,
                              [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, CC>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        if (etl::dim<0>(kernel) > 0) {
            const auto N = etl::dim<0>(input);
            const auto C = etl::dim<1>(kernel);
            const auto K = etl::dim<1>(input);

            const size_t k1 = etl::dim<2>(kernel);
            const size_t k2 = etl::dim<3>(kernel);

            input.ensure_cpu_up_to_date();
            kernel.ensure_cpu_up_to_date();

            // TODO Performance can be improved further by doing the
            // padding of the kernel inside the thread for small kernel (3x3, 5x5)

            if constexpr (padding_impl) {
                if (need_padding<T>(k1, k2, p1, p2)) {
                    const size_t pad = select_pad<T>(k1, k2);

                    if (cpp_likely(p1 == 0 && p2 == 0)) {
                        cpp_assert(pad, "Invalid configuration, need_padding shoud not return true");

                        auto padded_input  = common::pad_right_multi(input, pad);
                        auto padded_kernel = common::pad_right_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(0)(c), conv(i)(c), s1, s2,
                                                                                                   p1, p2, T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(k), padded_kernel(k)(c), conv(i)(c), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        } else {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(0)(c), conv(i)(c), s1, s2,
                                                                                                   p1, p2, T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(k), padded_kernel(k)(c), conv(i)(c), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        }
                    } else if (pad) {
                        auto padded_input  = common::pad_right_multi_double(input, pad, p1, p2);
                        auto padded_kernel = common::pad_right_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), padded_kernel(0)(c), conv(i)(c), s1, s2,
                                                                                                   0, 0, T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(k), padded_kernel(k)(c), conv(i)(c), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        } else {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), padded_kernel(0)(c), conv(i)(c), s1, s2,
                                                                                                   0, 0, T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(k), padded_kernel(k)(c), conv(i)(c), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        }
                    } else {
                        cpp_assert(!pad, "Invalid padding configuration");

                        auto padded_input = common::pad_right_multi_double(input, 0, p1, p2);

                        if (detail::prefer_sse<T>(k2)) {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, 0, 0,
                                                                                                   T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        } else {
                            auto fun_nc = [&](const size_t first, const size_t last) {
                                for (size_t nk = first; nk < last; ++nk) {
                                    const size_t i = nk / C;
                                    const size_t c = nk % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, 0, 0,
                                                                                                   T(0));

                                    for (size_t k = 1; k < K; ++k) {
                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
                        }
                    }

                    return;
                }
            }

            if (detail::prefer_sse<T>(k2)) {
                auto fun_nc = [&](const size_t first, const size_t last) {
                    for (size_t nk = first; nk < last; ++nk) {
                        const size_t i = nk / C;
                        const size_t c = nk % C;

                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, p1, p2, T(0));

                        for (size_t k = 1; k < K; ++k) {
                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input(i)(c), kernel(k)(c), conv(i)(c), s1, s2, p1, p2, T(1));
                        }
                    }
                };

                engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
            } else {
                auto fun_nc = [&](const size_t first, const size_t last) {
                    for (size_t nk = first; nk < last; ++nk) {
                        const size_t i = nk / C;
                        const size_t c = nk % C;

                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, p1, p2, T(0));

                        for (size_t k = 1; k < K; ++k) {
                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, p1, p2, T(1));
                        }
                    }
                };

                engine_dispatch_1d(fun_nc, 0, N * C, 4UL);
            }

            conv.invalidate_gpu();
        }
    } else {
        cpp_unreachable("Invalid call to vec::conv2_full");
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
void conv4_valid_filter([[maybe_unused]] const I& input,
                        [[maybe_unused]] const KK& kernel,
                        [[maybe_unused]] CC&& conv,
                        [[maybe_unused]] size_t s1,
                        [[maybe_unused]] size_t s2,
                        [[maybe_unused]] size_t p1,
                        [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, CC>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        if (etl::dim<0>(input) > 0) {
            const size_t N = etl::dim<0>(input);  // The number of images
            const size_t C = etl::dim<1>(input);  // The number of channels
            const size_t K = etl::dim<1>(kernel); // The number of kernels

            const size_t k1 = etl::dim<2>(kernel);
            const size_t k2 = etl::dim<3>(kernel);

            input.ensure_cpu_up_to_date();
            kernel.ensure_cpu_up_to_date();

            if constexpr (padding_impl) {
                if (need_padding<T>(k1, k2, p1, p2)) {
                    const size_t pad = select_pad<T>(k1, k2);

                    if (cpp_likely(p1 == 0 && p2 == 0)) {
                        cpp_assert(pad, "Invalid configuration, need_padding shoud not return true");

                        auto padded_input  = common::pad_right_multi(input, pad);
                        auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2,
                                                                                                   p1, p2, T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        } else {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2,
                                                                                                   p1, p2, T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        }
                    } else if (pad) {
                        auto padded_input  = common::pad_right_multi_double(input, pad, p1, p2);
                        auto padded_kernel = common::pad_right_flip_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2,
                                                                                                   0, 0, T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        } else {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2,
                                                                                                   0, 0, T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        }
                    } else {
                        cpp_assert(!pad, "Invalid padding configuration");

                        auto padded_input = common::pad_right_multi_double(input, 0, p1, p2);

                        if (detail::prefer_sse<T>(k2)) {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, 0, 0,
                                                                                                   T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        } else {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, 0, 0,
                                                                                                   T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        }
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

                engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
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

                engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
            }

            conv.invalidate_gpu();
        }
    } else {
        cpp_unreachable("Invalid call to vec::conv2_full");
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
void conv4_valid_filter_flipped([[maybe_unused]] const I& input,
                                [[maybe_unused]] const KK& kernel,
                                [[maybe_unused]] CC&& conv,
                                [[maybe_unused]] size_t s1,
                                [[maybe_unused]] size_t s2,
                                [[maybe_unused]] size_t p1,
                                [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, CC>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        if (etl::dim<0>(input) > 0) {
            const size_t N = etl::dim<0>(input);  // The number of images
            const size_t C = etl::dim<1>(input);  // The number of channels
            const size_t K = etl::dim<1>(kernel); // The number of kernels

            const size_t k1 = etl::dim<2>(kernel);
            const size_t k2 = etl::dim<3>(kernel);

            input.ensure_cpu_up_to_date();
            kernel.ensure_cpu_up_to_date();

            if constexpr (padding_impl) {
                if (need_padding<T>(k1, k2, p1, p2)) {
                    const size_t pad = select_pad<T>(k1, k2);

                    if (cpp_likely(p1 == 0 && p2 == 0)) {
                        cpp_assert(pad, "Invalid configuration, need_padding shoud not return true");

                        auto padded_input  = common::pad_right_multi(input, pad);
                        auto padded_kernel = common::pad_right_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2,
                                                                                                   p1, p2, T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        } else {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2,
                                                                                                   p1, p2, T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1,
                                                                                                       s2, p1, p2, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        }
                    } else if (pad) {
                        auto padded_input  = common::pad_right_multi_double(input, pad, p1, p2);
                        auto padded_kernel = common::pad_right_multi(kernel, pad);

                        if (detail::prefer_sse<T>(k2 + pad)) {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2,
                                                                                                   0, 0, T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        } else {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(0)(c), padded_kernel(0)(k), conv(k)(c), s1, s2,
                                                                                                   0, 0, T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), padded_kernel(i)(k), conv(k)(c), s1,
                                                                                                       s2, 0, 0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        }
                    } else {
                        cpp_assert(!pad, "Invalid padding configuration");

                        auto padded_input = common::pad_right_multi_double(input, 0, p1, p2);

                        if (detail::prefer_sse<T>(k2)) {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, 0, 0,
                                                                                                   T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        } else {
                            auto fun_kc = [&](const size_t first, const size_t last) {
                                //i = 0
                                for (size_t kc = first; kc < last; ++kc) {
                                    const size_t k = kc / C;
                                    const size_t c = kc % C;

                                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, 0, 0,
                                                                                                   T(0));
                                }

                                for (size_t i = 1; i < N; ++i) {
                                    for (size_t kc = first; kc < last; ++kc) {
                                        const size_t k = kc / C;
                                        const size_t c = kc % C;

                                        detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, 0,
                                                                                                       0, T(1));
                                    }
                                }
                            };

                            engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
                        }
                    }

                    conv.invalidate_gpu();

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

                engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
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

                engine_dispatch_1d(fun_kc, 0, K * C, 4UL);
            }

            conv.invalidate_gpu();
        }
    } else {
        cpp_unreachable("Invalid call to vec::conv2_full");
    }
}

} //end of namespace etl::impl::vec
