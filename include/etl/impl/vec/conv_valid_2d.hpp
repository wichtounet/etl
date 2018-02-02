//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/common/conv.hpp"
#include "etl/impl/vec/conv_valid_kernels.hpp"

namespace etl::impl::vec {

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_flipped([[maybe_unused]] const I& input,
                         [[maybe_unused]] const K& kernel,
                         [[maybe_unused]] C&& conv,
                         [[maybe_unused]] size_t s1,
                         [[maybe_unused]] size_t s2,
                         [[maybe_unused]] size_t p1,
                         [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, K, C>) {
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

        if constexpr (padding_impl) {
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

    } else {
        cpp_unreachable("Invalid call to vec::conv2_valid_multi_multi_flipped");
    }
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid([[maybe_unused]] const I& input,
                 [[maybe_unused]] const K& kernel,
                 [[maybe_unused]] C&& conv,
                 [[maybe_unused]] size_t s1,
                 [[maybe_unused]] size_t s2,
                 [[maybe_unused]] size_t p1,
                 [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, K, C>) {
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

        if constexpr (padding_impl) {
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
    } else {
        cpp_unreachable("Invalid call to vec::conv2_valid_multi_multi_flipped");
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
void conv2_valid_multi([[maybe_unused]] const I& input,
                       [[maybe_unused]] const KK& kernel,
                       [[maybe_unused]] C&& conv,
                       [[maybe_unused]] size_t s1,
                       [[maybe_unused]] size_t s2,
                       [[maybe_unused]] size_t p1,
                       [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, C>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        const size_t K  = etl::dim<0>(kernel);
        const size_t k2 = etl::dim<2>(kernel);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        if constexpr (padding_impl) {
            static constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
            static constexpr size_t SS = AS / 2;

            if (k2 < SS || k2 % AS > 0) {
                const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

                auto padded_input = common::pad_right(input, pad);

                if (detail::prefer_sse<T>(k2 + pad)) {
                    auto fun_k = [&](const size_t first, const size_t last) {
                        for (size_t k = first; k < last; ++k) {
                            auto padded_kernel = common::pad_right_flip(kernel(k), pad);

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input, padded_kernel, conv(k), s1, s2, p1, p2, 0.0);
                        }
                    };

                    engine_dispatch_1d(fun_k, 0, K, 2UL);
                } else {
                    auto fun_k = [&](const size_t first, const size_t last) {
                        for (size_t k = first; k < last; ++k) {
                            auto padded_kernel = common::pad_right_flip(kernel(k), pad);

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input, padded_kernel, conv(k), s1, s2, p1, p2, 0.0);
                        }
                    };

                    engine_dispatch_1d(fun_k, 0, K, 2UL);
                }

                return;
            }
        }

        if (detail::prefer_sse<T>(etl::dim<2>(kernel))) {
            auto fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input, kernel(k), conv(k), s1, s2, p1, p2, 0.0);
                }
            };

            engine_dispatch_1d(fun_k, 0, K, 2UL);
        } else {
            auto fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input, kernel(k), conv(k), s1, s2, p1, p2, 0.0);
                }
            };

            engine_dispatch_1d(fun_k, 0, K, 2UL);
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to vec::conv2_valid_multi_multi_flipped");
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
void conv2_valid_multi_flipped([[maybe_unused]] const I& input,
                               [[maybe_unused]] const KK& kernel,
                               [[maybe_unused]] C&& conv,
                               [[maybe_unused]] size_t s1,
                               [[maybe_unused]] size_t s2,
                               [[maybe_unused]] size_t p1,
                               [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, C>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        const size_t K  = etl::dim<0>(kernel);
        const size_t k2 = etl::dim<2>(kernel);

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        if constexpr (padding_impl) {
            constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
            constexpr size_t SS = AS / 2;

            if (k2 < SS || k2 % AS > 0) {
                const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

                auto padded_input = common::pad_right(input, pad);

                if (detail::prefer_sse<T>(k2 + pad)) {
                    auto fun_k = [&](const size_t first, const size_t last) {
                        for (size_t k = first; k < last; ++k) {
                            auto padded_kernel = common::pad_right(kernel(k), pad);

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input, padded_kernel, conv(k), s1, s2, p1, p2, T(0));
                        }
                    };

                    engine_dispatch_1d(fun_k, 0, K, 2UL);
                } else {
                    auto fun_k = [&](const size_t first, const size_t last) {
                        for (size_t k = first; k < last; ++k) {
                            auto padded_kernel = common::pad_right(kernel(k), pad);

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input, padded_kernel, conv(k), s1, s2, p1, p2, T(0));
                        }
                    };

                    engine_dispatch_1d(fun_k, 0, K, 2UL);
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

            engine_dispatch_1d(fun_k, 0, K, 2UL);
        } else {
            auto fun_k = [&](const size_t first, const size_t last) {
                for (size_t k = first; k < last; ++k) {
                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input, kernel(k), conv(k), s1, s2, p1, p2, T(0));
                }
            };

            engine_dispatch_1d(fun_k, 0, K, 2UL);
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to vec::conv2_valid_multi_multi_flipped");
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
void conv2_valid_multi_multi([[maybe_unused]] const I& input,
                             [[maybe_unused]] const KK& kernel,
                             [[maybe_unused]] C&& conv,
                             [[maybe_unused]] size_t s1,
                             [[maybe_unused]] size_t s2,
                             [[maybe_unused]] size_t p1,
                             [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, C>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        const size_t k2 = etl::dim<2>(kernel);
        const size_t K  = etl::dim<0>(kernel);
        const size_t N  = etl::dim<0>(input);
        const size_t KN = K * N;

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        if constexpr (padding_impl) {
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

                    engine_dispatch_1d(fun_kn, 0, KN, 2UL);
                } else {
                    auto fun_kn = [&](const size_t first, const size_t last) {
                        for (size_t kn = first; kn < last; ++kn) {
                            size_t k = kn / N;
                            size_t n = kn % N;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(n), padded_kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                        }
                    };

                    engine_dispatch_1d(fun_kn, 0, KN, 2UL);
                }

                return;
            }
        }

        if (detail::prefer_sse<T>(etl::dim<2>(kernel))) {
            auto fun_kn = [&](const size_t first, const size_t last) {
                for (size_t kn = first; kn < last; ++kn) {
                    size_t k = kn / N;
                    size_t n = kn % N;

                    detail::conv2_valid_micro_kernel<detail::safe_sse_vec>(input(n), kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                }
            };

            engine_dispatch_1d(fun_kn, 0, KN, 2UL);
        } else {
            auto fun_kn = [&](const size_t first, const size_t last) {
                for (size_t kn = first; kn < last; ++kn) {
                    size_t k = kn / N;
                    size_t n = kn % N;

                    detail::conv2_valid_micro_kernel<detail::safe_avx_vec>(input(n), kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                }
            };

            engine_dispatch_1d(fun_kn, 0, KN, 2UL);
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to vec::conv2_valid_multi_multi_flipped");
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
void conv2_valid_multi_multi_flipped([[maybe_unused]] const I& input,
                                     [[maybe_unused]] const KK& kernel,
                                     [[maybe_unused]] C&& conv,
                                     [[maybe_unused]] size_t s1,
                                     [[maybe_unused]] size_t s2,
                                     [[maybe_unused]] size_t p1,
                                     [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, KK, C>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        const size_t k2 = etl::dim<2>(kernel);
        const size_t K  = etl::dim<0>(kernel);
        const size_t N  = etl::dim<0>(input);
        const size_t KN = K * N;

        input.ensure_cpu_up_to_date();
        kernel.ensure_cpu_up_to_date();

        if constexpr (padding_impl) {
            constexpr size_t AS = std::is_same<T, float>::value ? 8 : 4;
            constexpr size_t SS = AS / 2;

            if (k2 < SS || k2 % AS > 0) {
                const size_t pad = k2 < SS ? SS - k2 % SS : AS - k2 % AS;

                auto padded_input  = common::pad_right_multi(input, pad);
                auto padded_kernel = common::pad_right_multi(kernel, pad);

                if (detail::prefer_sse<T>(k2 + pad)) {
                    auto fun_kn = [&](const size_t first, const size_t last) {
                        for (size_t kn = first; kn < last; ++kn) {
                            size_t k = kn / N;
                            size_t n = kn % N;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(padded_input(n), padded_kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                        }
                    };

                    engine_dispatch_1d(fun_kn, 0, KN, 2UL);
                } else {
                    auto fun_kn = [&](const size_t first, const size_t last) {
                        for (size_t kn = first; kn < last; ++kn) {
                            size_t k = kn / N;
                            size_t n = kn % N;

                            detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(padded_input(n), padded_kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                        }
                    };

                    engine_dispatch_1d(fun_kn, 0, KN, 2UL);
                }

                return;
            }
        }

        if (detail::prefer_sse<T>(etl::dim<2>(kernel))) {
            auto fun_kn = [&](const size_t first, const size_t last) {
                for (size_t kn = first; kn < last; ++kn) {
                    size_t k = kn / N;
                    size_t n = kn % N;

                    detail::conv2_valid_flipped_micro_kernel<detail::safe_sse_vec>(input(n), kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                }
            };

            engine_dispatch_1d(fun_kn, 0, KN, 2UL);
        } else {
            auto fun_kn = [&](const size_t first, const size_t last) {
                for (size_t kn = first; kn < last; ++kn) {
                    size_t k = kn / N;
                    size_t n = kn % N;

                    detail::conv2_valid_flipped_micro_kernel<detail::safe_avx_vec>(input(n), kernel(k), conv(k)(n), s1, s2, p1, p2, T(0));
                }
            };

            engine_dispatch_1d(fun_kn, 0, KN, 2UL);
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to vec::conv2_valid_multi_multi_flipped");
    }
}

} //end of namespace etl::impl::vec
