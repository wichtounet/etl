//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/common/conv.hpp"
#include "etl/impl/vec/conv.hpp"

namespace etl::impl::vec {

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi([[maybe_unused]] const I& input,
                            [[maybe_unused]] const K_T& kernels,
                            [[maybe_unused]] C&& conv,
                            [[maybe_unused]] size_t s1,
                            [[maybe_unused]] size_t s2,
                            [[maybe_unused]] size_t p1,
                            [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, K_T, C>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        const size_t K  = etl::dim<0>(kernels);
        const size_t i1 = etl::dim<0>(input);
        const size_t i2 = etl::dim<1>(input);
        const size_t k1 = etl::dim<1>(kernels);
        const size_t k2 = etl::dim<2>(kernels);

        // unit-strided result dimensions
        const size_t c1 = (i1 - k1 + 2 * p1) + 1;
        const size_t c2 = (i2 - k2 + 2 * p2) + 1;

        // real final dimensions
        const size_t f1 = etl::dim<1>(conv);
        const size_t f2 = etl::dim<2>(conv);

        input.ensure_cpu_up_to_date();
        kernels.ensure_cpu_up_to_date();

        auto prepared_k = force_temporary(kernels);

        // Flip the kernels
        prepared_k.deep_fflip_inplace();

        etl::dyn_matrix<T, 2> input_col(k1 * k2, c1 * c2);

        if (p1 || p2) {
            etl::dyn_matrix<T, 2> input_padded(i1 + 2 * p1, i2 + 2 * p2);
            input_padded = T(0);

            impl::common::pad_2d_input(input, input_padded, p1, p2);

            im2col_direct_tr(input_col, input_padded, k1, k2);
        } else {
            im2col_direct_tr(input_col, input, k1, k2);
        }

        if (s1 > 1 || s2 > 1) {
            etl::dyn_matrix<T, 3> tmp_result(K, c1, c2);

            gemm_large_kernel_rr_to_r<default_vec>(prepared_k.memory_start(), input_col.memory_start(), tmp_result.memory_start(), K, c1 * c2, k1 * k2, T(1), T(0));

            // Strided copy of the large result into the small result
            for (size_t k = 0; k < K; ++k) {
                for (size_t i = 0; i < f1; ++i) {
                    for (size_t j = 0; j < f2; ++j) {
                        conv(k, i, j) = tmp_result(k, i * s1, j * s2);
                    }
                }
            }
        } else {
            gemm_large_kernel_rr_to_r<default_vec>(prepared_k.memory_start(), input_col.memory_start(), conv.memory_start(), K, f1 * f2, k1 * k2, T(1), T(0));
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv2_valid_multi");
    }
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi_flipped([[maybe_unused]] I&& input,
                                    [[maybe_unused]] K_T&& kernels,
                                    [[maybe_unused]] C&& conv,
                                    [[maybe_unused]] size_t s1,
                                    [[maybe_unused]] size_t s2,
                                    [[maybe_unused]] size_t p1,
                                    [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, K_T, C>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        const size_t K  = etl::dim<0>(kernels);
        const size_t i1 = etl::dim<0>(input);
        const size_t i2 = etl::dim<1>(input);
        const size_t k1 = etl::dim<1>(kernels);
        const size_t k2 = etl::dim<2>(kernels);

        // unit-strided result dimensions
        const size_t c1 = (i1 - k1 + 2 * p1) + 1;
        const size_t c2 = (i2 - k2 + 2 * p2) + 1;

        // real final dimensions
        const size_t f1 = etl::dim<1>(conv);
        const size_t f2 = etl::dim<2>(conv);

        input.ensure_cpu_up_to_date();
        kernels.ensure_cpu_up_to_date();

        etl::dyn_matrix<T, 2> input_col(k1 * k2, c1 * c2);

        if (p1 || p2) {
            etl::dyn_matrix<T, 2> input_padded(i1 + 2 * p1, i2 + 2 * p2);
            input_padded = T(0);

            impl::common::pad_2d_input(input, input_padded, p1, p2);

            im2col_direct_tr(input_col, input_padded, k1, k2);
        } else {
            im2col_direct_tr(input_col, input, k1, k2);
        }

        if (s1 > 1 || s2 > 1) {
            etl::dyn_matrix<T, 3> tmp_result(K, c1, c2);

            gemm_large_kernel_rr_to_r<default_vec>(kernels.memory_start(), input_col.memory_start(), tmp_result.memory_start(), K, c1 * c2, k1 * k2, T(1), T(0));

            // Strided copy of the large result into the small result
            for (size_t k = 0; k < K; ++k) {
                for (size_t i = 0; i < f1; ++i) {
                    for (size_t j = 0; j < f2; ++j) {
                        conv(k, i, j) = tmp_result(k, i * s1, j * s2);
                    }
                }
            }
        } else {
            gemm_large_kernel_rr_to_r<default_vec>(kernels.memory_start(), input_col.memory_start(), conv.memory_start(), K, f1 * f2, k1 * k2, T(1), T(0));
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv2_valid_multi_flipped");
    }
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple images and multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi_multi([[maybe_unused]] const I& input,
                                  [[maybe_unused]] const K_T& kernels,
                                  [[maybe_unused]] C&& conv,
                                  [[maybe_unused]] size_t s1,
                                  [[maybe_unused]] size_t s2,
                                  [[maybe_unused]] size_t p1,
                                  [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, K_T, C>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        const size_t N  = etl::dim<0>(input);
        const size_t i1 = etl::dim<1>(input);
        const size_t i2 = etl::dim<2>(input);

        const size_t K  = etl::dim<0>(kernels);
        const size_t k1 = etl::dim<1>(kernels);
        const size_t k2 = etl::dim<2>(kernels);

        // unit-strided result dimensions
        const size_t c1 = (i1 - k1 + 2 * p1) + 1;
        const size_t c2 = (i2 - k2 + 2 * p2) + 1;

        // real final dimensions
        const size_t f1 = etl::dim<2>(conv);
        const size_t f2 = etl::dim<3>(conv);

        input.ensure_cpu_up_to_date();
        kernels.ensure_cpu_up_to_date();

        auto prepared_k = force_temporary(kernels);

        // Flip the kernels
        prepared_k.deep_fflip_inplace();

        etl::dyn_matrix<T, 2> input_col(k1 * k2, N * c1 * c2);

        if (p1 || p2) {
            etl::dyn_matrix<T, 3> input_padded(N, i1 + 2 * p1, i2 + 2 * p2);
            input_padded = T(0);

            for (size_t i = 0; i < N; ++i) {
                impl::common::pad_2d_input(input(i), input_padded(i), p1, p2);
            }

            im2col_direct_tr_multi(input_col, input_padded, k1, k2);
        } else {
            im2col_direct_tr_multi(input_col, input, k1, k2);
        }

        if (s1 > 1 || s2 > 1) {
            etl::dyn_matrix<T, 4> tmp_result(K, N, c1, c2);

            gemm_large_kernel_rr_to_r<default_vec>(prepared_k.memory_start(), input_col.memory_start(), tmp_result.memory_start(), K, N * c1 * c2, k1 * k2,
                                                   T(1), T(0));

            // Strided copy of the large result into the small result
            for (size_t k = 0; k < K; ++k) {
                for (size_t i = 0; i < N; ++i) {
                    for (size_t ii = 0; ii < f1; ++ii) {
                        for (size_t j = 0; j < f2; ++j) {
                            conv(k, i, ii, j) = tmp_result(k, i, ii * s1, j * s2);
                        }
                    }
                }
            }
        } else {
            gemm_large_kernel_rr_to_r<default_vec>(prepared_k.memory_start(), input_col.memory_start(), conv.memory_start(), K, N * c1 * c2, k1 * k2, T(1), T(0));
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv2_valid_multi_flipped");
    }
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple images and multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi_multi_flipped([[maybe_unused]] const I& input,
                                          [[maybe_unused]] const K_T& kernels,
                                          [[maybe_unused]] C&& conv,
                                          [[maybe_unused]] size_t s1,
                                          [[maybe_unused]] size_t s2,
                                          [[maybe_unused]] size_t p1,
                                          [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I, K_T, C>) {
        cpp_assert(vec_enabled, "Cannot use vectorized mode");
        cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

        using T = value_t<I>;

        const size_t N  = etl::dim<0>(input);
        const size_t i1 = etl::dim<1>(input);
        const size_t i2 = etl::dim<2>(input);

        const size_t K  = etl::dim<0>(kernels);
        const size_t k1 = etl::dim<1>(kernels);
        const size_t k2 = etl::dim<2>(kernels);

        // unit-strided result dimensions
        const size_t c1 = (i1 - k1 + 2 * p1) + 1;
        const size_t c2 = (i2 - k2 + 2 * p2) + 1;

        // real final dimensions
        const size_t f1 = etl::dim<2>(conv);
        const size_t f2 = etl::dim<3>(conv);

        input.ensure_cpu_up_to_date();
        kernels.ensure_cpu_up_to_date();

        etl::dyn_matrix<T, 2> input_col(k1 * k2, N * c1 * c2);

        if (p1 || p2) {
            etl::dyn_matrix<T, 3> input_padded(N, i1 + 2 * p1, i2 + 2 * p2);
            input_padded = T(0);

            for (size_t i = 0; i < N; ++i) {
                impl::common::pad_2d_input(input(i), input_padded(i), p1, p2);
            }

            im2col_direct_tr_multi(input_col, input_padded, k1, k2);
        } else {
            im2col_direct_tr_multi(input_col, input, k1, k2);
        }

        if (s1 > 1 || s2 > 1) {
            etl::dyn_matrix<T, 4> tmp_result(K, N, c1, c2);

            gemm_large_kernel_rr_to_r<default_vec>(kernels.memory_start(), input_col.memory_start(), tmp_result.memory_start(), K, N * c1 * c2, k1 * k2, T(1), T(0));

            // Strided copy of the large result into the small result
            for (size_t k = 0; k < K; ++k) {
                for (size_t i = 0; i < N; ++i) {
                    for (size_t ii = 0; ii < f1; ++ii) {
                        for (size_t j = 0; j < f2; ++j) {
                            conv(k, i, ii, j) = tmp_result(k, i, ii * s1, j * s2);
                        }
                    }
                }
            }
        } else {
            gemm_large_kernel_rr_to_r<default_vec>(kernels.memory_start(), input_col.memory_start(), conv.memory_start(), K, N * c1 * c2, k1 * k2, T(1), T(0));
        }

        conv.invalidate_gpu();
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv2_valid_multi_flipped");
    }
}

/*!
 * \brief Compute a 4D valid convolution using a vectorized matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix (already flipped)
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename KS_T, typename C_T>
void blas_conv4_valid_prepared(I_T&& input, K_T&& kernel, KS_T&& kernels, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I_T>;

    const auto N = etl::dim<0>(input);  // The number of images
    const auto K = etl::dim<0>(kernel); // The number of kernels
    const auto C = etl::dim<1>(input);  // The number of channels

    const auto n1 = etl::dim<2>(input);
    const auto n2 = etl::dim<3>(input);

    const auto m1 = etl::dim<2>(kernel);
    const auto m2 = etl::dim<3>(kernel);

    const auto c1 = etl::dim<2>(conv);
    const auto c2 = etl::dim<3>(conv);

    input.ensure_cpu_up_to_date();
    kernels.ensure_cpu_up_to_date();

    conv = T(0.0);

    auto batch_fun_n = [&](const size_t first, const size_t last) {
        if (last - first) {
            // unit-strided result dimensions
            const size_t sc1 = (n1 - m1 + 2 * p1) + 1;
            const size_t sc2 = (n2 - m2 + 2 * p2) + 1;

            etl::dyn_matrix<T, 2> input_col(m1 * m2, sc1 * sc2);

            // Optimize for the most common case
            if (cpp_likely(!p1 && !p2 && s1 == 1 && s2 == 1)) {
                for (size_t i = first; i < last; ++i) {
                    for (size_t c = 0; c < C; ++c) {
                        im2col_direct_tr(input_col, input(i)(c), m1, m2);

                        gemm_large_kernel_rr_to_r<default_vec>(kernels(c).memory_start(), input_col.memory_start(), conv(i).memory_start(), K, c1 * c2, m1 * m2,
                                                               T(1), T(1.0));
                    }
                }
            } else {
                etl::dyn_matrix<T, 2> input_padded(n1 + 2 * p1, n2 + 2 * p2);
                etl::dyn_matrix<T, 3> tmp_result(K, sc1, sc2);

                for (size_t i = first; i < last; ++i) {
                    for (size_t c = 0; c < C; ++c) {
                        if (p1 || p2) {
                            input_padded = T(0.0);

                            impl::common::pad_2d_input(input(i)(c), input_padded, p1, p2);

                            im2col_direct_tr(input_col, input_padded, m1, m2);
                        } else {
                            im2col_direct_tr(input_col, input(i)(c), m1, m2);
                        }

                        if (s1 > 1 || s2 > 1) {
                            gemm_large_kernel_rr_to_r<default_vec>(kernels(c).memory_start(), input_col.memory_start(), tmp_result.memory_start(), K, sc1 * sc2,
                                                                   m1 * m2, T(1), T(0.0));

                            // Strided copy of the large result into the small result
                            for (size_t k = 0; k < K; ++k) {
                                for (size_t ii = 0; ii < c1; ++ii) {
                                    for (size_t j = 0; j < c2; ++j) {
                                        conv(i, k, ii, j) += tmp_result(k, ii * s1, j * s2);
                                    }
                                }
                            }
                        } else {
                            gemm_large_kernel_rr_to_r<default_vec>(kernels(c).memory_start(), input_col.memory_start(), conv(i).memory_start(), K, c1 * c2,
                                                                   m1 * m2, T(1), T(1.0));
                        }
                    }
                }
            }
        }
    };

    engine_dispatch_1d_serial(batch_fun_n, 0, N, 2UL);

    conv.invalidate_gpu();
}

/*!
 * \brief Compute a 4D valid convolution using a vectorized matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid([[maybe_unused]] I_T&& input,
                      [[maybe_unused]] K_T&& kernel,
                      [[maybe_unused]] C_T&& conv,
                      [[maybe_unused]] size_t s1,
                      [[maybe_unused]] size_t s2,
                      [[maybe_unused]] size_t p1,
                      [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I_T, K_T, C_T>) {
        const auto K = etl::dim<0>(kernel); // The number of kernels
        const auto C = etl::dim<1>(input);  // The number of channels

        const auto m1 = etl::dim<2>(kernel);
        const auto m2 = etl::dim<3>(kernel);

        etl::dyn_matrix<value_t<I_T>, 4> kernels(C, K, m1, m2);

        for (size_t c = 0; c < C; ++c) {
            for (size_t k = 0; k < K; ++k) {
                kernels(c)(k) = fflip(kernel(k)(c));
            }
        }

        blas_conv4_valid_prepared(input, kernel, kernels, conv, s1, s2, p1, p2);
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv4_valid");
    }
}

/*!
 * \brief Compute a 4D valid convolution using a vectorized matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix, with flipped kernels
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_flipped([[maybe_unused]] I_T&& input,
                              [[maybe_unused]] K_T&& kernel,
                              [[maybe_unused]] C_T&& conv,
                              [[maybe_unused]] size_t s1,
                              [[maybe_unused]] size_t s2,
                              [[maybe_unused]] size_t p1,
                              [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I_T, K_T, C_T>) {
        const auto K = etl::dim<0>(kernel); // The number of kernels
        const auto C = etl::dim<1>(input);  // The number of channels

        const auto m1 = etl::dim<2>(kernel);
        const auto m2 = etl::dim<3>(kernel);

        etl::dyn_matrix<value_t<I_T>, 4> kernels(C, K, m1, m2);

        for (size_t c = 0; c < C; ++c) {
            for (size_t k = 0; k < K; ++k) {
                kernels(c)(k) = kernel(k)(c);
            }
        }

        blas_conv4_valid_prepared(input, kernel, kernels, conv, s1, s2, p1, p2);
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv4_valid");
    }
}

/*!
 * \brief Compute a 4D valid filter convolution using a vectorized matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix, with flipped kernels
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_filter_prepared(I_T&& input, K_T&& kernel, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(vec_enabled, "Cannot use vectorized mode");
    cpp_assert(vectorize_impl, "Cannot use vectorized implementation");

    using T = value_t<I_T>;

    const auto I = etl::dim<0>(input);
    const auto K = etl::dim<0>(conv);
    const auto C = etl::dim<1>(conv);

    const auto f1 = etl::dim<2>(conv);
    const auto f2 = etl::dim<3>(conv);

    const auto i1 = etl::dim<2>(input);
    const auto i2 = etl::dim<3>(input);

    const auto k1 = etl::dim<2>(kernel);
    const auto k2 = etl::dim<3>(kernel);

    // unit-strided result dimensions
    const size_t c1 = (i1 - k1 + 2 * p1) + 1;
    const size_t c2 = (i2 - k2 + 2 * p2) + 1;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    etl::dyn_matrix<T, 4> conv_temp(C, K, f1, f2);
    conv_temp = T(0);

    auto batch_fun_c = [&](const size_t first, const size_t last) {
        for (size_t c = first; c < last; ++c) {
            etl::dyn_matrix<T, 2> input_col(k1 * k2, c1 * c2);

            for (size_t i = 0; i < I; ++i) {
                // Optimize for the most common case
                if (cpp_likely(!p1 && !p2 && s1 == 1 && s2 == 1)) {
                    im2col_direct_tr(input_col, input(i)(c), k1, k2);
                    gemm_large_kernel_rr_to_r<default_vec>(kernel(i).memory_start(), input_col.memory_start(), conv_temp(c).memory_start(), K, f1 * f2, k1 * k2,
                                                           T(1), T(1.0));
                } else {
                    if (p1 || p2) {
                        etl::dyn_matrix<T, 2> input_padded(i1 + 2 * p1, i2 + 2 * p2);
                        input_padded = T(0);

                        impl::common::pad_2d_input(input(i)(c), input_padded, p1, p2);

                        im2col_direct_tr(input_col, input_padded, k1, k2);
                    } else {
                        im2col_direct_tr(input_col, input(i)(c), k1, k2);
                    }

                    if (s1 > 1 || s2 > 1) {
                        etl::dyn_matrix<T, 3> tmp_result(K, c1, c2);

                        gemm_large_kernel_rr_to_r<default_vec>(kernel(i).memory_start(), input_col.memory_start(), tmp_result.memory_start(), K, c1 * c2,
                                                               k1 * k2, T(1), T(0.0));

                        // Strided copy of the large result into the small result
                        for (size_t k = 0; k < K; ++k) {
                            for (size_t ii = 0; ii < f1; ++ii) {
                                for (size_t j = 0; j < f2; ++j) {
                                    conv_temp(c, k, ii, j) += tmp_result(k, ii * s1, j * s2);
                                }
                            }
                        }
                    } else {
                        gemm_large_kernel_rr_to_r<default_vec>(kernel(i).memory_start(), input_col.memory_start(), conv_temp(c).memory_start(), K, f1 * f2,
                                                               k1 * k2, T(1), T(1.0));
                    }
                }
            }
        }

        for (size_t c = first; c < last; ++c) {
            for (size_t k = 0; k < K; ++k) {
                conv(k)(c) = conv_temp(c)(k);
            }
        }
    };

    engine_dispatch_1d_serial(batch_fun_c, 0, C, 2UL);

    conv.invalidate_gpu();
}

/*!
 * \brief Compute a 4D valid filter convolution using a vectorized matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_filter([[maybe_unused]] I_T&& input,
                             [[maybe_unused]] K_T&& kernel,
                             [[maybe_unused]] C_T&& conv,
                             [[maybe_unused]] size_t s1,
                             [[maybe_unused]] size_t s2,
                             [[maybe_unused]] size_t p1,
                             [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I_T, K_T, C_T>) {
        auto prepared_k = force_temporary(kernel);

        // Flip the kernels
        prepared_k.deep_fflip_inplace();

        blas_conv4_valid_filter_prepared(input, prepared_k, conv, s1, s2, p1, p2);
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv4_valid");
    }
}

/*!
 * \brief Compute a 4D valid filter convolution using a vectorized matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix, with flipped kernels
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_filter_flipped([[maybe_unused]] I_T&& input,
                                     [[maybe_unused]] K_T&& kernel,
                                     [[maybe_unused]] C_T&& conv,
                                     [[maybe_unused]] size_t s1,
                                     [[maybe_unused]] size_t s2,
                                     [[maybe_unused]] size_t p1,
                                     [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I_T, K_T, C_T>) {
        blas_conv4_valid_filter_prepared(input, kernel, conv, s1, s2, p1, p2);
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv4_valid");
    }
}

/*!
 * \brief Compute a 4D valid backward convolution, with prepared (flipped) kernels,  using a BLAS matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix, with flipped kernels
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_back_prepared(I_T&& input, K_T&& kernel, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    using T = value_t<I_T>;

    const auto N = etl::dim<0>(input);
    const auto C = etl::dim<1>(kernel);
    const auto K = etl::dim<1>(input);

    const size_t i1 = etl::dim<2>(input);
    const size_t i2 = etl::dim<3>(input);
    const size_t k1 = etl::dim<2>(kernel);
    const size_t k2 = etl::dim<3>(kernel);

    // unit-strided result dimensions
    const size_t c1 = (i1 - k1 + 2 * p1) + 1;
    const size_t c2 = (i2 - k2 + 2 * p2) + 1;

    // real final dimensions
    const size_t f1 = etl::dim<2>(conv);
    const size_t f2 = etl::dim<3>(conv);

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    auto batch_fun_n = [&](const size_t first, const size_t last) {
        if (last - first) {
            etl::dyn_matrix<T, 2> input_col(k1 * k2, c1 * c2);

            // Optimize for the most common case
            if (cpp_likely(!p1 && !p2 && s1 == 1 && s2 == 1)) {
                for (size_t i = first; i < last; ++i) {
                    for (size_t k = 0; k < K; ++k) {
                        // use im2col on input(i)(k)
                        im2col_direct_tr(input_col, input(i)(k), k1, k2);

                        // conv(i) = kernel(k) * input_col
                        gemm_large_kernel_rr_to_r<default_vec>(kernel(k).memory_start(), input_col.memory_start(), conv(i).memory_start(), C, c1 * c2, k1 * k2,
                                                               T(1), T(1.0));
                    }
                }
            } else {
                etl::dyn_matrix<T, 2> input_padded(i1 + 2 * p1, i2 + 2 * p2);
                etl::dyn_matrix<T, 3> tmp_result(C, c1, c2);

                for (size_t i = first; i < last; ++i) {
                    for (size_t k = 0; k < K; ++k) {
                        // use im2col on input(i)(k)

                        if (p1 || p2) {
                            input_padded = T(0);
                            impl::common::pad_2d_input(input(i)(k), input_padded, p1, p2);
                            im2col_direct_tr(input_col, input_padded, k1, k2);
                        } else {
                            im2col_direct_tr(input_col, input(i)(k), k1, k2);
                        }

                        if (s1 > 1 || s2 > 1) {
                            // tmp_result = kernel(k) * input_col
                            gemm_large_kernel_rr_to_r<default_vec>(kernel(k).memory_start(), input_col.memory_start(), tmp_result.memory_start(), C, c1 * c2,
                                                                   k1 * k2, T(1), T(0.0));

                            // Strided copy of the large result into the small result
                            for (size_t c = 0; c < C; ++c) {
                                for (size_t m = 0; m < f1; ++m) {
                                    for (size_t n = 0; n < f2; ++n) {
                                        conv(i, c, m, n) += tmp_result(c, m * s1, n * s2);
                                    }
                                }
                            }
                        } else {
                            // conv(i) = kernel(k) * input_col
                            gemm_large_kernel_rr_to_r<default_vec>(kernel(k).memory_start(), input_col.memory_start(), conv(i).memory_start(), C, c1 * c2,
                                                                   k1 * k2, T(1), T(1.0));
                        }
                    }
                }
            }
        }
    };

    engine_dispatch_1d_serial(batch_fun_n, 0, N, engine_select_parallel(N, 2) && !is_blas_parallel);

    conv.invalidate_gpu();
}

/*!
 * \brief Compute a 4D valid backward convolution, using a BLAS matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix, with flipped kernels
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_back([[maybe_unused]] I_T&& input,
                           [[maybe_unused]] K_T&& kernel,
                           [[maybe_unused]] C_T&& conv,
                           [[maybe_unused]] size_t s1,
                           [[maybe_unused]] size_t s2,
                           [[maybe_unused]] size_t p1,
                           [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I_T, K_T, C_T>) {
        auto prepared_k = force_temporary(kernel);

        // Flip the kernels
        prepared_k.deep_fflip_inplace();

        blas_conv4_valid_back_prepared(input, prepared_k, conv, s1, s2, p1, p2);
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv4_valid");
    }
}

/*!
 * \brief Compute a 4D valid backward convolution, with flipped kernels,  using a BLAS matrix multiplication kernel
 * \param input The input matrix
 * \param kernel The kernel matrix, with flipped kernels
 * \param conv The output matrix
 * \param s1 The stride of the first dimension
 * \param s2 The stride of the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 */
template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_back_flipped([[maybe_unused]] I_T&& input,
                                   [[maybe_unused]] K_T&& kernel,
                                   [[maybe_unused]] C_T&& conv,
                                   [[maybe_unused]] size_t s1,
                                   [[maybe_unused]] size_t s2,
                                   [[maybe_unused]] size_t p1,
                                   [[maybe_unused]] size_t p2) {
    if constexpr (conv2_possible<vector_mode, I_T, K_T, C_T>) {
        blas_conv4_valid_back_prepared(input, kernel, conv, s1, s2, p1, p2);
    } else {
        cpp_unreachable("Invalid call to vec::blas_conv4_valid");
    }
}

} //end of namespace etl::impl::vec
