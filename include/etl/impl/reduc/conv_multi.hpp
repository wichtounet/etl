//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

namespace reduc {

/*!
 * \brief Pad the input matrix in the output matrix for convolution as multiplication
 * \param in The input matrix
 * \param out The output matrix
 */
template <typename F1, typename F2>
void complex_pad_4d(const F1& in, F2& out) {
    out.ensure_cpu_up_to_date();

    for (std::size_t outer1 = 0; outer1 < etl::dim<0>(in); ++outer1) {
        for (std::size_t outer2 = 0; outer2 < etl::dim<1>(in); ++outer2) {
            auto* direct = out(outer1)(outer2).memory_start();
            for (std::size_t i = 0; i < etl::dim<2>(in); ++i) {
                for (std::size_t j = 0; j < etl::dim<3>(in); ++j) {
                    direct[i * etl::dim<3>(out) + j] = in(outer1, outer2, i, j);
                }
            }
        }
    }
}

/*!
 * \brief Pad the input matrix in the output matrix for convolution as multiplication
 * \param in The input matrix
 * \param out The output matrix
 */
template <typename F1, typename F2>
void complex_pad_3d(const F1& in, F2& out) {
    out.ensure_cpu_up_to_date();

    for (std::size_t outer = 0; outer < etl::dim<0>(in); ++outer) {
        auto* direct = out(outer).memory_start();
        for (std::size_t i = 0; i < etl::dim<1>(in); ++i) {
            for (std::size_t j = 0; j < etl::dim<2>(in); ++j) {
                direct[i * etl::dim<2>(out) + j] = in(outer, i, j);
            }
        }
    }
}

/*!
 * \brief Pad the input matrix in the output matrix for convolution as multiplication
 * \param in The input matrix
 * \param out The output matrix
 * \param p1 The first dimension extra padding of the convolution
 * \param p2 The second dimension extra padding of the convolution
 */
template <typename F1, typename F2>
void pad_2d_input(const F1& in, F2&& out, size_t p1, size_t p2) {
    out.ensure_cpu_up_to_date();

    auto* direct = out.memory_start();

    for (std::size_t i = 0; i < etl::dim<0>(in); ++i) {
        for (std::size_t j = 0; j < etl::dim<1>(in); ++j) {
            direct[(i + p1) * etl::dim<1>(out) + (j + p2)] = in(i, j);
        }
    }
}

/*!
 * \brief Pad the input matrix in the output matrix for convolution as multiplication
 * \param in The input matrix
 * \param out The output matrix
 * \param p1 The first dimension extra padding of the convolution
 * \param p2 The second dimension extra padding of the convolution
 */
template <typename F1, typename F2>
void pad_3d_input(const F1& in, F2&& out, size_t p1, size_t p2) {
    out.ensure_cpu_up_to_date();

    for (std::size_t n = 0; n < etl::dim<0>(in); ++n) {
        auto* direct = out(n).memory_start();

        for (std::size_t i = 0; i < etl::dim<1>(in); ++i) {
            for (std::size_t j = 0; j < etl::dim<2>(in); ++j) {
                direct[(i + p1) * etl::dim<2>(out) + (j + p2)] = in(n, i, j);
            }
        }
    }
}

/*!
 * \brief FFT implementation of a 2D 'valid' convolution C = I * K, with multiple kernels.
 *
 * This works by doing a full convolution by FFT and then extracting
 * only the valid part of the convolution.
 *
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const std::size_t K = etl::dim<0>(kernels);
    const std::size_t i1 = etl::dim<0>(input);
    const std::size_t i2 = etl::dim<1>(input);
    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);

    // Dimensions of the final valid convolution (stride,padding)
    const std::size_t c1 = (i1 - k1 + 2 * p1) / s1 + 1;
    const std::size_t c2 = (i2 - k2 + 2 * p2) / s1 + 1;

    //Dimensions of the valid convolution (unit strided)
    const std::size_t v1 = (i1 - k1 + 2 * p1) + 1;
    const std::size_t v2 = (i2 - k2 + 2 * p2) + 1;

    // Dimensions of the full convolution
    const std::size_t t1 = (i1 + k1 + 2 * p1) - 1;
    const std::size_t t2 = (i2 + k2 + 2 * p2) - 1;

    // Dimensions of the 'full' borders
    const std::size_t b1 = (t1 - v1) / 2;
    const std::size_t b2 = (t2 - v2) / 2;

    input.ensure_cpu_up_to_date();
    kernels.ensure_cpu_up_to_date();

    etl::dyn_matrix<etl::complex<value_t<I>>> input_padded(t1, t2);
    etl::dyn_matrix<etl::complex<value_t<I>>, 3> kernels_padded(K, t1, t2);
    etl::dyn_matrix<etl::complex<value_t<I>>, 3> tmp_result(K, t1, t2);

    pad_2d_input(input, input_padded, p1, p2);
    complex_pad_3d(kernels, kernels_padded);

    input_padded.fft2_inplace();
    kernels_padded.fft2_many_inplace();

    for (std::size_t k = 0; k < K; ++k) {
        tmp_result(k) = input_padded >> kernels_padded(k);
    }

    tmp_result.ifft2_many_inplace();

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t i = 0; i < c1; ++i) {
            for (std::size_t j = 0; j < c2; ++j) {
                conv(k, i, j) = tmp_result(k, i * s1 + b1, j * s2 + b2).real;
            }
        }
    }

    conv.invalidate_gpu();
}

/*!
 * \brief FFT implementation of a 2D 'valid' convolution C = I * K, with multiple kernels.
 *
 * This works by doing a full convolution by FFT and then extracting
 * only the valid part of the convolution.
 *
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi_multi(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const std::size_t N = etl::dim<0>(input);
    const std::size_t i1 = etl::dim<1>(input);
    const std::size_t i2 = etl::dim<2>(input);

    const std::size_t K = etl::dim<0>(kernels);
    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);

    // Dimensions of the final valid convolution (stride,padding)
    const std::size_t c1 = (i1 - k1 + 2 * p1) / s1 + 1;
    const std::size_t c2 = (i2 - k2 + 2 * p2) / s1 + 1;

    //Dimensions of the valid convolution (unit strided)
    const std::size_t v1 = (i1 - k1 + 2 * p1) + 1;
    const std::size_t v2 = (i2 - k2 + 2 * p2) + 1;

    // Dimensions of the full convolution
    const std::size_t t1 = (i1 + k1 + 2 * p1) - 1;
    const std::size_t t2 = (i2 + k2 + 2 * p2) - 1;

    // Dimensions of the 'full' borders
    const std::size_t b1 = (t1 - v1) / 2;
    const std::size_t b2 = (t2 - v2) / 2;

    input.ensure_cpu_up_to_date();
    kernels.ensure_cpu_up_to_date();

    etl::dyn_matrix<etl::complex<value_t<I>>, 3> input_padded(N, t1, t2);
    etl::dyn_matrix<etl::complex<value_t<I>>, 3> kernels_padded(K, t1, t2);
    etl::dyn_matrix<etl::complex<value_t<I>>, 4> tmp_result(K, N, t1, t2);

    pad_3d_input(input, input_padded, p1, p2);
    complex_pad_3d(kernels, kernels_padded);

    input_padded.fft2_many_inplace();
    kernels_padded.fft2_many_inplace();

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t n = 0; n < N; ++n) {
            tmp_result(k)(n) = input_padded(n) >> kernels_padded(k);
        }
    }

    tmp_result.ifft2_many_inplace();

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t n = 0; n < N; ++n) {
            for (std::size_t i = 0; i < c1; ++i) {
                for (std::size_t j = 0; j < c2; ++j) {
                    conv(k, n, i, j) = tmp_result(k, n, i * s1 + b1, j * s2 + b2).real;
                }
            }
        }
    }

    conv.invalidate_gpu();
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const std::size_t K  = etl::dim<0>(kernels);
    const std::size_t i1 = etl::dim<0>(input);
    const std::size_t i2 = etl::dim<1>(input);
    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);

    // unit-strided result dimensions
    const std::size_t c1 = (i1 - k1 + 2 * p1) + 1;
    const std::size_t c2 = (i2 - k2 + 2 * p2) + 1;

    // real final dimensions
    const std::size_t f1 = etl::dim<1>(conv);
    const std::size_t f2 = etl::dim<2>(conv);

    input.ensure_cpu_up_to_date();
    kernels.ensure_cpu_up_to_date();

    auto prepared_k = force_temporary(kernels);

    // Flip the kernels
    prepared_k.deep_fflip_inplace();

    etl::dyn_matrix<value_t<I>, 2> input_col(k1 * k2, c1 * c2);

    if(p1 || p2){
        etl::dyn_matrix<value_t<I>, 2> input_padded(i1 + 2 * p1, i2 + 2 * p2);
        input_padded = value_t<I>(0);

        pad_2d_input(input, input_padded, p1, p2);

        im2col_direct_tr(input_col, input_padded, k1, k2);
    } else {
        im2col_direct_tr(input_col, input, k1, k2);
    }

    if(s1 > 1 || s2 > 1){
        etl::dyn_matrix<value_t<I>, 3> tmp_result(K, c1, c2);

        etl::reshape(tmp_result, K, c1 * c2) = mul(etl::reshape(prepared_k, K, k1 * k2), input_col);

        // Strided copy of the large result into the small result
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t i = 0; i < f1; ++i) {
                for (std::size_t j = 0; j < f2; ++j) {
                    conv(k, i, j) = tmp_result(k, i * s1, j * s2);
                }
            }
        }
    } else {
        etl::reshape(conv, K, f1 * f2) = mul(etl::reshape(prepared_k, K, k1 * k2), input_col);
    }

    conv.invalidate_gpu();
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple images and multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi_multi(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const std::size_t N  = etl::dim<0>(input);
    const std::size_t i1 = etl::dim<1>(input);
    const std::size_t i2 = etl::dim<2>(input);

    const std::size_t K  = etl::dim<0>(kernels);
    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);

    // unit-strided result dimensions
    const std::size_t c1 = (i1 - k1 + 2 * p1) + 1;
    const std::size_t c2 = (i2 - k2 + 2 * p2) + 1;

    // real final dimensions
    const std::size_t f1 = etl::dim<2>(conv);
    const std::size_t f2 = etl::dim<3>(conv);

    input.ensure_cpu_up_to_date();
    kernels.ensure_cpu_up_to_date();

    auto prepared_k = force_temporary(kernels);

    // Flip the kernels
    prepared_k.deep_fflip_inplace();

    etl::dyn_matrix<value_t<I>, 2> input_col(k1 * k2, N * c1 * c2);

    if(p1 || p2){
        etl::dyn_matrix<value_t<I>, 3> input_padded(N, i1 + 2 * p1, i2 + 2 * p2);
        input_padded = value_t<I>(0);

        for(std::size_t i = 0; i < N; ++i){
            pad_2d_input(input(i), input_padded(i), p1, p2);
        }

        im2col_direct_tr_multi(input_col, input_padded, k1, k2);
    } else {
        im2col_direct_tr_multi(input_col, input, k1, k2);
    }

    if(s1 > 1 || s2 > 1){
        etl::dyn_matrix<value_t<I>, 4> tmp_result(K, N, c1, c2);

        etl::reshape(tmp_result, K, N * c1 * c2) = mul(etl::reshape(prepared_k, K, k1 * k2), input_col);

        // Strided copy of the large result into the small result
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t ii = 0; ii < f1; ++ii) {
                    for (std::size_t j = 0; j < f2; ++j) {
                        conv(k, i, ii, j) = tmp_result(k, i, ii * s1, j * s2);
                    }
                }
            }
        }
    } else {
        etl::reshape(conv, K, N * f1 * f2) = mul(etl::reshape(prepared_k, K, k1 * k2), input_col);
    }

    conv.invalidate_gpu();
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple images and multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi_multi_flipped(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const std::size_t N  = etl::dim<0>(input);
    const std::size_t i1 = etl::dim<1>(input);
    const std::size_t i2 = etl::dim<2>(input);

    const std::size_t K  = etl::dim<0>(kernels);
    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);

    // unit-strided result dimensions
    const std::size_t c1 = (i1 - k1 + 2 * p1) + 1;
    const std::size_t c2 = (i2 - k2 + 2 * p2) + 1;

    // real final dimensions
    const std::size_t f1 = etl::dim<2>(conv);
    const std::size_t f2 = etl::dim<3>(conv);

    input.ensure_cpu_up_to_date();
    kernels.ensure_cpu_up_to_date();

    etl::dyn_matrix<value_t<I>, 2> input_col(k1 * k2, N * c1 * c2);

    if(p1 || p2){
        etl::dyn_matrix<value_t<I>, 3> input_padded(N, i1 + 2 * p1, i2 + 2 * p2);
        input_padded = value_t<I>(0);

        for(std::size_t i = 0; i < N; ++i){
            pad_2d_input(input(i), input_padded(i), p1, p2);
        }

        im2col_direct_tr_multi(input_col, input_padded, k1, k2);
    } else {
        im2col_direct_tr_multi(input_col, input, k1, k2);
    }

    if(s1 > 1 || s2 > 1){
        etl::dyn_matrix<value_t<I>, 4> tmp_result(K, N, c1, c2);

        etl::reshape(tmp_result, K, N * c1 * c2) = mul(etl::reshape(kernels, K, k1 * k2), input_col);

        // Strided copy of the large result into the small result
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t i = 0; i < N; ++i) {
                for (std::size_t ii = 0; ii < f1; ++ii) {
                    for (std::size_t j = 0; j < f2; ++j) {
                        conv(k, i, ii, j) = tmp_result(k, i, ii * s1, j * s2);
                    }
                }
            }
        }
    } else {
        etl::reshape(conv, K, N * f1 * f2) = mul(etl::reshape(kernels, K, k1 * k2), input_col);
    }

    conv.invalidate_gpu();
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi_flipped(I&& input, K_T&& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    auto kernels_f = etl::force_temporary(kernels);

    kernels_f.deep_fflip_inplace();

    fft_conv2_valid_multi(input, kernels_f, conv, s1, s2, p1, p2);
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi_multi_flipped(I&& input, K_T&& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    auto kernels_f = etl::force_temporary(kernels);

    kernels_f.deep_fflip_inplace();

    fft_conv2_valid_multi_multi(input, kernels_f, conv, s1, s2, p1, p2);
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi_flipped(I&& input, K_T&& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const std::size_t K  = etl::dim<0>(kernels);
    const std::size_t i1 = etl::dim<0>(input);
    const std::size_t i2 = etl::dim<1>(input);
    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);

    // unit-strided result dimensions
    const std::size_t c1 = (i1 - k1 + 2 * p1) + 1;
    const std::size_t c2 = (i2 - k2 + 2 * p2) + 1;

    // real final dimensions
    const std::size_t f1 = etl::dim<1>(conv);
    const std::size_t f2 = etl::dim<2>(conv);

    input.ensure_cpu_up_to_date();
    kernels.ensure_cpu_up_to_date();

    etl::dyn_matrix<value_t<I>, 2> input_col(k1 * k2, c1 * c2);

    if(p1 || p2){
        etl::dyn_matrix<value_t<I>, 2> input_padded(i1 + 2 * p1, i2 + 2 * p2);
        input_padded = value_t<I>(0);

        pad_2d_input(input, input_padded, p1, p2);

        im2col_direct_tr(input_col, input_padded, k1, k2);
    } else {
        im2col_direct_tr(input_col, input, k1, k2);
    }

    if(s1 > 1 || s2 > 1){
        etl::dyn_matrix<value_t<I>, 3> tmp_result(K, c1, c2);

        etl::reshape(tmp_result, K, c1 * c2) = mul(etl::reshape(kernels, K, k1 * k2), input_col);

        // Strided copy of the large result into the small result
        for (std::size_t k = 0; k < K; ++k) {
            for (std::size_t i = 0; i < f1; ++i) {
                for (std::size_t j = 0; j < f2; ++j) {
                    conv(k, i, j) = tmp_result(k, i * s1, j * s2);
                }
            }
        }
    } else {
        etl::reshape(conv, K, f1 * f2) = mul(etl::reshape(kernels, K, k1 * k2), input_col);
    }

    conv.invalidate_gpu();
}

template <typename I_T, typename K_T, typename KS_T, typename C_T>
void blas_conv4_valid_prepared(I_T&& input, K_T&& kernel, KS_T&& kernels, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
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

    conv = value_t<I_T>(0.0);

    auto batch_fun_n = [&](const size_t first, const size_t last) {
        if (last - first) {
            SERIAL_SECTION {
                // unit-strided result dimensions
                const std::size_t sc1 = (n1 - m1 + 2 * p1) + 1;
                const std::size_t sc2 = (n2 - m2 + 2 * p2) + 1;

                etl::dyn_matrix<value_t<I_T>, 2> input_col(m1 * m2, sc1 * sc2);

                // Optimize for the most common case
                if (cpp_likely(!p1 && !p2 && s1 == 1 && s2 == 1)) {
                    for (std::size_t i = first; i < last; ++i) {
                        for (std::size_t c = 0; c < C; ++c) {
                            im2col_direct_tr(input_col, input(i)(c), m1, m2);
                            etl::reshape(conv(i), K, c1 * c2) += mul(etl::reshape(kernels(c), K, m1 * m2), input_col);
                        }
                    }
                } else {
                    etl::dyn_matrix<value_t<I_T>, 2> input_padded(n1 + 2 * p1, n2 + 2 * p2);
                    etl::dyn_matrix<value_t<I_T>, 3> tmp_result(K, sc1, sc2);

                    for (std::size_t i = first; i < last; ++i) {
                        for (std::size_t c = 0; c < C; ++c) {
                            if (p1 || p2) {
                                input_padded = value_t<I_T>(0.0);

                                pad_2d_input(input(i)(c), input_padded, p1, p2);

                                im2col_direct_tr(input_col, input_padded, m1, m2);
                            } else {
                                im2col_direct_tr(input_col, input(i)(c), m1, m2);
                            }

                            if (s1 > 1 || s2 > 1) {
                                etl::reshape(tmp_result, K, sc1 * sc2) = mul(etl::reshape(kernels(c), K, m1 * m2), input_col);

                                // Strided copy of the large result into the small result
                                for (std::size_t k = 0; k < K; ++k) {
                                    for (std::size_t ii = 0; ii < c1; ++ii) {
                                        for (std::size_t j = 0; j < c2; ++j) {
                                            conv(i, k, ii, j) += tmp_result(k, ii * s1, j * s2);
                                        }
                                    }
                                }
                            } else {
                                etl::reshape(conv(i), K, c1 * c2) += mul(etl::reshape(kernels(c), K, m1 * m2), input_col);
                            }
                        }
                    }
                }
            }
        }
    };

    dispatch_1d_any(select_parallel(N, 2), batch_fun_n, 0, N);

    conv.invalidate_gpu();
}

template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid(I_T&& input, K_T&& kernel, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto K = etl::dim<0>(kernel); // The number of kernels
    const auto C = etl::dim<1>(input);  // The number of channels

    const auto m1 = etl::dim<2>(kernel);
    const auto m2 = etl::dim<3>(kernel);

    etl::dyn_matrix<value_t<I_T>, 4> kernels(C, K, m1, m2);

    for(std::size_t c = 0; c < C; ++c){
        for(std::size_t k = 0; k < K; ++k){
            kernels(c)(k) = fflip(kernel(k)(c));
        }
    }

    blas_conv4_valid_prepared(input, kernel, kernels, conv, s1, s2, p1, p2);
}

template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_flipped(I_T&& input, K_T&& kernel, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto K = etl::dim<0>(kernel); // The number of kernels
    const auto C = etl::dim<1>(input);  // The number of channels

    const auto m1 = etl::dim<2>(kernel);
    const auto m2 = etl::dim<3>(kernel);

    etl::dyn_matrix<value_t<I_T>, 4> kernels(C, K, m1, m2);

    for(std::size_t c = 0; c < C; ++c){
        for(std::size_t k = 0; k < K; ++k){
            kernels(c)(k) = kernel(k)(c);
        }
    }

    blas_conv4_valid_prepared(input, kernel, kernels, conv, s1, s2, p1, p2);
}

template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_filter_prepared(I_T&& input, K_T&& kernel, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
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
    const std::size_t c1 = (i1 - k1 + 2 * p1) + 1;
    const std::size_t c2 = (i2 - k2 + 2 * p2) + 1;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    etl::dyn_matrix<value_t<I_T>, 4> conv_temp(C, K, f1, f2);
    conv_temp = value_t<I_T>(0);

    auto batch_fun_c = [&](const size_t first, const size_t last) {
        if (last - first) {
            SERIAL_SECTION {
                for (std::size_t c = first; c < last; ++c) {
                    etl::dyn_matrix<value_t<I_T>, 2> input_col(k1 * k2, c1 * c2);

                    for (std::size_t i = 0; i < I; ++i) {
                        // Optimize for the most common case
                        if (cpp_likely(!p1 && !p2 && s1 == 1 && s2 == 1)) {
                            im2col_direct_tr(input_col, input(i)(c), k1, k2);
                            etl::reshape(conv_temp(c), K, f1 * f2) += mul(etl::reshape(kernel(i), K, k1 * k2), input_col);
                        } else {
                            if (p1 || p2) {
                                etl::dyn_matrix<value_t<I_T>, 2> input_padded(i1 + 2 * p1, i2 + 2 * p2);
                                input_padded = value_t<I_T>(0);

                                pad_2d_input(input(i)(c), input_padded, p1, p2);

                                im2col_direct_tr(input_col, input_padded, k1, k2);
                            } else {
                                im2col_direct_tr(input_col, input(i)(c), k1, k2);
                            }

                            if (s1 > 1 || s2 > 1) {
                                etl::dyn_matrix<value_t<I_T>, 3> tmp_result(K, c1, c2);

                                etl::reshape(tmp_result, K, c1 * c2) = mul(etl::reshape(kernel(i), K, k1 * k2), input_col);

                                // Strided copy of the large result into the small result
                                for (std::size_t k = 0; k < K; ++k) {
                                    for (std::size_t ii = 0; ii < f1; ++ii) {
                                        for (std::size_t j = 0; j < f2; ++j) {
                                            conv_temp(c, k, ii, j) += tmp_result(k, ii * s1, j * s2);
                                        }
                                    }
                                }
                            } else {
                                etl::reshape(conv_temp(c), K, f1 * f2) += mul(etl::reshape(kernel(i), K, k1 * k2), input_col);
                            }
                        }
                    }
                }

                for (std::size_t c = 0; c < C; ++c) {
                    for (std::size_t k = 0; k < K; ++k) {
                        conv(k)(c) = conv_temp(c)(k);
                    }
                }
            }
        }
    };

    dispatch_1d_any(select_parallel(C, 2), batch_fun_c, 0, C);

    conv.invalidate_gpu();
}

template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_filter(I_T&& input, K_T&& kernel, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    auto prepared_k = force_temporary(kernel);

    // Flip the kernels
    prepared_k.deep_fflip_inplace();

    blas_conv4_valid_filter_prepared(input, prepared_k, conv, s1, s2, p1, p2);
}

template <typename I_T, typename K_T, typename C_T>
void blas_conv4_valid_filter_flipped(I_T&& input, K_T&& kernel, C_T&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    blas_conv4_valid_filter_prepared(input, kernel, conv, s1, s2, p1, p2);
}

} //end of namespace reduc
} //end of namespace impl
} //end of namespace etl
