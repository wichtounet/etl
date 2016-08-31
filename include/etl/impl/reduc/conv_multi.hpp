//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
    for (std::size_t outer1 = 0; outer1 < etl::dim<0>(in); ++outer1) {
        for (std::size_t outer2 = 0; outer2 < etl::dim<1>(in); ++outer2) {
            auto* direct = out(outer1)(outer2).memory_start();
            for (std::size_t i = 0; i < etl::dim<2>(in); ++i) {
                for (std::size_t j = 0; j < etl::dim<3>(in); ++j) {
                    direct[i * out.template dim<3>() + j] = in(outer1, outer2, i, j);
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
    for (std::size_t outer = 0; outer < etl::dim<0>(in); ++outer) {
        auto* direct = out(outer).memory_start();
        for (std::size_t i = 0; i < etl::dim<1>(in); ++i) {
            for (std::size_t j = 0; j < etl::dim<2>(in); ++j) {
                direct[i * out.template dim<2>() + j] = in(outer, i, j);
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
void complex_pad_2d(const F1& in, F2& out) {
    auto* direct = out.memory_start();
    for (std::size_t i = 0; i < etl::dim<0>(in); ++i) {
        for (std::size_t j = 0; j < etl::dim<1>(in); ++j) {
            direct[i * out.template dim<1>() + j] = in(i, j);
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
    cpp_unused(p1);
    cpp_unused(p2);

    const std::size_t K = etl::dim<0>(kernels);

    const std::size_t i1 = etl::dim<0>(input);
    const std::size_t i2 = etl::dim<1>(input);

    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);

    const std::size_t c1 = (i1 - k1) / s1 + 1;
    const std::size_t c2 = (i2 - k2) / s1 + 1;
    const std::size_t v1 = i1 - k1 + 1;
    const std::size_t v2 = i2 - k2 + 1;
    const std::size_t t1 = i1 + k1 - 1;
    const std::size_t t2 = i2 + k2 - 1;
    const std::size_t b1 = (t1 - v1) / 2;
    const std::size_t b2 = (t2 - v2) / 2;

    etl::dyn_matrix<std::complex<value_t<I>>> input_padded(t1, t2);
    etl::dyn_matrix<std::complex<value_t<I>>, 3> kernels_padded(K, t1, t2);
    etl::dyn_matrix<std::complex<value_t<I>>, 3> tmp_result(K, t1, t2);

    complex_pad_2d(input, input_padded);
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
                conv(k, i, j) = tmp_result(k, i * s1 + b1, j * s2 + b2).real();
            }
        }
    }
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi(const I& input, const K_T& kernels, C&& conv) {
    const std::size_t K  = etl::dim<0>(kernels);
    const std::size_t v1 = etl::dim<0>(input);
    const std::size_t v2 = etl::dim<1>(input);
    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);
    const std::size_t f1 = etl::dim<1>(conv);
    const std::size_t f2 = etl::dim<2>(conv);

    auto prepared_k = force_temporary(kernels);

    prepared_k.deep_fflip_inplace();

    etl::dyn_matrix<value_t<I>, 2> input_col(k1 * k2, (v1 - k1 + 1) * (v2 - k2 + 1));
    im2col_direct_tr(input_col, input, k1, k2);

    etl::reshape(conv, K, f1 * f2) = mul(etl::reshape(prepared_k, K, k1 * k2), input_col);
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void fft_conv2_valid_multi_flipped(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    auto kernels_f = etl::force_temporary(kernels);

    kernels_f.deep_fflip_inplace();

    fft_conv2_valid_multi(input, kernels_f, conv, s1, s2, p1, p2);
}

/*!
 * \brief BLAS implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void blas_conv2_valid_multi_flipped(const I& input, const K_T& kernels, C&& conv) {
    const std::size_t K  = etl::dim<0>(kernels);
    const std::size_t v1 = etl::dim<0>(input);
    const std::size_t v2 = etl::dim<1>(input);
    const std::size_t f1 = etl::dim<1>(conv);
    const std::size_t f2 = etl::dim<2>(conv);
    const std::size_t k1 = etl::dim<1>(kernels);
    const std::size_t k2 = etl::dim<2>(kernels);

    etl::dyn_matrix<value_t<I>, 2> input_col(k1 * k2, (v1 - k1 + 1) * (v2 - k2 + 1));
    im2col_direct_tr(input_col, input, k1, k2);

    etl::reshape(conv, K, f1 * f2) = mul(etl::reshape(kernels, K, k1 * k2), input_col);
}

} //end of namespace reduc
} //end of namespace impl
} //end of namespace etl
