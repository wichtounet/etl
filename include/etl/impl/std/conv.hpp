//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

namespace standard {

/*!
 * \brief Standard implementation of a 1D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_full(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    for (std::size_t i = first; i < last; ++i) {
        const auto lo = i >= size(kernel) - 1 ? i - (size(kernel) - 1) : 0;
        const auto hi = (i < size(input) - 1 ? i : size(input) - 1) + 1;

        typename I::value_type temp = 0.0;

        for (std::size_t j = lo; j < hi; ++j) {
            temp += input[j] * kernel[i - j];
        }

        conv[i] = temp;
    }
}

/*!
 * \brief Standard implementation of a 1D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_same(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    for (std::size_t j = first; j < last; ++j) {
        std::size_t l_lo = std::max<int>(0, j - (size(kernel) - 1) / 2);
        std::size_t l_hi = std::min<int>(size(input) - 1, j + size(kernel) / 2) + 1;

        typename I::value_type temp = 0.0;

        for (std::size_t l = l_lo; l < l_hi; ++l) {
            temp += input(l) * kernel(j - l + size(kernel) / 2);
        }

        conv[j] = temp;
    }
}

/*!
 * \brief Standard implementation of a 1D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_valid(const I& input, const K& kernel, C&& conv, std::size_t first, std::size_t last) {
    for (std::size_t j = first; j < last; ++j) {
        typename I::value_type temp = 0.0;

        for (std::size_t l = j; l < j + size(kernel); ++l) {
            temp += input[l] * kernel[j + size(kernel) - 1 - l];
        }

        conv[j] = temp;
    }
}

/*!
 * \brief Standard implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full(const I& input, const K& kernel, C&& conv) {
    for (std::size_t i = 0; i < rows(conv); ++i) {
        auto k_lo = std::max<int>(0, i - rows(kernel) + 1);
        auto k_hi = std::min(rows(input) - 1, i) + 1;

        for (std::size_t j = 0; j < columns(conv); ++j) {
            auto l_lo = std::max<int>(0, j - columns(kernel) + 1);
            auto l_hi = std::min(columns(input) - 1, j) + 1;

            typename I::value_type temp = 0.0;

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l < l_hi; ++l) {
                    temp += input(k, l) * kernel(i - k, j - l);
                }
            }

            conv(i, j) = temp;
        }
    }
}

/*!
 * \brief Standard implementation of a 2D 'same' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same(const I& input, const K& kernel, C&& conv) {
    for (std::size_t i = 0; i < rows(conv); ++i) {
        std::size_t k_lo = std::max<int>(0, i - (rows(kernel) - 1) / 2);
        std::size_t k_hi = std::min<int>(rows(input) - 1, i + rows(kernel) / 2) + 1;

        for (std::size_t j = 0; j < columns(conv); ++j) {
            std::size_t l_lo = std::max<int>(0, j - (columns(kernel) - 1) / 2);
            std::size_t l_hi = std::min<int>(columns(input) - 1, j + columns(kernel) / 2) + 1;

            typename I::value_type temp = 0.0;

            for (std::size_t k = k_lo; k < k_hi; ++k) {
                for (std::size_t l = l_lo; l < l_hi; ++l) {
                    temp += input(k, l) * kernel(i - k + rows(kernel) / 2, j - l + columns(kernel) / 2);
                }
            }

            conv(i, j) = temp;
        }
    }
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid(const I& input, const K& kernel, C&& conv) {
    for (std::size_t i = 0; i < rows(conv); ++i) {
        for (std::size_t j = 0; j < columns(conv); ++j) {
            typename I::value_type temp = 0.0;

            for (std::size_t k = i; k < i + rows(kernel); ++k) {
                for (std::size_t l = j; l < j + columns(kernel); ++l) {
                    temp += input(k, l) * kernel((i + rows(kernel) - 1 - k), (j + columns(kernel) - 1 - l));
                }
            }

            conv(i, j) = temp;
        }
    }
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi(const I& input, const K& kernels, C&& conv) {
    for (size_t k = 0; k < etl::dim<0>(kernels); ++k) {
        conv(k) = conv_2d_valid(input, kernels(k));
    }
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid_multi_flipped(const I& input, const K& kernels, C&& conv) {
    for (size_t k = 0; k < etl::dim<0>(kernels); ++k) {
        conv(k) = conv_2d_valid(input, fflip(kernels(k)));
    }
}

/*!
 * \brief Standard implementation of multiple 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv3_valid_multi(const I& input, const K& kernels, C&& conv) {
    for (size_t k = 0; k < etl::dim<0>(kernels); ++k) {
        conv(k) = conv_2d_valid_multi(input(k), kernels(k));
    }
}

/*!
 * \brief Standard implementation of multiple 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv3_valid_multi_flipped(const I& input, const K& kernels, C&& conv) {
    for (size_t k = 0; k < etl::dim<0>(kernels); ++k) {
        conv(k) = conv_2d_valid_multi_flipped(input(k), kernels(k));
    }
}

/*!
 * \brief Standard implementation of a 2D 'full' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_multi(const I& input, const K& kernels, C&& conv) {
    for (size_t k = 0; k < etl::dim<0>(kernels); ++k) {
        conv(k) = conv_2d_full(input, kernels(k));
    }
}

/*!
 * \brief Standard implementation of a 2D 'same' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_multi(const I& input, const K& kernels, C&& conv) {
    for (size_t k = 0; k < etl::dim<0>(kernels); ++k) {
        conv(k) = conv_2d_same(input, kernels(k));
    }
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
