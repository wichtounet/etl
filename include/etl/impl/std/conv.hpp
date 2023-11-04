//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Standard implementation of a 1D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param first The index where to start in the output matrix
 * \param last The index where to stop in the output matrix
 */
template <typename I, typename K, typename C>
void conv1_full(const I& input, const K& kernel, C&& conv, size_t first, size_t last) {
    for (size_t i = first; i < last; ++i) {
        const auto lo = i >= etl::size(kernel) - 1 ? i - (etl::size(kernel) - 1) : 0;
        const auto hi = (i < etl::size(input) - 1 ? i : etl::size(input) - 1) + 1;

        value_t<I> temp = 0.0;

        for (size_t j = lo; j < hi; ++j) {
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
void conv1_same(const I& input, const K& kernel, C&& conv, size_t first, size_t last) {
    for (size_t j = first; j < last; ++j) {
        size_t l_lo(std::max<int>(0, j - (etl::size(kernel) - 1) / 2));
        size_t l_hi(std::min<int>(etl::size(input) - 1, j + etl::size(kernel) / 2) + 1);

        value_t<I> temp = 0.0;

        for (size_t l = l_lo; l < l_hi; ++l) {
            temp += input(l) * kernel(j - l + etl::size(kernel) / 2);
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
void conv1_valid(const I& input, const K& kernel, C&& conv, size_t first, size_t last) {
    for (size_t j = first; j < last; ++j) {
        value_t<I> temp = 0.0;

        for (size_t l = j; l < j + etl::size(kernel); ++l) {
            temp += input[l] * kernel[j + etl::size(kernel) - 1 - l];
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
    for (size_t i = 0; i < rows(conv); ++i) {
        auto k_lo = std::max<int>(0, i - rows(kernel) + 1);
        auto k_hi = std::min(rows(input) - 1, i) + 1;

        for (size_t j = 0; j < columns(conv); ++j) {
            auto l_lo = std::max<int>(0, j - columns(kernel) + 1);
            auto l_hi = std::min(columns(input) - 1, j) + 1;

            value_t<I> temp = 0.0;

            for (size_t k = k_lo; k < k_hi; ++k) {
                for (size_t l = l_lo; l < l_hi; ++l) {
                    temp += input(k, l) * kernel(i - k, j - l);
                }
            }

            conv(i, j) = temp;
        }
    }
}

/*!
 * \brief Standard implementation of a 2D 'full' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_full_flipped(const I& input, const K& kernel, C&& conv) {
    for (size_t i = 0; i < rows(conv); ++i) {
        auto k_lo = std::max<int>(0, i - rows(kernel) + 1);
        auto k_hi = std::min(rows(input) - 1, i) + 1;

        for (size_t j = 0; j < columns(conv); ++j) {
            auto l_lo = std::max<int>(0, j - columns(kernel) + 1);
            auto l_hi = std::min(columns(input) - 1, j) + 1;

            value_t<I> temp = 0.0;

            for (size_t k = k_lo; k < k_hi; ++k) {
                for (size_t l = l_lo; l < l_hi; ++l) {
                    temp += input(k, l) * kernel(rows(kernel) - 1 - (i - k), columns(kernel) - 1 - (j - l));
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
    for (size_t i = 0; i < rows(conv); ++i) {
        size_t k_lo = std::max<int>(0, i - (rows(kernel) - 1) / 2);
        size_t k_hi = std::min<int>(rows(input) - 1, i + rows(kernel) / 2) + 1;

        for (size_t j = 0; j < columns(conv); ++j) {
            size_t l_lo = std::max<int>(0, j - (columns(kernel) - 1) / 2);
            size_t l_hi = std::min<int>(columns(input) - 1, j + columns(kernel) / 2) + 1;

            value_t<I> temp = 0.0;

            for (size_t k = k_lo; k < k_hi; ++k) {
                for (size_t l = l_lo; l < l_hi; ++l) {
                    temp += input(k, l) * kernel(i - k + rows(kernel) / 2, j - l + columns(kernel) / 2);
                }
            }

            conv(i, j) = temp;
        }
    }
}

/*!
 * \brief Standard implementation of a 2D 'same' convolution C = I * K, with
 * flipped kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_same_flipped(const I& input, const K& kernel, C&& conv) {
    for (size_t i = 0; i < rows(conv); ++i) {
        size_t k_lo = std::max<int>(0, i - (rows(kernel) - 1) / 2);
        size_t k_hi = std::min<int>(rows(input) - 1, i + rows(kernel) / 2) + 1;

        for (size_t j = 0; j < columns(conv); ++j) {
            size_t l_lo = std::max<int>(0, j - (columns(kernel) - 1) / 2);
            size_t l_hi = std::min<int>(columns(input) - 1, j + columns(kernel) / 2) + 1;

            value_t<I> temp = 0.0;

            for (size_t k = k_lo; k < k_hi; ++k) {
                for (size_t l = l_lo; l < l_hi; ++l) {
                    temp += input(k, l) * kernel(rows(kernel) - 1 - (i - k + rows(kernel) / 2), columns(kernel) - 1 - (j - l + columns(kernel) / 2));
                }
            }

            conv(i, j) = temp;
        }
    }
}

/*!
 * \brief Compute the value of a border pixel for a 2d valid
 * convolution. This is only used with padding.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param i The first index of the pixel to compute
 * \param j The second index of the pixel to compute
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding
 * \param p2 The second dimension padding
 */
template <typename I, typename K, typename C>
inline void conv2_valid_border(
    const I& input, const K& kernel, C&& conv, size_t i, size_t j, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta = value_t<I>(0.0)) {
    value_t<I> temp = 0.0;

    const auto s_i = i * s1;
    const auto s_j = j * s2;

    for (size_t k = 0; k < rows(kernel); ++k) {
        for (size_t l = 0; l < columns(kernel); ++l) {
            if (s_i + k >= p1 && (s_i + k) - p1 < rows(input) && s_j + l >= p2 && (s_j + l) - p2 < columns(input)) {
                const size_t i_i = (s_i + k) - p1;
                const size_t i_j = (s_j + l) - p2;

                temp += input(i_i, i_j) * kernel(rows(kernel) - 1 - k, columns(kernel) - 1 - l);
            }
        }
    }

    if (beta == value_t<I>(0.0)) {
        conv(i, j) = temp;
    } else {
        conv(i, j) = beta * conv(i, j) + temp;
    }
}

/*!
 * \brief Compute the value of a border pixel for a 2d valid
 * convolution, with flipped kernels. This is only used with padding.
 *
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 * \param i The first index of the pixel to compute
 * \param j The second index of the pixel to compute
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 * \param p1 The first dimension padding
 * \param p2 The second dimension padding
 */
template <typename I, typename K, typename C>
inline void conv2_valid_flipped_border(
    const I& input, const K& kernel, C&& conv, size_t i, size_t j, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta = value_t<I>(0.0)) {
    value_t<I> temp = 0.0;

    const auto s_i = i * s1;
    const auto s_j = j * s2;

    for (size_t k = 0; k < rows(kernel); ++k) {
        for (size_t l = 0; l < columns(kernel); ++l) {
            if (s_i + k >= p1 && (s_i + k) - p1 < rows(input) && s_j + l >= p2 && (s_j + l) - p2 < columns(input)) {
                const size_t i_i = (s_i + k) - p1;
                const size_t i_j = (s_j + l) - p2;

                temp += input(i_i, i_j) * kernel(k, l);
            }
        }
    }

    if (beta == value_t<I>(0.0)) {
        conv(i, j) = temp;
    } else {
        conv(i, j) = beta * conv(i, j) + temp;
    }
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv2_valid(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta = value_t<I>(0.0)) {
    // Do the outer parts of the convolution

    if (p1 || p2) {
        for (size_t i = 0; i < p1; ++i) {
            for (size_t j = 0; j < columns(conv); ++j) {
                conv2_valid_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t i = rows(conv) - p1; i < rows(conv); ++i) {
            for (size_t j = 0; j < columns(conv); ++j) {
                conv2_valid_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t j = 0; j < p2; ++j) {
            for (size_t i = p1; i < rows(conv) - p1; ++i) {
                conv2_valid_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t j = columns(conv) - p2; j < columns(conv); ++j) {
            for (size_t i = p1; i < rows(conv) - p1; ++i) {
                conv2_valid_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }
    }

    // Do the central part of the valid convolution (no padding)

    for (size_t i = p1; i < rows(conv) - p1; ++i) {
        for (size_t j = p2; j < columns(conv) - p2; ++j) {
            const auto i_i = i * s1 - p1;
            const auto i_j = j * s2 - p2;

            value_t<I> temp = 0.0;

            for (size_t k = 0; k < rows(kernel); ++k) {
                for (size_t l = 0; l < columns(kernel); ++l) {
                    temp += input(i_i + k, i_j + l) * kernel(rows(kernel) - 1 - k, columns(kernel) - 1 - l);
                }
            }

            if (beta == value_t<I>(0.0)) {
                conv(i, j) = temp;
            } else {
                conv(i, j) = beta * conv(i, j) + temp;
            }
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
void conv2_valid_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta = value_t<I>(0.0)) {
    // Do the outer parts of the convolution

    if (p1 || p2) {
        for (size_t i = 0; i < p1; ++i) {
            for (size_t j = 0; j < columns(conv); ++j) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t i = rows(conv) - p1; i < rows(conv); ++i) {
            for (size_t j = 0; j < columns(conv); ++j) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t j = 0; j < p2; ++j) {
            for (size_t i = p1; i < rows(conv) - p1; ++i) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t j = columns(conv) - p2; j < columns(conv); ++j) {
            for (size_t i = p1; i < rows(conv) - p1; ++i) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }
    }

    // Do the central part of the valid convolution (no padding)

    for (size_t i = p1; i < rows(conv) - p1; ++i) {
        for (size_t j = p2; j < columns(conv) - p2; ++j) {
            const auto i_i = i * s1 - p1;
            const auto i_j = j * s2 - p2;

            value_t<I> temp = 0.0;

            for (size_t k = 0; k < rows(kernel); ++k) {
                for (size_t l = 0; l < columns(kernel); ++l) {
                    temp += input(i_i + k, i_j + l) * kernel(k, l);
                }
            }

            if (beta == value_t<I>(0.0)) {
                conv(i, j) = temp;
            } else {
                conv(i, j) = beta * conv(i, j) + temp;
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_valid(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(etl::dim<1>(input) == etl::dim<1>(kernel), "Invalid number of channels");
    cpp_assert(etl::dim<0>(input) == etl::dim<0>(conv), "Invalid number of images");

    if (etl::dim<1>(kernel) > 0) {
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
            for (size_t k = 0; k < etl::dim<0>(kernel); ++k) {
                conv2_valid(input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, p1, p2, 0.0);

                for (size_t c = 1; c < etl::dim<1>(kernel); ++c) {
                    conv2_valid(input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, p1, p2, 1.0);
                }
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_valid_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(etl::dim<1>(input) == etl::dim<1>(kernel), "Invalid number of channels");
    cpp_assert(etl::dim<0>(input) == etl::dim<0>(conv), "Invalid number of images");

    if (etl::dim<1>(kernel) > 0) {
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
            for (size_t k = 0; k < etl::dim<0>(kernel); ++k) {
                conv2_valid_flipped(input(i)(0), kernel(k)(0), conv(i)(k), s1, s2, p1, p2, 0.0);

                for (size_t c = 1; c < etl::dim<1>(kernel); ++c) {
                    conv2_valid_flipped(input(i)(c), kernel(k)(c), conv(i)(k), s1, s2, p1, p2, 1.0);
                }
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_valid_back(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(etl::dim<1>(input) == etl::dim<0>(kernel), "Invalid dimensions for std::conv4_valid_back");
    cpp_assert(etl::dim<1>(kernel) == etl::dim<1>(conv), "Invalid dimensions for std::conv4_valid_back");
    cpp_assert(etl::dim<0>(input) == etl::dim<0>(conv), "Invalid dimensions for std::conv4_valid_back");

    if (etl::dim<0>(kernel) > 0) {
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
            for (size_t c = 0; c < etl::dim<1>(kernel); ++c) {
                conv2_valid(input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, p1, p2, 0.0);

                for (size_t k = 1; k < etl::dim<0>(kernel); ++k) {
                    conv2_valid(input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, p1, p2, 1.0);
                }
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_valid_back_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(etl::dim<1>(input) == etl::dim<0>(kernel), "Invalid dimensions for std::conv4_valid_back_flipped");
    cpp_assert(etl::dim<1>(kernel) == etl::dim<1>(conv), "Invalid dimensions for std::conv4_valid_back_flipped");
    cpp_assert(etl::dim<0>(input) == etl::dim<0>(conv), "Invalid dimensions for std::conv4_valid_back_flipped");

    if (etl::dim<0>(kernel) > 0) {
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
            for (size_t c = 0; c < etl::dim<1>(kernel); ++c) {
                conv2_valid_flipped(input(i)(0), kernel(0)(c), conv(i)(c), s1, s2, p1, p2, 0.0);

                for (size_t k = 1; k < etl::dim<0>(kernel); ++k) {
                    conv2_valid_flipped(input(i)(k), kernel(k)(c), conv(i)(c), s1, s2, p1, p2, 1.0);
                }
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 4D 'valid' convolution C = I * K, where the output are considered to be kernels
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_valid_filter(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(etl::dim<0>(input) == etl::dim<0>(kernel), "Invalid number of channels");
    cpp_assert(etl::dim<1>(input) == etl::dim<1>(conv), "Invalid number of images");
    cpp_assert(etl::dim<1>(kernel) == etl::dim<0>(conv), "Invalid number of images");

    if (etl::dim<0>(input) > 0) {
        //i = 0
        for (size_t k = 0; k < etl::dim<1>(kernel); ++k) {
            for (size_t c = 0; c < etl::dim<1>(input); ++c) {
                conv2_valid(input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, p1, p2, 0.0);
            }
        }

        for (size_t i = 1; i < etl::dim<0>(input); ++i) {
            for (size_t k = 0; k < etl::dim<1>(kernel); ++k) {
                for (size_t c = 0; c < etl::dim<1>(input); ++c) {
                    conv2_valid(input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, p1, p2, 1.0);
                }
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_valid_filter_flipped(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    cpp_assert(etl::dim<0>(input) == etl::dim<0>(kernel), "Invalid number of channels");
    cpp_assert(etl::dim<1>(input) == etl::dim<1>(conv), "Invalid number of images");
    cpp_assert(etl::dim<1>(kernel) == etl::dim<0>(conv), "Invalid number of images");

    if (etl::dim<0>(input) > 0) {
        //i = 0
        for (size_t k = 0; k < etl::dim<1>(kernel); ++k) {
            for (size_t c = 0; c < etl::dim<1>(input); ++c) {
                conv2_valid_flipped(input(0)(c), kernel(0)(k), conv(k)(c), s1, s2, p1, p2, 0.0);
            }
        }

        for (size_t i = 1; i < etl::dim<0>(input); ++i) {
            for (size_t k = 0; k < etl::dim<1>(kernel); ++k) {
                for (size_t c = 0; c < etl::dim<1>(input); ++c) {
                    conv2_valid_flipped(input(i)(c), kernel(i)(k), conv(k)(c), s1, s2, p1, p2, 1.0);
                }
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_full(const I& input, const K& kernel, C&& conv) {
    cpp_assert(etl::dim<1>(input) == etl::dim<0>(kernel), "Invalid number of channels");
    cpp_assert(etl::dim<0>(input) == etl::dim<0>(conv), "Invalid number of images");

    conv = 0.0;

    for (size_t ii = 0; ii < etl::dim<0>(input); ++ii) {
        for (size_t kk = 0; kk < etl::dim<0>(kernel); ++kk) {
            for (size_t cc = 0; cc < etl::dim<1>(kernel); ++cc) {
                for (size_t i = 0; i < etl::dim<2>(conv); ++i) {
                    auto k_lo = std::max<int>(0, i - etl::dim<2>(kernel) + 1);
                    auto k_hi = std::min(etl::dim<2>(input) - 1, i) + 1;

                    for (size_t j = 0; j < etl::dim<3>(conv); ++j) {
                        auto l_lo = std::max<int>(0, j - etl::dim<3>(kernel) + 1);
                        auto l_hi = std::min(etl::dim<3>(input) - 1, j) + 1;

                        value_t<I> temp = 0.0;

                        for (size_t k = k_lo; k < k_hi; ++k) {
                            for (size_t l = l_lo; l < l_hi; ++l) {
                                temp += input(ii, kk, k, l) * kernel(kk, cc, i - k, j - l);
                            }
                        }

                        conv(ii, cc, i, j) += temp;
                    }
                }
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 4D 'valid' convolution C = I * K
 * \param input The input matrix
 * \param kernel The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K, typename C>
void conv4_full_flipped(const I& input, const K& kernel, C&& conv) {
    cpp_assert(etl::dim<1>(input) == etl::dim<0>(kernel), "Invalid number of channels");
    cpp_assert(etl::dim<0>(input) == etl::dim<0>(conv), "Invalid number of images");

    conv = 0.0;

    for (size_t ii = 0; ii < etl::dim<0>(input); ++ii) {
        for (size_t kk = 0; kk < etl::dim<0>(kernel); ++kk) {
            for (size_t cc = 0; cc < etl::dim<1>(kernel); ++cc) {
                for (size_t i = 0; i < etl::dim<2>(conv); ++i) {
                    auto k_lo = std::max<int>(0, i - etl::dim<2>(kernel) + 1);
                    auto k_hi = std::min(etl::dim<2>(input) - 1, i) + 1;

                    for (size_t j = 0; j < etl::dim<3>(conv); ++j) {
                        auto l_lo = std::max<int>(0, j - etl::dim<3>(kernel) + 1);
                        auto l_hi = std::min(etl::dim<3>(input) - 1, j) + 1;

                        value_t<I> temp = 0.0;

                        for (size_t k = k_lo; k < k_hi; ++k) {
                            for (size_t l = l_lo; l < l_hi; ++l) {
                                temp += input(ii, kk, k, l) * kernel(kk, cc, etl::dim<2>(kernel) - 1 - (i - k), etl::dim<3>(kernel) - 1 - (j - l));
                            }
                        }

                        conv(ii, cc, i, j) += temp;
                    }
                }
            }
        }
    }
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void conv2_valid_multi(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto K = etl::dim<0>(kernels);

    auto fun_k = [&](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_valid(input, kernels(k), conv(k), s1, s2, p1, p2);
        }
    };

    engine_dispatch_1d(fun_k, 0, K, 2UL);
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void conv2_valid_multi_flipped(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto K = etl::dim<0>(kernels);

    auto fun_k = [&](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_valid_flipped(input, kernels(k), conv(k), s1, s2, p1, p2);
        }
    };

    engine_dispatch_1d(fun_k, 0, K, 2UL);
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple images and multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void conv2_valid_multi_multi(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto K  = etl::dim<0>(kernels);
    const auto N  = etl::dim<0>(input);
    const auto KN = K * N;

    auto fun_kn = [&](const size_t first, const size_t last) {
        for (size_t kn = first; kn < last; ++kn) {
            auto k = kn / N;
            auto n = kn % N;

            conv2_valid(input(n), kernels(k), conv(k)(n), s1, s2, p1, p2);
        }
    };

    engine_dispatch_1d(fun_kn, 0, KN, 2UL);
}

/*!
 * \brief Standard implementation of a 2D 'valid' convolution C = I * K, with multiple images and multiple flipped kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void conv2_valid_multi_multi_flipped(const I& input, const K_T& kernels, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
    const auto K  = etl::dim<0>(kernels);
    const auto N  = etl::dim<0>(input);
    const auto KN = K * N;

    auto fun_kn = [&](const size_t first, const size_t last) {
        for (size_t kn = first; kn < last; ++kn) {
            auto k = kn / N;
            auto n = kn % N;

            conv2_valid_flipped(input(n), kernels(k), conv(k)(n), s1, s2, p1, p2);
        }
    };

    engine_dispatch_1d(fun_kn, 0, KN, 2UL);
}

/*!
 * \brief Standard implementation of a 2D 'full' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void conv2_full_multi(const I& input, const K_T& kernels, C&& conv) {
    const auto K = etl::dim<0>(kernels);

    auto fun_k = [&](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_full(input, kernels(k), conv(k));
        }
    };

    engine_dispatch_1d(fun_k, 0, K, 2UL);
}

/*!
 * \brief Standard implementation of a 2D 'full' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void conv2_full_multi_flipped(const I& input, const K_T& kernels, C&& conv) {
    const auto K = etl::dim<0>(kernels);

    auto fun_k = [&](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_full_flipped(input, kernels(k), conv(k));
        }
    };

    engine_dispatch_1d(fun_k, 0, K, 2UL);
}

/*!
 * \brief Standard implementation of a 2D 'same' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void conv2_same_multi(const I& input, const K_T& kernels, C&& conv) {
    const auto K = etl::dim<0>(kernels);

    auto fun_k = [&](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_same(input, kernels(k), conv(k));
        }
    };

    engine_dispatch_1d(fun_k, 0, K, 2UL);
}

/*!
 * \brief Standard implementation of a 2D 'same' convolution C = I * K, with multiple kernels
 * \param input The input matrix
 * \param kernels The kernel matrix
 * \param conv The output matrix
 */
template <typename I, typename K_T, typename C>
void conv2_same_multi_flipped(const I& input, const K_T& kernels, C&& conv) {
    const auto K = etl::dim<0>(kernels);

    auto fun_k = [&](const size_t first, const size_t last) {
        for (size_t k = first; k < last; ++k) {
            conv2_same_flipped(input, kernels(k), conv(k));
        }
    };

    engine_dispatch_1d(fun_k, 0, K, 2UL);
}

} //end of namespace etl::impl::standard
