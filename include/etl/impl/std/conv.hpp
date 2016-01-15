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

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
