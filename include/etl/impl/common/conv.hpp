//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

namespace common {

/*!
 * \brief Compute the left part of the kernel for a same convolution
 * \param in Pointer to the memory of the input
 * \param n The size of the input
 * \param kernel Pointer to the memory of the kernel
 * \param m The size of the kernel
 * \param out Pointer to the output
 * \param first The beginning of the range of the input to consider
 * \param last The end of the range of the input to consider
 */
template <typename T>
void left_same_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out, std::size_t first, std::size_t last) {
    cpp_unused(n);

    std::size_t left  = (m - 1) / 2;
    std::size_t right = m / 2;

    //Left invalid part
    for (std::size_t j = first; j < std::min(last, left); ++j) {
        T temp = 0.0;

        for (std::size_t l = 0; l <= j + right; ++l) {
            temp += in[l] * kernel[j - l + right];
        }

        out[j] = temp;
    }
}

/*!
 * \brief Compute the right part of the kernel for a same convolution
 * \param in Pointer to the memory of the input
 * \param n The size of the input
 * \param kernel Pointer to the memory of the kernel
 * \param m The size of the kernel
 * \param out Pointer to the output
 * \param first The beginning of the range of the input to consider
 * \param last The end of the range of the input to consider
 */
template <typename T>
void right_same_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out, std::size_t first, std::size_t last) {
    std::size_t left  = (m - 1) / 2;
    std::size_t right = m / 2;

    //Right invalid part
    for (std::size_t j = std::max(first, n - right); j < std::min(last, n); ++j) {
        T temp = 0.0;

        std::size_t hi = std::min<int>(n - 1, j + right);
        for (std::size_t l = j - left; l <= hi; ++l) {
            temp += in[l] * kernel[j - l + m / 2];
        }

        out[j] = temp;
    }
}

/*!
 * \brief Compute the left part of the kernel for a full convolution
 * \param in Pointer to the memory of the input
 * \param n The size of the input
 * \param kernel Pointer to the memory of the kernel
 * \param m The size of the kernel
 * \param out Pointer to the output
 * \param first The beginning of the range of the input to consider
 * \param last The end of the range of the input to consider
 */
template <typename T>
void left_full_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out, std::size_t first, std::size_t last) {
    std::size_t left = m - 1;

    //Left invalid part
    for (std::size_t i = first; i < std::min(last, left); ++i) {
        const auto hi = i < n - 1 ? i : n - 1;

        T temp = 0.0;

        for (std::size_t j = 0; j <= hi; ++j) {
            temp += in[j] * kernel[i - j];
        }

        out[i] = temp;
    }
}

/*!
 * \brief Compute the right part of the kernel for a full convolution
 * \param in Pointer to the memory of the input
 * \param n The size of the input
 * \param kernel Pointer to the memory of the kernel
 * \param m The size of the kernel
 * \param out Pointer to the output
 * \param first The beginning of the range of the input to consider
 * \param last The end of the range of the input to consider
 */
template <typename T>
void right_full_kernel(const T* in, const std::size_t n, const T* kernel, std::size_t m, T* out, std::size_t first, std::size_t last) {
    std::size_t right = m - 1;

    auto c = n + m - 1;

    //Right invalid part
    for (std::size_t i = std::max(first, c - right); i < std::min(last, c); ++i) {
        const auto lo = i >= m - 1 ? i - (m - 1) : 0;
        const auto hi = i < n - 1 ? i : n - 1;

        T temp = 0.0;

        for (std::size_t j = lo; j <= hi; ++j) {
            temp += in[j] * kernel[i - j];
        }

        out[i] = temp;
    }
}

} //end of namespace common
} //end of namespace impl
} //end of namespace etl
