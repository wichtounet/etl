//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementation of common convolution functions
 *
 * Note: the padding functions should use sub views instead of computing the
 * indices directly, but this causes sever performance degradations
 */

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

template <typename I>
etl::dyn_matrix<value_t<I>, 2> pad_right(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 2> padded_input(etl::dim<0>(input), etl::dim<1>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        direct_copy_n(input.memory_start() + i * etl::dim<1>(input), padded_input.memory_start() + i * etl::dim<1>(padded_input), etl::dim<1>(input));
    }

    return padded_input;
}

template <typename I>
etl::dyn_matrix<value_t<I>, 2> pad_right_flip(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 2> flipped(etl::dim<0>(input), etl::dim<1>(input));

    std::reverse_copy(input.begin(), input.end(), flipped.begin());

    etl::dyn_matrix<T, 2> padded_input(etl::dim<0>(input), etl::dim<1>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(flipped); ++i){
        direct_copy_n(flipped.memory_start() + i * flipped.dim(1), padded_input.memory_start() + i * etl::dim<1>(padded_input), flipped.dim(1));
    }

    return padded_input;
}

template <typename I, cpp_enable_if((decay_traits<I>::dimensions() == 3))>
etl::dyn_matrix<value_t<I>, 3> pad_right_multi(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 3> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            direct_copy_n(
                input.memory_start() + i * etl::dim<1>(input) * etl::dim<2>(input) + j * etl::dim<2>(input),
                padded_input.memory_start() + i * etl::dim<1>(padded_input) * etl::dim<2>(padded_input) + j * etl::dim<2>(padded_input),
                etl::dim<2>(input)); }
    }

    return padded_input;
}

template <typename I, cpp_enable_if((decay_traits<I>::dimensions() == 4))>
etl::dyn_matrix<value_t<I>, 4> pad_right_multi(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 4> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input) + pad);

    padded_input = 0;

    const auto C1 = etl::dim<1>(input) * etl::dim<2>(input) * etl::dim<3>(input);
    const auto C2 = etl::dim<2>(input) * etl::dim<3>(input);
    const auto C3 = etl::dim<3>(input);

    const auto PC1 = etl::dim<1>(input) * etl::dim<2>(input) * (etl::dim<3>(input) + pad);
    const auto PC2 = etl::dim<2>(input) * (etl::dim<3>(input) + pad);
    const auto PC3 = (etl::dim<3>(input) + pad);

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            for(size_t k = 0; k < etl::dim<2>(input); ++k){
                direct_copy_n(
                    input.memory_start() + i * C1 + j * C2 + k * C3,
                    padded_input.memory_start() + i * PC1 + j * PC2 + k * PC3,
                    etl::dim<3>(input));
            }
        }
    }

    return padded_input;
}

template <typename I, cpp_enable_if((decay_traits<I>::dimensions() == 3))>
etl::dyn_matrix<value_t<I>, 3> pad_right_flip_multi(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 3> flipped(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input));

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        std::reverse_copy(
            input.memory_start() + i * etl::dim<1>(input) * etl::dim<2>(input),
            input.memory_start() + (i+1) * etl::dim<1>(input) * etl::dim<2>(input),
            flipped.memory_start() + i * etl::dim<1>(input) * etl::dim<2>(input));
    }

    etl::dyn_matrix<T, 3> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            direct_copy_n(
                flipped.memory_start() + i * flipped.dim(1) * flipped.dim(2) + j * flipped.dim(2),
                padded_input.memory_start() + i * etl::dim<1>(padded_input) * etl::dim<2>(padded_input) + j * etl::dim<2>(padded_input),
                etl::dim<2>(input));
        }
    }

    return padded_input;
}

template <typename I, cpp_enable_if((decay_traits<I>::dimensions() == 4))>
etl::dyn_matrix<value_t<I>, 4> pad_right_flip_multi(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 4> flipped(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input));

    const auto C1 = etl::dim<1>(input) * etl::dim<2>(input) * etl::dim<3>(input);
    const auto C2 = etl::dim<2>(input) * etl::dim<3>(input);
    const auto C3 = etl::dim<3>(input);

    const auto PC1 = etl::dim<1>(input) * etl::dim<2>(input) * (etl::dim<3>(input) + pad);
    const auto PC2 = etl::dim<2>(input) * (etl::dim<3>(input) + pad);
    const auto PC3 = (etl::dim<3>(input) + pad);

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            std::reverse_copy(
                input.memory_start() + i * C1 + j * C2,
                input.memory_start() + i * C1 + (j+1) * C2,
                flipped.memory_start() + i * C1 + j * C2);
        }
    }

    etl::dyn_matrix<T, 4> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            for(size_t k = 0; k < etl::dim<2>(input); ++k){
                direct_copy_n(
                    flipped.memory_start() + i * C1 + j * C2 + k * C3,
                    padded_input.memory_start() + i * PC1 + j * PC2 + k * PC3,
                    etl::dim<3>(input));
            }
        }
    }

    return padded_input;
}

} //end of namespace common
} //end of namespace impl
} //end of namespace etl
