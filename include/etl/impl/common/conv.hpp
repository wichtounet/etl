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

template <typename T>
etl::dyn_matrix<T, 2> pad_right(const opaque_memory<T, 2>& input, size_t pad){
    etl::dyn_matrix<T, 2> padded_input(input.dim(0), input.dim(1) + pad);

    padded_input = 0;

    for(size_t i = 0; i < input.dim(0); ++i){
        direct_copy_n(input.memory_start() + i * input.dim(1), padded_input.memory_start() + i * padded_input.dim(1), input.dim(1));
    }

    return padded_input;
}

template <typename I>
etl::dyn_matrix<value_t<I>, 2> pad_right_general(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 2> padded_input(etl::dim<0>(input), etl::dim<1>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        direct_copy_n(input(i).memory_start(), padded_input(i).memory_start(), etl::dim<1>(input));
    }

    return padded_input;
}

template <typename I>
etl::dyn_matrix<value_t<I>, 2> pad_right_flip_general(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 2> flipped(etl::dim<0>(input), etl::dim<1>(input));

    std::reverse_copy(input.begin(), input.end(), flipped.begin());

    etl::dyn_matrix<T, 2> padded_input(etl::dim<0>(input), etl::dim<1>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(flipped); ++i){
        direct_copy_n(flipped(i).memory_start(), padded_input(i).memory_start(), etl::dim<1>(flipped));
    }

    return padded_input;
}

template <typename I, cpp_enable_if((decay_traits<I>::dimensions() == 3))>
etl::dyn_matrix<value_t<I>, 3> pad_right_multi_general(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 3> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            direct_copy_n(input(i)(j).memory_start(), padded_input(i)(j).memory_start(), etl::dim<2>(input));
        }
    }

    return padded_input;
}

template <typename I, cpp_enable_if((decay_traits<I>::dimensions() == 4))>
etl::dyn_matrix<value_t<I>, 4> pad_right_multi_general(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 4> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            for(size_t k = 0; k < etl::dim<2>(input); ++k){
                direct_copy_n(input(i)(j)(k).memory_start(), padded_input(i)(j)(k).memory_start(), etl::dim<3>(input));
            }
        }
    }

    return padded_input;
}

template <typename I, cpp_enable_if((decay_traits<I>::dimensions() == 3))>
etl::dyn_matrix<value_t<I>, 3> pad_right_flip_multi_general(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 3> flipped(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input));

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        std::reverse_copy(input(i).memory_start(), input(i).memory_end(), flipped(i).memory_start());
    }

    etl::dyn_matrix<T, 3> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            direct_copy_n(flipped(i)(j).memory_start(), padded_input(i)(j).memory_start(), etl::dim<2>(input));
        }
    }

    return padded_input;
}

template <typename I, cpp_enable_if((decay_traits<I>::dimensions() == 4))>
etl::dyn_matrix<value_t<I>, 4> pad_right_flip_multi_general(const I& input, size_t pad){
    using T = value_t<I>;

    etl::dyn_matrix<T, 4> flipped(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input));

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            std::reverse_copy(input(i)(j).memory_start(), input(i)(j).memory_end(), flipped(i)(j).memory_start());
        }
    }

    etl::dyn_matrix<T, 4> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input), etl::dim<3>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            for(size_t k = 0; k < etl::dim<2>(input); ++k){
                direct_copy_n(flipped(i)(j)(k).memory_start(), padded_input(i)(j)(k).memory_start(), etl::dim<3>(input));
            }
        }
    }

    return padded_input;
}

template <typename T>
etl::dyn_matrix<T, 3> pad_right_multi(const opaque_memory<T, 3>& input, size_t pad){
    etl::dyn_matrix<T, 3> padded_input(input.dim(0), input.dim(1), input.dim(2) + pad);

    padded_input = 0;

    for(size_t i = 0; i < input.dim(0); ++i){
        for(size_t j = 0; j < input.dim(1); ++j){
            direct_copy_n(
                input.memory_start() + i * input.dim(1) * input.dim(2) + j * input.dim(2),
                padded_input.memory_start() + i * padded_input.dim(1) * padded_input.dim(2) + j * padded_input.dim(2),
                input.dim(2));
        }
    }

    return padded_input;
}

template <typename T>
etl::dyn_matrix<T, 4> pad_right_multi(const opaque_memory<T, 4>& input, size_t pad){
    etl::dyn_matrix<T, 4> padded_input(input.dim(0), input.dim(1), input.dim(2), input.dim(3) + pad);

    padded_input = 0;

    const auto C1 = input.dim(1) * input.dim(2) * input.dim(3);
    const auto C2 = input.dim(2) * input.dim(3);
    const auto C3 = input.dim(3);

    const auto PC1 = input.dim(1) * input.dim(2) * (input.dim(3) + pad);
    const auto PC2 = input.dim(2) * (input.dim(3) + pad);
    const auto PC3 = (input.dim(3) + pad);

    for(size_t i = 0; i < input.dim(0); ++i){
        for(size_t j = 0; j < input.dim(1); ++j){
            for(size_t k = 0; k < input.dim(2); ++k){
                direct_copy_n(
                    input.memory_start() + i * C1 + j * C2 + k * C3,
                    padded_input.memory_start() + i * PC1 + j * PC2 + k * PC3,
                    input.dim(3));
            }
        }
    }

    return padded_input;
}

template <typename T>
etl::dyn_matrix<T, 2> pad_right_flip(const opaque_memory<T, 2>& input, size_t pad){
    etl::dyn_matrix<T, 2> flipped(input.dim(0), input.dim(1));
    std::reverse_copy(input.memory_start(), input.memory_end(), flipped.memory_start());

    etl::dyn_matrix<T, 2> padded_input(input.dim(0), input.dim(1) + pad);

    padded_input = 0;

    for(size_t i = 0; i < input.dim(0); ++i){
        direct_copy_n(flipped.memory_start() + i * flipped.dim(1), padded_input.memory_start() + i * padded_input.dim(1), flipped.dim(1));
    }

    return padded_input;
}

template <typename T>
etl::dyn_matrix<T, 3> pad_right_flip_multi(const opaque_memory<T, 3>& input, size_t pad){
    etl::dyn_matrix<T, 3> flipped(input.dim(0), input.dim(1), input.dim(2));

    for(size_t i = 0; i < input.dim(0); ++i){
        std::reverse_copy(
            input.memory_start() + i * input.dim(1) * input.dim(2),
            input.memory_start() + (i+1) * input.dim(1) * input.dim(2),
            flipped.memory_start() + i * input.dim(1) * input.dim(2));
    }

    etl::dyn_matrix<T, 3> padded_input(input.dim(0), input.dim(1), input.dim(2) + pad);

    padded_input = 0;

    for(size_t i = 0; i < input.dim(0); ++i){
        for(size_t j = 0; j < input.dim(1); ++j){
            direct_copy_n(
                flipped.memory_start() + i * flipped.dim(1) * flipped.dim(2) + j * flipped.dim(2),
                padded_input.memory_start() + i * padded_input.dim(1) * padded_input.dim(2) + j * padded_input.dim(2),
                input.dim(2));
        }
    }

    return padded_input;
}

template <typename T>
etl::dyn_matrix<T, 4> pad_right_flip_multi(const opaque_memory<T, 4>& input, size_t pad){
    etl::dyn_matrix<T, 4> flipped(input.dim(0), input.dim(1), input.dim(2), input.dim(3));

    const auto C1 = input.dim(1) * input.dim(2) * input.dim(3);
    const auto C2 = input.dim(2) * input.dim(3);
    const auto C3 = input.dim(3);

    const auto PC1 = input.dim(1) * input.dim(2) * (input.dim(3) + pad);
    const auto PC2 = input.dim(2) * (input.dim(3) + pad);
    const auto PC3 = (input.dim(3) + pad);

    for(size_t i = 0; i < input.dim(0); ++i){
        for(size_t j = 0; j < input.dim(1); ++j){
            std::reverse_copy(
                input.memory_start() + i * C1 + j * C2,
                input.memory_start() + i * C1 + (j+1) * C2,
                flipped.memory_start() + i * C1 + j * C2);
        }
    }

    etl::dyn_matrix<T, 4> padded_input(input.dim(0), input.dim(1), input.dim(2), input.dim(3) + pad);

    padded_input = 0;

    for(size_t i = 0; i < input.dim(0); ++i){
        for(size_t j = 0; j < input.dim(1); ++j){
            for(size_t k = 0; k < input.dim(2); ++k){
                direct_copy_n(
                    flipped.memory_start() + i * C1 + j * C2 + k * C3,
                    padded_input.memory_start() + i * PC1 + j * PC2 + k * PC3,
                    input.dim(3));
            }
        }
    }

    return padded_input;
}

} //end of namespace common
} //end of namespace impl
} //end of namespace etl
