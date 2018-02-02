//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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

namespace etl::impl::common {

/*!
 * \brief Pad the input matrix in the output matrix for convolution as multiplication
 * \param in The input matrix
 * \param out The output matrix
 */
template <typename F1, typename F2>
void complex_pad_3d(const F1& in, F2& out) {
    out.ensure_cpu_up_to_date();

    for (size_t outer = 0; outer < etl::dim<0>(in); ++outer) {
        auto* direct = out(outer).memory_start();
        for (size_t i = 0; i < etl::dim<1>(in); ++i) {
            for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                direct[i * etl::dim<2>(out) + j] = in(outer, i, j);
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
void complex_pad_4d(const F1& in, F2& out) {
    out.ensure_cpu_up_to_date();

    for (size_t outer1 = 0; outer1 < etl::dim<0>(in); ++outer1) {
        for (size_t outer2 = 0; outer2 < etl::dim<1>(in); ++outer2) {
            auto* direct = out(outer1)(outer2).memory_start();
            for (size_t i = 0; i < etl::dim<2>(in); ++i) {
                for (size_t j = 0; j < etl::dim<3>(in); ++j) {
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
 * \param p1 The first dimension extra padding of the convolution
 * \param p2 The second dimension extra padding of the convolution
 */
template <typename F1, typename F2>
void pad_2d_input(const F1& in, F2&& out, size_t p1, size_t p2) {
    out.ensure_cpu_up_to_date();

    auto* direct = out.memory_start();

    for (size_t i = 0; i < etl::dim<0>(in); ++i) {
        for (size_t j = 0; j < etl::dim<1>(in); ++j) {
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

    for (size_t n = 0; n < etl::dim<0>(in); ++n) {
        auto* direct = out(n).memory_start();

        for (size_t i = 0; i < etl::dim<1>(in); ++i) {
            for (size_t j = 0; j < etl::dim<2>(in); ++j) {
                direct[(i + p1) * etl::dim<2>(out) + (j + p2)] = in(n, i, j);
            }
        }
    }
}

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
template <typename I_T, typename K_T, typename C_T>
void left_same_kernel(const I_T* in, const size_t n, const K_T* kernel, size_t m, C_T* out, size_t first, size_t last) {
    cpp_unused(n);

    size_t left  = (m - 1) / 2;
    size_t right = m / 2;

    //Left invalid part
    for (size_t j = first; j < std::min(last, left); ++j) {
        C_T temp(0);

        for (size_t l = 0; l <= j + right; ++l) {
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
template <typename I_T, typename K_T, typename C_T>
void right_same_kernel(const I_T* in, const size_t n, const K_T* kernel, size_t m, C_T* out, size_t first, size_t last) {
    size_t left  = (m - 1) / 2;
    size_t right = m / 2;

    //Right invalid part
    for (size_t j = std::max(first, n - right); j < std::min(last, n); ++j) {
        C_T temp(0);

        size_t hi = std::min<int>(n - 1, j + right);
        for (size_t l = j - left; l <= hi; ++l) {
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
template <typename I_T, typename K_T, typename C_T>
void left_full_kernel(const I_T* in, const size_t n, const K_T* kernel, size_t m, C_T* out, size_t first, size_t last) {
    size_t left = m - 1;

    //Left invalid part
    for (size_t i = first; i < std::min(last, left); ++i) {
        const auto hi = i < n - 1 ? i : n - 1;

        C_T temp(0);

        for (size_t j = 0; j <= hi; ++j) {
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
template <typename I_T, typename K_T, typename C_T>
void right_full_kernel(const I_T* in, const size_t n, const K_T* kernel, size_t m, C_T* out, size_t first, size_t last) {
    size_t right = m - 1;

    auto c = n + m - 1;

    //Right invalid part
    for (size_t i = std::max(first, c - right); i < std::min(last, c); ++i) {
        const auto lo = i >= m - 1 ? i - (m - 1) : 0;
        const auto hi = i < n - 1 ? i : n - 1;

        C_T temp(0);

        for (size_t j = lo; j <= hi; ++j) {
            temp += in[j] * kernel[i - j];
        }

        out[i] = temp;
    }
}

/*!
 * \brief Return a new matrix equivalent to padding the last dimension of the input matrix to the right
 * \param input The matrix to pad
 * \param pad The number of padding elements
 * \return a new matrix containing the result
 */
template <typename I>
etl::dyn_matrix<value_t<I>, 2> pad_right(const I& input, size_t pad){
    using T = value_t<I>;

    input.ensure_cpu_up_to_date();

    etl::dyn_matrix<T, 2> padded_input(etl::dim<0>(input), etl::dim<1>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        direct_copy_n(input.memory_start() + i * etl::dim<1>(input), padded_input.memory_start() + i * etl::dim<1>(padded_input), etl::dim<1>(input));
    }

    return padded_input;
}

/*!
 * \brief Return a new matrix equivalent to padding the last dimension of the flipped input matrix to the right
 * \param input The matrix to pad
 * \param pad The number of padding elements
 * \return a new matrix containing the result
 */
template <typename I>
etl::dyn_matrix<value_t<I>, 2> pad_right_flip(const I& input, size_t pad){
    using T = value_t<I>;

    input.ensure_cpu_up_to_date();

    etl::dyn_matrix<T, 2> flipped(etl::dim<0>(input), etl::dim<1>(input));

    std::reverse_copy(input.begin(), input.end(), flipped.begin());

    etl::dyn_matrix<T, 2> padded_input(etl::dim<0>(input), etl::dim<1>(input) + pad);

    padded_input = 0;

    for(size_t i = 0; i < etl::dim<0>(flipped); ++i){
        direct_copy_n(flipped.memory_start() + i * etl::dim<1>(flipped), padded_input.memory_start() + i * etl::dim<1>(padded_input), etl::dim<1>(flipped));
    }

    return padded_input;
}

/*!
 * \brief Return a new matrix equivalent to padding the last dimension of the input matrix to the right
 * \param input The matrix to pad
 * \param pad The number of padding elements
 * \return a new matrix containing the result
 */
template <typename I, cpp_enable_iff(is_3d<I>)>
etl::dyn_matrix<value_t<I>, 3> pad_right_multi(const I& input, size_t pad){
    using T = value_t<I>;

    input.ensure_cpu_up_to_date();

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

/*!
 * \brief Return a new matrix equivalent to padding the last dimension of the input matrix to the right
 * \param input The matrix to pad
 * \param pad The number of padding elements
 * \return a new matrix containing the result
 */
template <typename I, cpp_enable_iff(is_4d<I>)>
etl::dyn_matrix<value_t<I>, 4> pad_right_multi(const I& input, size_t pad){
    using T = value_t<I>;

    input.ensure_cpu_up_to_date();

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

/*!
 * \brief Return a new matrix equivalent to padding the last dimension of the input matrix to the right and padding directly the input with convolution zero padding.
 *
 * \param input The matrix to pad
 * \param pad The number of padding elements
 * \param p1 The first dimension padding
 * \param p2 The second dimension padding
 *
 * \return a new matrix containing the result
 */
template <typename I, cpp_enable_iff(is_4d<I>)>
etl::dyn_matrix<value_t<I>, 4> pad_right_multi_double(const I& input, size_t pad, size_t p1, size_t p2){
    using T = value_t<I>;

    input.ensure_cpu_up_to_date();

    etl::dyn_matrix<T, 4> padded_input(etl::dim<0>(input), etl::dim<1>(input), etl::dim<2>(input) + 2 * p1, etl::dim<3>(input) + pad + 2 * p2);

    padded_input = 0;

    const auto C1 = etl::dim<1>(input) * etl::dim<2>(input) * etl::dim<3>(input);
    const auto C2 = etl::dim<2>(input) * etl::dim<3>(input);
    const auto C3 = etl::dim<3>(input);

    const auto PC1 = etl::dim<1>(padded_input) * etl::dim<2>(padded_input) * etl::dim<3>(padded_input);
    const auto PC2 = etl::dim<2>(padded_input) * etl::dim<3>(padded_input);
    const auto PC3 = etl::dim<3>(padded_input);

    for(size_t i = 0; i < etl::dim<0>(input); ++i){
        for(size_t j = 0; j < etl::dim<1>(input); ++j){
            for(size_t k = 0; k < etl::dim<2>(input); ++k){
                direct_copy_n(
                    input.memory_start() + i * C1 + j * C2 + k * C3,
                    padded_input.memory_start() + i * PC1 + j * PC2 + (p1 + k) * PC3 + p2,
                    etl::dim<3>(input));
            }
        }
    }

    return padded_input;
}

/*!
 * \brief Return a new matrix equivalent to padding the last dimension of the flipped input matrix to the right
 * \param input The matrix to pad
 * \param pad The number of padding elements
 * \return a new matrix containing the result
 */
template <typename I, cpp_enable_iff(is_3d<I>)>
etl::dyn_matrix<value_t<I>, 3> pad_right_flip_multi(const I& input, size_t pad){
    using T = value_t<I>;

    input.ensure_cpu_up_to_date();

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
                flipped.memory_start() + i * etl::dim<1>(flipped) * etl::dim<2>(flipped) + j * etl::dim<2>(flipped),
                padded_input.memory_start() + i * etl::dim<1>(padded_input) * etl::dim<2>(padded_input) + j * etl::dim<2>(padded_input),
                etl::dim<2>(input));
        }
    }

    return padded_input;
}

/*!
 * \brief Return a new matrix equivalent to padding the last dimension of the flipped input matrix to the right
 * \param input The matrix to pad
 * \param pad The number of padding elements
 * \return a new matrix containing the result
 */
template <typename I, cpp_enable_iff(is_4d<I>)>
etl::dyn_matrix<value_t<I>, 4> pad_right_flip_multi(const I& input, size_t pad){
    using T = value_t<I>;

    input.ensure_cpu_up_to_date();

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

/*!
 * \brief Returns a matrix corresponding to the input with some amount of inner
 * padding.
 *
 * This should only be used for fractionally-strided convolution
 *
 * \param in The input matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 *
 * \return A matrix containing the same elements as the input with some innner
 * padding.
 */
template <typename I, cpp_enable_iff(etl::dimensions<I>() == 2)>
etl::dyn_matrix<value_t<I>, 2> inner_pad(const I& in, size_t s1, size_t s2) {
    etl::dyn_matrix<value_t<I>, 2> result((etl::dim<0>(in) - 1) * s1 + 1, (etl::dim<1>(in) - 1) * s2 + 1);

    result = 0;

    in.ensure_cpu_up_to_date();

    for (size_t i = 0; i < etl::dim<0>(in); ++i) {
        for (size_t j = 0; j < etl::dim<1>(in); ++j) {
            result(i * s1, j * s2) = in(i, j);
        }
    }

    return result;
}

/*!
 * \brief Returns a matrix corresponding to the input with some amount of inner
 * padding.
 *
 * This should only be used for fractionally-strided convolution
 *
 * \param in The input matrix
 * \param s1 The first dimension stride
 * \param s2 The second dimension stride
 *
 * \return A matrix containing the same elements as the input with some innner
 * padding.
 */
template <typename I, cpp_enable_iff(etl::dimensions<I>() == 4)>
etl::dyn_matrix<value_t<I>, 4> inner_pad(const I& in, size_t s1, size_t s2) {
    etl::dyn_matrix<value_t<I>, 4> result(etl::dim<0>(in), etl::dim<1>(in), (etl::dim<2>(in) - 1) * s1 + 1, (etl::dim<3>(in) - 1) * s2 + 1);

    result = 0;

    in.ensure_cpu_up_to_date();

    for (size_t p = 0; p < etl::dim<0>(in); ++p) {
        for (size_t q = 0; q < etl::dim<1>(in); ++q) {
            for (size_t i = 0; i < etl::dim<2>(in); ++i) {
                for (size_t j = 0; j < etl::dim<3>(in); ++j) {
                    result(p, q, i * s1, j * s2) = in(p, q, i, j);
                }
            }
        }
    }

    return result;
}

} //end of namespace etl::impl::common
