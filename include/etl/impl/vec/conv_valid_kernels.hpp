//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/common/conv.hpp"

// The different optimized kernels
#include "etl/impl/vec/conv_3x4.hpp"
#include "etl/impl/vec/conv_3x8.hpp"
#include "etl/impl/vec/conv_8x8.hpp"
#include "etl/impl/vec/conv_nx8.hpp"
#include "etl/impl/vec/conv_nx16.hpp"
#include "etl/impl/vec/conv_5x8.hpp"

/*
 * Performance notes:
 *
 * Several things are suboptimal in this implementation
 * a) Ideally, the valid_micro_kernel should directly use input.loadu instead of
 * going through the memory and vec_type::loadu(mem), but compilers do not seem
 * to be able to go through layers of sub view for memory_start
 * b) Ideally the micro kernel should use conv(i, j) directly instead of using
 * the pointer. But there is a small overhead when using this technique :s
 */

namespace etl {

namespace impl {

namespace vec {

namespace detail {

/*!
 * \brief Handle a point in the border for computation of valid convolution.
 *
 * \param input The input matrix of dimensions (n1, n2)
 * \param kernel The kernel matrix of dimensions (m1, m2)
 * \param conv The output matrix
 * \param i The first index to compute
 * \param j The second index to compute
 * \param s1 The stride in the first dimension
 * \param s2 The stride in the second dimension
 * \param p1 The padding in the first dimension
 * \param p2 The padding in the second dimension
 * \param beta The multiplicative for the previous values of out
 */
template <typename I, typename K, typename C>
inline void conv2_valid_flipped_border(const I& input, const K& kernel, C&& conv, size_t i, size_t j, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta) {
    using T = value_t<I>;

    const size_t n1 = etl::dim<0>(input);
    const size_t n2 = etl::dim<1>(input);

    const size_t m1 = etl::dim<0>(kernel);
    const size_t m2 = etl::dim<1>(kernel);

    T temp = T(0);

    const size_t s_i = i * s1;
    const size_t s_j = j * s2;

    for (size_t k = 0; k < m1; ++k) {
        for (size_t l = 0; l < m2; ++l) {
            if (s_i + k >= p1 && (s_i + k) - p1 < n1 && s_j + l >= p2 && (s_j + l) - p2 < n2) {
                const size_t i_i = (s_i + k) - p1;
                const size_t i_j = (s_j + l) - p2;

                temp += input(i_i, i_j) * kernel(k, l);
            }
        }
    }

    if (beta == T(0)) {
        conv(i, j) = temp;
    } else {
        conv(i, j) = beta * conv(i, j) + temp;
    }
}

/*!
 * \brief Vectorized implementation of the inner valid computation
 * of a 2D convolution.
 *
 * \param in The input matrix of dimensions (n1, n2)
 * \param n1 The first dimension  of the input
 * \param n2 The first dimension  of the input
 * \param kkk The kernel matrix of dimensions (m1, m2)
 * \param m1 The first dimension  of the kernel
 * \param m2 The first dimension  of the kernel
 * \param out The output matrix
 * \param s1 The stride in the first dimension
 * \param s2 The stride in the second dimension
 * \param p1 The padding in the first dimension
 * \param p2 The padding in the second dimension
 * \param beta The multiplicative for the previous values of out
 */
template <typename V, typename T>
void conv2_valid_flipped_inner_kernel(const T* in, size_t n1, size_t n2, const T* kkk, size_t m1, size_t m2, T* out, size_t s1, size_t s2, size_t p1, size_t p2, T beta) {
    using vec_type = V;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    const size_t c1 = (n1 - m1 + 2 * p1) / s1 + 1;
    const size_t c2 = (n2 - m2 + 2 * p2) / s2 + 1;

    if (cpp_likely(!p1 && !p2 && s1 == 1 && s2 == 1)) {
        if (vec_size == 4 && m1 == 3 && m2 == 4) {
            conv2_valid_flipped_micro_kernel_3x4<V>(in, n1, n2, kkk, out, beta);
            return;
        } else if (vec_size == 8 && m1 == 3 && m2 == 8) {
            conv2_valid_flipped_micro_kernel_3x8<V>(in, n1, n2, kkk, out, beta);
            return;
        } else if (vec_size == 8 && m1 == 5 && m2 == 8) {
            conv2_valid_flipped_micro_kernel_5x8<V>(in, n1, n2, kkk, out, beta);
            return;
        } else if (vec_size == 8 && m1 == 8 && m2 == 8) {
            conv2_valid_flipped_micro_kernel_8x8<V>(in, n1, n2, kkk, out, beta);
            return;
        } else if (vec_size == 8 && m2 == 8) {
            conv2_valid_flipped_micro_kernel_nx8<V>(in, n1, n2, kkk, m1, out, beta);
            return;
        } else if (vec_size == 8 && m2 == 16) {
            conv2_valid_flipped_micro_kernel_nx16<V>(in, n1, n2, kkk, m1, out, beta);
            return;
        }
    }

    if (beta == T(0)) {
        for (size_t i = p1; i < c1 - p1; ++i) {
            size_t j = p2;

            for (; j + 7 < c2 - p2; j += 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);

                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        auto i2 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 1) * s2 - p2) + l);
                        auto i3 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 2) * s2 - p2) + l);
                        auto i4 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 3) * s2 - p2) + l);
                        auto i5 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 4) * s2 - p2) + l);
                        auto i6 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 5) * s2 - p2) + l);
                        auto i7 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 6) * s2 - p2) + l);
                        auto i8 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 7) * s2 - p2) + l);

                        r1 = vec_type::fmadd(i1, k1, r1);
                        r2 = vec_type::fmadd(i2, k1, r2);
                        r3 = vec_type::fmadd(i3, k1, r3);
                        r4 = vec_type::fmadd(i4, k1, r4);
                        r5 = vec_type::fmadd(i5, k1, r5);
                        r6 = vec_type::fmadd(i6, k1, r6);
                        r7 = vec_type::fmadd(i7, k1, r7);
                        r8 = vec_type::fmadd(i8, k1, r8);
                    }
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
                out[i * c2 + j + 2] = vec_type::hadd(r3);
                out[i * c2 + j + 3] = vec_type::hadd(r4);
                out[i * c2 + j + 4] = vec_type::hadd(r5);
                out[i * c2 + j + 5] = vec_type::hadd(r6);
                out[i * c2 + j + 6] = vec_type::hadd(r7);
                out[i * c2 + j + 7] = vec_type::hadd(r8);
            }

            for (; j + 1 < c2 - p2; j += 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);

                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        auto i2 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 1) * s2 - p2) + l);

                        r1 = vec_type::fmadd(i1, k1, r1);
                        r2 = vec_type::fmadd(i2, k1, r2);
                    }
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
                out[i * c2 + j + 1] = vec_type::hadd(r2);
            }

            if (j < c2 - p2) {
                auto r1 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);
                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        r1      = vec_type::fmadd(i1, k1, r1);
                    }
                }

                out[i * c2 + j + 0] = vec_type::hadd(r1);
            }
        }
    } else {
        for (size_t i = p1; i < c1 - p1; ++i) {
            size_t j = p2;

            for (; j + 7 < c2 - p2; j += 8) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();
                auto r3 = vec_type::template zero<T>();
                auto r4 = vec_type::template zero<T>();
                auto r5 = vec_type::template zero<T>();
                auto r6 = vec_type::template zero<T>();
                auto r7 = vec_type::template zero<T>();
                auto r8 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);

                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        auto i2 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 1) * s2 - p2) + l);
                        auto i3 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 2) * s2 - p2) + l);
                        auto i4 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 3) * s2 - p2) + l);
                        auto i5 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 4) * s2 - p2) + l);
                        auto i6 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 5) * s2 - p2) + l);
                        auto i7 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 6) * s2 - p2) + l);
                        auto i8 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 7) * s2 - p2) + l);

                        r1 = vec_type::fmadd(i1, k1, r1);
                        r2 = vec_type::fmadd(i2, k1, r2);
                        r3 = vec_type::fmadd(i3, k1, r3);
                        r4 = vec_type::fmadd(i4, k1, r4);
                        r5 = vec_type::fmadd(i5, k1, r5);
                        r6 = vec_type::fmadd(i6, k1, r6);
                        r7 = vec_type::fmadd(i7, k1, r7);
                        r8 = vec_type::fmadd(i8, k1, r8);
                    }
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
                out[i * c2 + j + 2] = beta * out[i * c2 + j + 2] + vec_type::hadd(r3);
                out[i * c2 + j + 3] = beta * out[i * c2 + j + 3] + vec_type::hadd(r4);
                out[i * c2 + j + 4] = beta * out[i * c2 + j + 4] + vec_type::hadd(r5);
                out[i * c2 + j + 5] = beta * out[i * c2 + j + 5] + vec_type::hadd(r6);
                out[i * c2 + j + 6] = beta * out[i * c2 + j + 6] + vec_type::hadd(r7);
                out[i * c2 + j + 7] = beta * out[i * c2 + j + 7] + vec_type::hadd(r8);
            }

            for (; j + 1 < c2 - p2; j += 2) {
                auto r1 = vec_type::template zero<T>();
                auto r2 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);

                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        auto i2 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 1) * s2 - p2) + l);

                        r1 = vec_type::fmadd(i1, k1, r1);
                        r2 = vec_type::fmadd(i2, k1, r2);
                    }
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
                out[i * c2 + j + 1] = beta * out[i * c2 + j + 1] + vec_type::hadd(r2);
            }

            if (j < c2 - p2) {
                auto r1 = vec_type::template zero<T>();

                const size_t i_i = i * s1 - p1;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = 0; l + vec_size - 1 < m2; l += vec_size) {
                        auto k1 = vec_type::loadu(kkk + k * m2 + l);
                        auto i1 = vec_type::loadu(in + (i_i + k) * n2 + ((j + 0) * s2 - p2) + l);
                        r1      = vec_type::fmadd(i1, k1, r1);
                    }
                }

                out[i * c2 + j + 0] = beta * out[i * c2 + j + 0] + vec_type::hadd(r1);
            }
        }
    }

    if (!padding_impl && m2 % vec_size != 0) {
        size_t rem = m2 % vec_size;
        for (size_t i = p1; i < c1 - p1; ++i) {
            for (size_t j = p2; j < c2 - p2; ++j) {
                T temp = 0.0;

                const size_t i_i = i * s1 - p1;
                const size_t i_j = j * s2 - p2;

                for (size_t k = 0; k < m1; ++k) {
                    for (size_t l = m2 - rem; l < m2; ++l) {
                        temp += in[i_i + k * n2 + i_j + l] * kkk[k * m2 + l];
                    }
                }

                out[i * c2 + j] += temp;
            }
        }
    }
}

/*!
 * \brief Outer kernel for the vectorized implementation of a 2D valid convolution.
 *
 * This will handle the border cases and delegate the inner work to
 * the optimized inner kernel.
 *
 * \param input The input matrix of dimensions (n1, n2)
 * \param kernel The kernel matrix of dimensions (m1, m2)
 * \param conv The output matrix
 * \param s1 The stride in the first dimension
 * \param s2 The stride in the second dimension
 * \param p1 The padding in the first dimension
 * \param p2 The padding in the second dimension
 * \param beta The multiplicative for the previous values of out
 */
template <typename V, typename I, typename K, typename C>
void conv2_valid_flipped_micro_kernel(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta) {
    using T        = value_t<I>;

    const size_t n1 = etl::dim<0>(input);
    const size_t n2 = etl::dim<1>(input);

    const size_t m1 = etl::dim<0>(kernel);
    const size_t m2 = etl::dim<1>(kernel);

    const size_t c1 = (n1 - m1 + 2 * p1) / s1 + 1;
    const size_t c2 = (n2 - m2 + 2 * p2) / s2 + 1;

    input.ensure_cpu_up_to_date();
    kernel.ensure_cpu_up_to_date();

    if (beta != T(0)) {
        conv.ensure_cpu_up_to_date();
    }

    auto* kkk = kernel.memory_start();
    auto* in  = input.memory_start();
    auto* out = conv.memory_start();

    conv2_valid_flipped_inner_kernel<V>(in, n1, n2, kkk, m1, m2, out, s1, s2, p1, p2, beta);

    conv.invalidate_gpu();

    if (cpp_unlikely(p1 || p2)) {
        for (size_t i = 0; i < p1; ++i) {
            for (size_t j = 0; j < c2; ++j) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t i = c1 - p1; i < c1; ++i) {
            for (size_t j = 0; j < c2; ++j) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t j = 0; j < p2; ++j) {
            for (size_t i = p1; i < c1 - p1; ++i) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }

        for (size_t j = c2 - p2; j < c2; ++j) {
            for (size_t i = p1; i < c1 - p1; ++i) {
                conv2_valid_flipped_border(input, kernel, conv, i, j, s1, s2, p1, p2, beta);
            }
        }
    }
}

/*!
 * \brief Outer kernel for the vectorized implementation of a 2D valid convolution, with kernels not flipped.
 *
 * This will handle the border cases and delegate the inner work to
 * the optimized inner kernel.
 *
 * \param input The input matrix of dimensions (n1, n2)
 * \param kernel The kernel matrix of dimensions (m1, m2)
 * \param conv The output matrix
 * \param s1 The stride in the first dimension
 * \param s2 The stride in the second dimension
 * \param p1 The padding in the first dimension
 * \param p2 The padding in the second dimension
 * \param beta The multiplicative for the previous values of out
 */
template <typename V, typename I, typename K, typename C>
void conv2_valid_micro_kernel(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2, value_t<I> beta) {
    etl::dyn_matrix<value_t<I>, 2> kernel_reverse(etl::dim<0>(kernel), etl::dim<1>(kernel));

    std::reverse_copy(kernel.begin(), kernel.end(), kernel_reverse.begin());

    conv2_valid_flipped_micro_kernel<V>(input, kernel_reverse, conv, s1, s2, p1, p2, beta);
}

} // end of namespace detail

} //end of namespace vec
} //end of namespace impl
} //end of namespace etl
