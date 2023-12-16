//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for 2D convolution operations
 */

#pragma once

namespace etl::detail {

/*!
 * \brief The functor impl for 2D full conv
 */
struct conv2_full_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        constexpr_select auto impl = select_conv2_impl_new<conv_type::FULL, I, K, C>();

        if
            constexpr_select(impl == etl::conv_impl::VEC) {
                inc_counter("impl:vec");
                impl::vec::conv2_full(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::CUDNN) {
                inc_counter("impl:cudnn");
                impl::cudnn::conv2_full(smart_forward_gpu(input), smart_forward_gpu(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                inc_counter("impl:std");
                impl::standard::conv2_full(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::FFT_STD) {
                inc_counter("impl:fft_std");
                impl::standard::conv2_full_fft(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::FFT_MKL) {
                inc_counter("impl:fft_mkl");
                impl::blas::conv2_full(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::FFT_CUFFT) {
                inc_counter("impl:fft_cufft");
                impl::cufft::conv2_full(smart_forward(input), smart_forward(kernel), conv);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D full conv
 */
struct conv2_full_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        constexpr_select auto impl = select_conv2_impl_new<conv_type::FULL, I, K, C>();

        if
            constexpr_select(impl == etl::conv_impl::VEC) {
                inc_counter("impl:vec");
                impl::vec::conv2_full_flipped(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::CUDNN) {
                inc_counter("impl:cudnn");
                impl::cudnn::conv2_full_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                inc_counter("impl:std");
                impl::standard::conv2_full_flipped(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::FFT_STD) {
                inc_counter("impl:fft_std");
                impl::standard::conv2_full_fft_flipped(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::FFT_MKL) {
                inc_counter("impl:fft_mkl");
                impl::blas::conv2_full_flipped(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::FFT_CUFFT) {
                inc_counter("impl:fft_cufft");
                impl::cufft::conv2_full_flipped(smart_forward(input), smart_forward(kernel), conv);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D same conv
 */
struct conv2_same_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        constexpr_select auto impl = select_conv2_impl_new<conv_type::SAME, I, K, C>();

        if
            constexpr_select(impl == etl::conv_impl::VEC) {
                inc_counter("impl:vec");
                impl::vec::conv2_same(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                inc_counter("impl:std");
                impl::standard::conv2_same(smart_forward(input), smart_forward(kernel), conv);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D same conv
 */
struct conv2_same_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        constexpr_select auto impl = select_conv2_impl_new<conv_type::SAME, I, K, C>();

        if
            constexpr_select(impl == etl::conv_impl::VEC) {
                inc_counter("impl:vec");
                impl::vec::conv2_same_flipped(smart_forward(input), smart_forward(kernel), conv);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                inc_counter("impl:std");
                impl::standard::conv2_same_flipped(smart_forward(input), smart_forward(kernel), conv);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        constexpr_select auto impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if /*constepxr_select*/ (impl == etl::conv_impl::VEC) {
            inc_counter("impl:vec");
            impl::vec::conv2_valid(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if
            constexpr_select(impl == etl::conv_impl::CUDNN) {
                inc_counter("impl:cudnn");
                impl::cudnn::conv2_valid(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, S1, S2, P1, P2);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                inc_counter("impl:std");
                impl::standard::conv2_valid(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        constexpr_select auto impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if /*constepxr_select*/ (impl == etl::conv_impl::VEC) {
                inc_counter("impl:vec");
            impl::vec::conv2_valid_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if
            constexpr_select(impl == etl::conv_impl::CUDNN) {
                inc_counter("impl:cudnn");
                impl::cudnn::conv2_valid_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, S1, S2, P1, P2);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                inc_counter("impl:std");
                impl::standard::conv2_valid_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
struct dyn_conv2_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        constexpr_select auto impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if /*constepxr_select*/ (impl == etl::conv_impl::VEC) {
            inc_counter("impl:vec");
            impl::vec::conv2_valid(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if
            constexpr_select(impl == etl::conv_impl::CUDNN) {
                inc_counter("impl:cudnn");
                impl::cudnn::conv2_valid(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, s1, s2, p1, p2);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                inc_counter("impl:std");
                impl::standard::conv2_valid(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
struct dyn_conv2_valid_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        constexpr_select auto impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if /*constepxr_select*/ (impl == etl::conv_impl::VEC) {
            inc_counter("impl:vec");
            impl::vec::conv2_valid_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if
            constexpr_select(impl == etl::conv_impl::CUDNN) {
                inc_counter("impl:cudnn");
                impl::cudnn::conv2_valid_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, s1, s2, p1, p2);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                inc_counter("impl:std");
                impl::standard::conv2_valid_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

} //end of namespace etl::detail
