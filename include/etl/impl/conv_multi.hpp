//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for "multi" convolution operations
 */

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(I&& input, K&& kernel, C&& conv) {
        constexpr_select auto impl = select_conv_valid_multi_impl<I, K, C>();

        if constexpr_select (impl == etl::conv_multi_impl::BLAS_VEC) {
            impl::vec::blas_conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::BLAS_MKL) {
            impl::blas::blas_conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VALID_FFT_MKL) {
            impl::blas::fft_conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_valid_multi(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VEC) {
            impl::vec::conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_multi_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(I&& input, K&& kernel, C&& conv) {
        constexpr_select auto impl = select_conv_valid_multi_impl<I, K, C>();

        if constexpr_select (impl == etl::conv_multi_impl::BLAS_VEC) {
            impl::vec::blas_conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::BLAS_MKL) {
            impl::blas::blas_conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VALID_FFT_MKL) {
            impl::blas::fft_conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_valid_multi_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VEC) {
            impl::vec::conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_multi_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(I&& input, K&& kernel, C&& conv) {
        constexpr_select auto impl = select_conv_valid_multi_multi_impl<I, K, C>();

        if constexpr_select (impl == etl::conv_multi_impl::BLAS_VEC) {
            impl::vec::blas_conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::BLAS_MKL) {
            impl::blas::blas_conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VALID_FFT_MKL) {
            impl::blas::fft_conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VEC) {
            impl::vec::conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_multi_multi_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(I&& input, K&& kernel, C&& conv) {
        constexpr_select auto impl = select_conv_valid_multi_multi_impl<I, K, C>();

        if constexpr_select (impl == etl::conv_multi_impl::BLAS_VEC) {
            impl::vec::blas_conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::BLAS_MKL) {
            impl::blas::blas_conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VALID_FFT_MKL) {
            impl::blas::fft_conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VEC) {
            impl::vec::conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if constexpr_select (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
struct dyn_conv2_valid_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(I&& input, K&& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        constexpr_select auto impl = select_conv_valid_multi_impl<I, K, C>();

        if constexpr_select (impl == etl::conv_multi_impl::BLAS_VEC) {
            impl::vec::blas_conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::BLAS_MKL) {
            impl::blas::blas_conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VALID_FFT_MKL) {
            impl::blas::fft_conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_valid_multi(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VEC) {
            impl::vec::conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
struct dyn_conv2_valid_multi_flipped_impl {
     /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(I&& input, K&& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        constexpr_select auto impl = select_conv_valid_multi_impl<I, K, C>();

        if constexpr_select (impl == etl::conv_multi_impl::BLAS_VEC) {
            impl::vec::blas_conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::BLAS_MKL) {
            impl::blas::blas_conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VALID_FFT_MKL) {
            impl::blas::fft_conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_valid_multi_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VEC) {
            impl::vec::conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
struct dyn_conv2_valid_multi_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(I&& input, K&& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        constexpr_select auto impl = select_conv_valid_multi_multi_impl<I, K, C>();

        if constexpr_select (impl == etl::conv_multi_impl::BLAS_VEC) {
            impl::vec::blas_conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::BLAS_MKL) {
            impl::blas::blas_conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VALID_FFT_MKL) {
            impl::blas::fft_conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VEC) {
            impl::vec::conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi_multi(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
struct dyn_conv2_valid_multi_multi_flipped_impl {
     /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(I&& input, K&& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        constexpr_select auto impl = select_conv_valid_multi_multi_impl<I, K, C>();

        if constexpr_select (impl == etl::conv_multi_impl::BLAS_VEC) {
            impl::vec::blas_conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::BLAS_MKL) {
            impl::blas::blas_conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VALID_FFT_MKL) {
            impl::blas::fft_conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::VEC) {
            impl::vec::conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if constexpr_select (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi_multi_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

} //end of namespace detail

} //end of namespace etl
