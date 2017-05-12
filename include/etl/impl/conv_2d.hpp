//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for 2D convolution operations
 */

#pragma once

namespace etl {

namespace detail {

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
        auto impl = select_conv2_impl_new<conv_type::FULL, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_STD) {
            impl::standard::conv2_full_fft(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_MKL) {
            impl::blas::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_CUFFT) {
            impl::cufft::conv2_full(input, kernel, conv);
        } else {
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
        auto impl = select_conv2_impl_new<conv_type::FULL, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_STD) {
            impl::standard::conv2_full_fft_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_MKL) {
            impl::blas::conv2_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_CUFFT) {
            impl::cufft::conv2_full_flipped(input, kernel, conv);
        } else {
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
        auto impl = select_conv2_impl_new<conv_type::SAME, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_same(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_same(input, kernel, conv);
        } else {
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
        auto impl = select_conv2_impl_new<conv_type::SAME, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_same_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_same_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_valid(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else {
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
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_valid(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid(input, kernel, conv, s1, s2, p1, p2);
        } else {
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
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv2_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_backward_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if(all_fast<K>::value)>
    static void apply(const I& input, const K& kernel, C& conv) {
        // Need K1 / K2 to compute transposed padding
        static constexpr const size_t K1 = etl::dim<0, K>();
        static constexpr const size_t K2 = etl::dim<1, K>();

        // 1. Handle unit strides
        if /* constexpr */ (S1 == 1 && S2 == 1){
            if /* constexpr */ (P1 == 0 && P2 == 0){
                // Unit strides, non-zero padding -> Full convolution
                conv2_full_impl::apply(input, kernel, conv);
            } else {
                // Unit strides, zero padding -> Valid convolution with the correct padding
                conv2_valid_impl<1, 1, K1 - P1 - 1, K2 - P2 - 1>::apply(input, kernel, conv);
            }
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_input = impl::common::inner_pad(input, S1, S2);

            if /* constexpr */ (P1 == 0 && P2 == 0){
                // Non-unit strides, non-zero padding -> Fractionally-strided full convolution
                conv2_full_impl::apply(strided_input, kernel, conv);
            } else {
                // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
                conv2_valid_impl<1, 1, K1 - P1 - 1, K2 - P2 - 1>::apply(strided_input, kernel, conv);
            }
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv with flipped kernels
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_backward_flipped_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if(all_fast<K>::value)>
    static void apply(const I& input, const K& kernel, C& conv) {
        // Need K1 / K2 to compute transposed padding
        static constexpr const size_t K1 = etl::dim<0, K>();
        static constexpr const size_t K2 = etl::dim<1, K>();

        // 1. Handle unit strides
        if /* constexpr */ (S1 == 1 && S2 == 1){
            if /* constexpr */ (P1 == 0 && P2 == 0){
                // Unit strides, non-zero padding -> Full convolution
                conv2_full_flipped_impl::apply(input, kernel, conv);
            } else {
                // Unit strides, zero padding -> Valid convolution with the correct padding
                conv2_valid_flipped_impl<1, 1, K1 - P1 - 1, K2 - P2 - 1>::apply(input, kernel, conv);
            }
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_input = impl::common::inner_pad(input, S1, S2);

            if /* constexpr */ (P1 == 0 && P2 == 0){
                // Non-unit strides, non-zero padding -> Fractionally-strided full convolution
                conv2_full_flipped_impl::apply(strided_input, kernel, conv);
            } else {
                // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
                conv2_valid_flipped_impl<1, 1, K1 - P1 - 1, K2 - P2 - 1>::apply(strided_input, kernel, conv);
            }
        }
    }
};

} //end of namespace detail

} //end of namespace etl
