//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for 4D convolution operations
 */

namespace etl {

namespace detail {

/*!
 * \brief The functor impl for 4D valid conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct dyn_conv4_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct dyn_conv4_valid_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_filter_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid_filter(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_filter(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_filter(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_filter(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_filter_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            if(S1 > 1 || S2 > 1 || P1 || P2){
                // For some reasons, CUDNN backward filter cross correlation does
                // not work correclty (or does not work the way I expect it to work)
                // The padding may not be done as I thought
                if(vec_enabled){
                    impl::vec::conv4_valid_filter_flipped(input, kernel, conv, S1, S2, P1, P2);
                } else {
                    impl::standard::conv4_valid_filter_flipped(input, kernel, conv, S1, S2, P1, P2);
                }
            } else {
                impl::cudnn::conv4_valid_filter_flipped(input, kernel, conv, S1, S2, P1, P2);
            }
        } else if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_filter_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_filter_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_filter_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct dyn_conv4_valid_filter_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid_filter(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_filter(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_filter(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_filter(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct dyn_conv4_valid_filter_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            if(s1 > 1 || s2 > 1 || p1 || p2){
                // For some reasons, CUDNN backward filter cross correlation does
                // not work correclty (or does not work the way I expect it to work)
                // The padding may not be done as I thought
                if(vec_enabled){
                    impl::vec::conv4_valid_filter_flipped(input, kernel, conv, s1, s2, p1, p2);
                } else {
                    impl::standard::conv4_valid_filter_flipped(input, kernel, conv, s1, s2, p1, p2);
                }
            } else {
                impl::cudnn::conv4_valid_filter_flipped(input, kernel, conv, s1, s2, p1, p2);
            }
        } else if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_filter_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_filter_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_filter_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid_back conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_back_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_valid_back_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_back(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_back(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_back(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_back(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid_back conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_back_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_valid_back_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_back_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_back_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_back_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_back_flipped(input, kernel, conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid_back conv
 */
struct dyn_conv4_valid_back_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto impl = select_conv4_valid_back_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_back(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_back(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_back(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_back(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid_back conv
 */
struct dyn_conv4_valid_back_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        auto impl = select_conv4_valid_back_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_back_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_back_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_back_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_back_flipped(input, kernel, conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D full conv
 */
struct conv4_full_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_full_impl<I, K, C>(etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_full(input, kernel, conv);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_full(input, kernel, conv);
        } else if (impl == etl::conv4_impl::FFT_STD) {
            impl::standard::conv4_full_fft(input, kernel, conv);
        } else if (impl == etl::conv4_impl::FFT_MKL) {
            impl::blas::conv4_full(input, kernel, conv);
        } else if (impl == etl::conv4_impl::FFT_CUFFT) {
            impl::cufft::conv4_full(input, kernel, conv);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_full(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D full conv
 */
struct conv4_full_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_full_impl<I, K, C>(etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv4_impl::FFT_STD) {
            impl::standard::conv4_full_fft_flipped(input, kernel, conv);
        } else if (impl == etl::conv4_impl::FFT_MKL) {
            impl::blas::conv4_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv4_impl::FFT_CUFFT) {
            impl::cufft::conv4_full_flipped(input, kernel, conv);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_full_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv
 */
struct dyn_conv4_backward_filter_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        // Need K1 / K2 to compute transposed padding
        const size_t k1 = etl::dim<2>(kernel);
        const size_t k2 = etl::dim<3>(kernel);

        // 1. Handle unit strides
        if (s1 == 1 && s2 == 1){
            // Unit strides -> Valid convolution with the correct padding
            dyn_conv4_valid_filter_impl::apply(input, kernel, conv, 1, 1, k1 - p1 - 1, k2 - p2 - 1);
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_input = impl::common::inner_pad(input, s1, s2);

            // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
            dyn_conv4_valid_filter_impl::apply(strided_input, kernel, conv, 1, 1, k1 - p1 - 1, k2 - p2 - 1);
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv with flipped kernels
 */
struct dyn_conv4_backward_filter_flipped_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        // Need K1 / K2 to compute transposed padding
        const size_t k1 = etl::dim<2>(kernel);
        const size_t k2 = etl::dim<3>(kernel);

        // 1. Handle unit strides
        if (s1 == 1 && s2 == 1){
            // Unit strides, zero padding -> Valid convolution with the correct padding
            dyn_conv4_valid_filter_flipped_impl::apply(input, kernel, conv, 1, 1, k1 - p1 - 1, k2 - p2 - 1);
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_input = impl::common::inner_pad(input, s1, s2);

            // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
            dyn_conv4_valid_filter_flipped_impl::apply(strided_input, kernel, conv, 1, 1, k1 - p1 - 1, k2 - p2 - 1);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
