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

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_full";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_full");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_full");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_full");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0), "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(conv, 1) == etl::dim(kernel, 1), "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(input, 1) == etl::dim(kernel, 0), "Invalid dimensions for conv4_full");

        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 2) + etl::dim(kernel, 2) - 1, "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(conv, 3) == etl::dim(input, 3) + etl::dim(kernel, 3) - 1, "Invalid dimensions for conv4_full");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_full");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_full");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_full");

        static_assert(etl::dim<0,C>() == etl::dim<0,I>(), "Invalid dimensions for conv4_full");
        static_assert(etl::dim<1,C>() == etl::dim<1,K>(), "Invalid dimensions for conv4_full");
        static_assert(etl::dim<1,I>() == etl::dim<0,K>(), "Invalid dimensions for conv4_full");

        static_assert(etl::dim<2,C>() == etl::dim<2,I>() + etl::dim<2,K>() - 1, "Invalid dimensions for conv4_full");
        static_assert(etl::dim<3,C>() == etl::dim<3,I>() + etl::dim<3,K>() - 1, "Invalid dimensions for conv4_full");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(input, 0);
        } else if(d == 1){
            return etl::dim(kernel, 1);
        } else {
            return etl::dim(input, d) + etl::dim(kernel, d) - 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<0,I>()
            :  D == 1 ? etl::dim<1,K>()
            : etl::dim<D,I>() + etl::dim<D,K>() - 1;
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

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_full_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv4_full_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv4_full_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(input, 0);
        } else if(d == 1){
            return etl::dim(kernel, 1);
        } else {
            return etl::dim(input, d) + etl::dim(kernel, d) - 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<0,I>()
            :  D == 1 ? etl::dim<1,K>()
            : etl::dim<D,I>() + etl::dim<D,K>() - 1;
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_backward_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        // The GPU implementation needs the real forward parameters, not the
        // converted backward parameters
        if /* constexpr*/ (cudnn_enabled && all_floating<I, K, C>::value){
            impl::cudnn::conv4_backward(input, kernel, conv, S1, S2, P1, P2);
            return;
        }

        // Need K1 / K2 to compute transposed padding
        const size_t K1 = etl::dim<2>(kernel);
        const size_t K2 = etl::dim<3>(kernel);

        // 1. Handle unit strides
        if /* constexpr */ (S1 == 1 && S2 == 1){
            if /* constexpr */ (P1 == 0 && P2 == 0){
                // Unit strides, non-zero padding -> Full convolution
                conv4_full_impl::apply(input, kernel, conv);
            } else {
                // Unit strides, zero padding -> Valid convolution with the correct padding
                dyn_conv4_valid_back_impl::apply(input, kernel, conv, 1, 1, K1 - P1 - 1, K2 - P2 - 1);
            }
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_input = impl::common::inner_pad(input, S1, S2);

            if /* constexpr */ (P1 == 0 && P2 == 0){
                // Non-unit strides, non-zero padding -> Fractionally-strided full convolution
                conv4_full_impl::apply(strided_input, kernel, conv);
            } else {
                // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
                dyn_conv4_valid_back_impl::apply(strided_input, kernel, conv, 1, 1, K1 - P1 - 1, K2 - P2 - 1);
            }
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv with flipped kernels
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_backward_flipped_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        // The GPU implementation needs the real forward parameters, not the
        // converted backward parameters
        if /* constexpr*/ (cudnn_enabled && all_floating<I, K, C>::value){
            impl::cudnn::conv4_backward_flipped(input, kernel, conv, S1, S2, P1, P2);
            return;
        }

        // Need K1 / K2 to compute transposed padding
        const size_t K1 = etl::dim<2>(kernel);
        const size_t K2 = etl::dim<3>(kernel);

        // 1. Handle unit strides
        if /* constexpr */ (S1 == 1 && S2 == 1){
            if /* constexpr */ (P1 == 0 && P2 == 0){
                // Unit strides, non-zero padding -> Full convolution
                conv4_full_flipped_impl::apply(input, kernel, conv);
            } else {
                // Unit strides, zero padding -> Valid convolution with the correct padding
                dyn_conv4_valid_back_flipped_impl::apply(input, kernel, conv, 1, 1, K1 - P1 - 1, K2 - P2 - 1);
            }
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_input = impl::common::inner_pad(input, S1, S2);

            if /* constexpr */ (P1 == 0 && P2 == 0){
                // Non-unit strides, non-zero padding -> Fractionally-strided full convolution
                conv4_full_flipped_impl::apply(strided_input, kernel, conv);
            } else {
                // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
                dyn_conv4_valid_back_flipped_impl::apply(strided_input, kernel, conv, 1, 1, K1 - P1 - 1, K2 - P2 - 1);
            }
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv
 */
struct dyn_conv4_backward_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        // The GPU implementation needs the real forward parameters, not the
        // converted backward parameters
        if /* constexpr*/ (cudnn_enabled && all_floating<I, K, C>::value){
            impl::cudnn::conv4_backward(input, kernel, conv, s1, s2, p1, p2);
            return;
        }

        // Need K1 / K2 to compute transposed padding
        const size_t k1 = etl::dim<2>(kernel);
        const size_t k2 = etl::dim<3>(kernel);

        // 1. Handle unit strides
        if (s1 == 1 && s2 == 1){
            if (p1 == 0 && p2 == 0){
                // Unit strides, non-zero padding -> Full convolution
                conv4_full_impl::apply(input, kernel, conv);
            } else {
                // Unit strides, zero padding -> Valid convolution with the correct padding
                dyn_conv4_valid_back_impl::apply(input, kernel, conv, 1, 1, k1 - p1 - 1, k2 - p2 - 1);
            }
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_input = impl::common::inner_pad(input, s1, s2);

            if (p1 == 0 && p2 == 0){
                // Non-unit strides, non-zero padding -> Fractionally-strided full convolution
                conv4_full_impl::apply(strided_input, kernel, conv);
            } else {
                // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
                dyn_conv4_valid_back_impl::apply(strided_input, kernel, conv, 1, 1, k1 - p1 - 1, k2 - p2 - 1);
            }
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv with flipped kernels
 */
struct dyn_conv4_backward_flipped_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv, size_t s1, size_t s2, size_t p1, size_t p2) {
        // The GPU implementation needs the real forward parameters, not the
        // converted backward parameters
        if /* constexpr*/ (cudnn_enabled && all_floating<I, K, C>::value){
            impl::cudnn::conv4_backward_flipped(input, kernel, conv, s1, s2, p1, p2);
            return;
        }

        // Need K1 / K2 to compute transposed padding
        const size_t k1 = etl::dim<2>(kernel);
        const size_t k2 = etl::dim<3>(kernel);

        // 1. Handle unit strides
        if (s1 == 1 && s2 == 1){
            if (p1 == 0 && p2 == 0){
                // Unit strides, non-zero padding -> Full convolution
                conv4_full_flipped_impl::apply(input, kernel, conv);
            } else {
                // Unit strides, zero padding -> Valid convolution with the correct padding
                dyn_conv4_valid_back_flipped_impl::apply(input, kernel, conv, 1, 1, k1 - p1 - 1, k2 - p2 - 1);
            }
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_input = impl::common::inner_pad(input, s1, s2);

            if (p1 == 0 && p2 == 0){
                // Non-unit strides, non-zero padding -> Fractionally-strided full convolution
                conv4_full_flipped_impl::apply(strided_input, kernel, conv);
            } else {
                // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
                dyn_conv4_valid_back_flipped_impl::apply(strided_input, kernel, conv, 1, 1, k1 - p1 - 1, k2 - p2 - 1);
            }
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_backward_filter_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        // 1. Handle unit strides
        if /* constexpr */ (S1 == 1 && S2 == 1){
            // Unit strides -> Valid convolution with the correct padding
            dyn_conv4_valid_filter_impl::apply(input, kernel, conv, 1, 1, P1, P2);
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the kernel
            auto strided_kernel = impl::common::inner_pad(kernel, S1, S2);

            // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
            dyn_conv4_valid_filter_impl::apply(input, strided_kernel, conv, 1, 1, P1, P2);
        }
    }
};

/*!
 * \brief The functor impl for 2D tranposed conv with flipped kernels
 */
template<size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_backward_filter_flipped_impl {
    /*!
     * \brief Apply the backward convolution
     *
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C& conv) {
        // 1. Handle unit strides
        if /* constexpr */ (S1 == 1 && S2 == 1){
            // Unit strides, zero padding -> Valid convolution with the correct padding
            dyn_conv4_valid_filter_flipped_impl::apply(input, kernel, conv, 1, 1, P1, P2);
        }
        // 2. Handle non_unit strides
        else {
            // Fractionally-strided convolution needs inner padding of the input
            auto strided_kernel = impl::common::inner_pad(kernel, S1, S2);

            // Non-unit strides, zero padding -> Fractionally-strided Valid convolution with the correct padding
            dyn_conv4_valid_filter_flipped_impl::apply(input, strided_kernel, conv, 1, 1, P1, P2);
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
