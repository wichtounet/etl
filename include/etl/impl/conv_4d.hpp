//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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
template <size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
#ifndef ETL_MANUAL_SELECT
        if /*constexpr*/ (impl::cudnn::conv_possible<I, K, C>) {
            impl::cudnn::conv4_forward(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, S1, S2, P1, P2);
        } else {
#endif
            auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

            if (impl == etl::conv4_impl::CUDNN) {
                impl::cudnn::conv4_forward(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, S1, S2, P1, P2);
            } else if (impl == etl::conv4_impl::BLAS_VEC) {
                impl::vec::blas_conv4_valid(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            } else if (impl == etl::conv4_impl::BLAS_MKL) {
                impl::blas::blas_conv4_valid(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            } else if (impl == etl::conv4_impl::VEC) {
                impl::vec::conv4_valid(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            } else if (impl == etl::conv4_impl::STD) {
                impl::standard::conv4_valid(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            } else {
                cpp_unreachable("Invalid conv implementation selection");
            }
#ifndef ETL_MANUAL_SELECT
        }
#endif
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
#ifndef ETL_MANUAL_SELECT
        if /*constexpr*/ (impl::cudnn::conv_possible<I, K, C>) {
            impl::cudnn::conv4_forward_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, S1, S2, P1, P2);
        } else {
#endif
            auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

            if (impl == etl::conv4_impl::CUDNN) {
                impl::cudnn::conv4_forward_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, S1, S2, P1, P2);
            } else if (impl == etl::conv4_impl::BLAS_VEC) {
                impl::vec::blas_conv4_valid_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            } else if (impl == etl::conv4_impl::BLAS_MKL) {
                impl::blas::blas_conv4_valid_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            } else if (impl == etl::conv4_impl::VEC) {
                impl::vec::conv4_valid_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            } else if (impl == etl::conv4_impl::STD) {
                impl::standard::conv4_valid_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
            } else {
                cpp_unreachable("Invalid conv implementation selection");
            }
#ifndef ETL_MANUAL_SELECT
        }
#endif
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
#ifndef ETL_MANUAL_SELECT
        if /*constexpr*/ (impl::cudnn::conv_possible<I, K, C>) {
            impl::cudnn::conv4_forward(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, s1, s2, p1, p2);
        } else {
#endif
            auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

            if (impl == etl::conv4_impl::CUDNN) {
                impl::cudnn::conv4_forward(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, s1, s2, p1, p2);
            } else if (impl == etl::conv4_impl::BLAS_VEC) {
                impl::vec::blas_conv4_valid(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            } else if (impl == etl::conv4_impl::BLAS_MKL) {
                impl::blas::blas_conv4_valid(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            } else if (impl == etl::conv4_impl::VEC) {
                impl::vec::conv4_valid(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            } else if (impl == etl::conv4_impl::STD) {
                impl::standard::conv4_valid(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            } else {
                cpp_unreachable("Invalid conv implementation selection");
            }
#ifndef ETL_MANUAL_SELECT
        }
#endif
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
#ifndef ETL_MANUAL_SELECT
        if /*constexpr*/ (impl::cudnn::conv_possible<I, K, C>) {
            impl::cudnn::conv4_forward_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, s1, s2, p1, p2);
        } else {
#endif
            auto impl = select_conv4_valid_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

            if (impl == etl::conv4_impl::CUDNN) {
                impl::cudnn::conv4_forward_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv, s1, s2, p1, p2);
            } else if (impl == etl::conv4_impl::BLAS_VEC) {
                impl::vec::blas_conv4_valid_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            } else if (impl == etl::conv4_impl::BLAS_MKL) {
                impl::blas::blas_conv4_valid_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            } else if (impl == etl::conv4_impl::VEC) {
                impl::vec::conv4_valid_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            } else if (impl == etl::conv4_impl::STD) {
                impl::standard::conv4_valid_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
            } else {
                cpp_unreachable("Invalid conv implementation selection");
            }
#ifndef ETL_MANUAL_SELECT
        }
#endif
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_filter_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_valid_filter_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_filter(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_filter(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_filter(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
struct conv4_valid_filter_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_valid_filter_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_filter_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_filter_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_filter_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
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
        auto impl = select_conv4_valid_filter_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_filter(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_filter(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_filter(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
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
        auto impl = select_conv4_valid_filter_impl<I, K, C>(etl::dim<2>(input), etl::dim<3>(input), etl::dim<2>(kernel), etl::dim<3>(kernel));

        if (impl == etl::conv4_impl::BLAS_VEC) {
            impl::vec::blas_conv4_valid_filter_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_filter_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_filter_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid_back conv
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
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
            impl::vec::blas_conv4_valid_back(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_back(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_back(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_back(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid_back conv
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
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
            impl::vec::blas_conv4_valid_back_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_back_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_back_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_back_flipped(smart_forward(input), smart_forward(kernel), conv, S1, S2, P1, P2);
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
            impl::vec::blas_conv4_valid_back(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_back(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_back(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_back(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
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
            impl::vec::blas_conv4_valid_back_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::BLAS_MKL) {
            impl::blas::blas_conv4_valid_back_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::VEC) {
            impl::vec::conv4_valid_back_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_back_flipped(smart_forward(input), smart_forward(kernel), conv, s1, s2, p1, p2);
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
#ifndef ETL_MANUAL_SELECT
        if /*constexpr*/ (impl::cudnn::conv_possible<I, K, C>) {
            impl::cudnn::conv4_backward_data_full(smart_forward_gpu(input), smart_forward_gpu(kernel), conv);
        } else {
#endif
            auto impl = select_conv4_full_impl<I, K, C>(etl::dim<2>(kernel), etl::dim<3>(kernel));

            if (impl == etl::conv4_impl::CUDNN) {
                impl::cudnn::conv4_backward_data_full(smart_forward_gpu(input), smart_forward_gpu(kernel), conv);
            } else if (impl == etl::conv4_impl::VEC) {
                impl::vec::conv4_full(smart_forward(input), smart_forward(kernel), conv);
            } else if (impl == etl::conv4_impl::FFT_STD) {
                impl::standard::conv4_full_fft(smart_forward(input), smart_forward(kernel), conv);
            } else if (impl == etl::conv4_impl::FFT_MKL) {
                impl::blas::conv4_full(smart_forward(input), smart_forward(kernel), conv);
            } else if (impl == etl::conv4_impl::FFT_CUFFT) {
                impl::cufft::conv4_full(smart_forward_gpu(input), smart_forward_gpu(kernel), conv);
            } else if (impl == etl::conv4_impl::STD) {
                impl::standard::conv4_full(smart_forward(input), smart_forward(kernel), conv);
            } else {
                cpp_unreachable("Invalid conv implementation selection");
            }
#ifndef ETL_MANUAL_SELECT
        }
#endif
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
#ifndef ETL_MANUAL_SELECT
        if /*constexpr*/ (impl::cudnn::conv_possible<I, K, C>) {
            impl::cudnn::conv4_backward_data_full_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv);
        } else {
#endif
            auto impl = select_conv4_full_impl<I, K, C>(etl::dim<2>(kernel), etl::dim<3>(kernel));

            if (impl == etl::conv4_impl::CUDNN) {
                impl::cudnn::conv4_backward_data_full_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv);
            } else if (impl == etl::conv4_impl::VEC) {
                impl::vec::conv4_full_flipped(smart_forward(input), smart_forward(kernel), conv);
            } else if (impl == etl::conv4_impl::FFT_STD) {
                impl::standard::conv4_full_fft_flipped(smart_forward(input), smart_forward(kernel), conv);
            } else if (impl == etl::conv4_impl::FFT_MKL) {
                impl::blas::conv4_full_flipped(smart_forward(input), smart_forward(kernel), conv);
            } else if (impl == etl::conv4_impl::FFT_CUFFT) {
                impl::cufft::conv4_full_flipped(smart_forward_gpu(input), smart_forward_gpu(kernel), conv);
            } else if (impl == etl::conv4_impl::STD) {
                impl::standard::conv4_full_flipped(smart_forward(input), smart_forward(kernel), conv);
            } else {
                cpp_unreachable("Invalid conv implementation selection");
            }
#ifndef ETL_MANUAL_SELECT
        }
#endif
    }
};

} //end of namespace detail

} //end of namespace etl
