//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains selectors for convolution implementations.
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different types of convolution
 */
enum class conv_type {
    VALID,       ///< Valid convolution
    VALID_MULTI, ///< Valid convolution, with multiple kernels
    SAME,        ///< Same convolution
    SAME_MULTI,  ///< Same convolution, with multiple kernels
    FULL,        ///< Full convolution
    FULL_MULTI   ///< Full convolution, with multiple kernels
};

/*!
 * \brief Indicates if the given convolution type has multiple kernels
 * \param type The convolution type to test
 * \return true if the given type has multiple kernels, false otherwise
 */
constexpr bool is_multi(conv_type type){
    return type == conv_type::VALID_MULTI || type == conv_type::SAME_MULTI || type == conv_type::FULL_MULTI;
}

namespace detail {

/*!
 * \brief Select the implementation of the conv of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_default_conv1_impl_new() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_impl::STD;
    }

    static constexpr bool vec = vec_enabled;
    static constexpr bool mkl = is_mkl_enabled;
    static constexpr bool cufft = is_cufft_enabled;

    if(cufft && TT == conv_type::FULL){
        //TODO This should only be done for some sizes
        return etl::conv_impl::FFT_CUFFT;
    } else if(mkl && TT == conv_type::FULL){
        //TODO This should only be done for some sizes
        return etl::conv_impl::FFT_MKL;
    } else if (vec) {
        return etl::conv_impl::VEC;
    } else {
        return etl::conv_impl::STD;
    }
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_conv1_impl_new() {
    auto default_impl = select_default_conv1_impl_new<TT, I, K, C>();

    //COVERAGE_EXCLUDE_BEGIN
    if (local_context().conv_selector.forced) {
        auto forced = local_context().conv_selector.impl;

        switch (forced) {
            //MKL cannot always be used
            case conv_impl::FFT_MKL:
                if (!is_mkl_enabled) {
                    std::cerr << "Forced selection to MKL fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!is_cufft_enabled) {
                    std::cerr << "Forced selection to CUFFT fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //VEC cannot always be used
            case conv_impl::VEC:
                if (!vec_enabled) {
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }
    //COVERAGE_EXCLUDE_END

    return default_impl;
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_default_conv2_impl_new() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_impl::STD;
    }

    static constexpr bool vec = vec_enabled;
    static constexpr bool cufft = is_cufft_enabled;
    static constexpr bool cudnn = is_cudnn_enabled;
    static constexpr bool mkl = is_mkl_enabled;

    // Full has more
    if (TT == conv_type::FULL) {
        if (cufft) {
            return etl::conv_impl::FFT_CUFFT;
        } else if (mkl) {
            return etl::conv_impl::FFT_MKL;
        } else if (cudnn) {
            return etl::conv_impl::CUDNN;
        }
    }

    if (vec) {
        return etl::conv_impl::VEC;
    } else {
        return etl::conv_impl::STD;
    }
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_conv2_impl_new() {
    auto default_impl = select_default_conv2_impl_new<TT, I, K, C>();

    //COVERAGE_EXCLUDE_BEGIN
    if (local_context().conv_selector.forced) {
        auto forced = local_context().conv_selector.impl;

        switch (forced) {
            //VEC cannot always be used
            case conv_impl::VEC:
                if (!vec_enabled) {
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUDNN cannot always be used
            case conv_impl::CUDNN:
                if (!is_cudnn_enabled) {
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //MKL cannot always be used
            case conv_impl::FFT_MKL:
                if (!is_mkl_enabled) {
                    std::cerr << "Forced selection to MKL conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!is_cufft_enabled) {
                    std::cerr << "Forced selection to CUFFT conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }
    //COVERAGE_EXCLUDE_END

    return default_impl;
}

//TODO Remove this once everything is vectorized correctly

/*!
 * \brief Select the implementation of the conv of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_default_conv_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_impl::STD;
    }

    static constexpr bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

    if (avx) {
        return etl::conv_impl::AVX;
    } else if (sse) {
        return etl::conv_impl::SSE;
    } else if(is_cudnn_enabled && (TT == conv_type::VALID || TT == conv_type::FULL) && decay_traits<I>::dimensions() == 2){
        return etl::conv_impl::CUDNN;
    } else {
        return etl::conv_impl::STD;
    }
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_conv_impl() {
    auto default_impl = select_default_conv_impl<TT, I, K, C>();

    //COVERAGE_EXCLUDE_BEGIN
    if (local_context().conv_selector.forced) {
        auto forced = local_context().conv_selector.impl;

        switch (forced) {
            //MKL cannot always be used
            case conv_impl::FFT_MKL:
                if (!is_mkl_enabled) {
                    std::cerr << "Forced selection to MKL fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!is_cufft_enabled) {
                    std::cerr << "Forced selection to CUFFT fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUDNN cannot always be used
            case conv_impl::CUDNN:
                if (!is_cudnn_enabled) {
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //AVX cannot always be used
            case conv_impl::AVX:
                if (!avx_enabled) {
                    std::cerr << "Forced selection to AVX sum implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //SSE cannot always be used
            case conv_impl::SSE:
                if (!sse3_enabled) {
                    std::cerr << "Forced selection to SSE sum implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }
    //COVERAGE_EXCLUDE_END

    return default_impl;
}

/*!
 * \brief Select the implementation of the 4D conv of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv4_impl select_default_conv4_valid_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv4_impl::STD;
    }

    static constexpr bool cudnn = is_cudnn_enabled;
    static constexpr bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;
    static constexpr bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;

    if(cudnn){
        return etl::conv4_impl::CUDNN;
    }

    if (conv4_prefer_blas) {
        if (is_cublas_enabled || is_mkl_enabled) {
            return etl::conv4_impl::BLAS;
        } else if (avx) {
            return etl::conv4_impl::AVX;
        } else if (sse) {
            return etl::conv4_impl::SSE;
        }
    } else {
        if (avx) {
            return etl::conv4_impl::AVX;
        } else if (is_cublas_enabled || is_mkl_enabled) {
            return etl::conv4_impl::BLAS;
        } else if (sse) {
            return etl::conv4_impl::SSE;
        }
    }

    return etl::conv4_impl::BLAS;
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv4_impl select_conv4_valid_impl() {
    if (local_context().conv4_selector.forced) {
        auto forced = local_context().conv4_selector.impl;

        switch (forced) {
            //SSE cannot always be used
            case conv4_impl::SSE:
                if (!sse3_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to SSE conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //AVX cannot always be used
            case conv4_impl::AVX:
                if (!avx_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to AVX conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUDNN cannot always be used
            case conv4_impl::CUDNN:
                if (!is_cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_valid_impl<I, K, C>();
}

/*!
 * \brief Select the implementation of the 4D conv of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv4_impl select_default_conv4_full_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv4_impl::STD;
    }

    static constexpr bool cudnn = is_cudnn_enabled;
    static constexpr bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;
    static constexpr bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;

    if(cudnn){
        return etl::conv4_impl::CUDNN;
    } else if(is_cufft_enabled){
        return etl::conv4_impl::FFT_CUFFT;
    } else if(is_mkl_enabled){
        return etl::conv4_impl::FFT_MKL;
    } else if(avx){
        return etl::conv4_impl::AVX;
    } else if(sse){
        return etl::conv4_impl::SSE;
    } else {
        return etl::conv4_impl::FFT_STD;
    }
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv4_impl select_conv4_full_impl() {
    if (local_context().conv4_selector.forced) {
        auto forced = local_context().conv4_selector.impl;

        switch (forced) {
            //SSE cannot always be used
            case conv4_impl::SSE:
                if (!sse3_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to SSE conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //AVX cannot always be used
            case conv4_impl::AVX:
                if (!avx_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to AVX conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUDNN cannot always be used
            case conv4_impl::CUDNN:
                if (!is_cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUFFT cannot always be used
            case conv4_impl::FFT_CUFFT:
                if (!is_mkl_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to FFT_CUFFT conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //MKL cannot always be used
            case conv4_impl::FFT_MKL:
                if (!is_mkl_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to FFT_MKL conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_full_impl<I, K, C>();
}

/*!
 * \brief Select the implementation of the conv multi of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv_multi_impl select_default_conv_valid_multi() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    static constexpr bool cudnn = is_cudnn_enabled;
    static constexpr bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

    if(cudnn && decay_traits<I>::dimensions() == 2){
        //TODO Should only be used with (very?) large sizes
        return etl::conv_multi_impl::CUDNN;
    }

    if (is_mkl_enabled && conv_valid_fft) {
        return etl::conv_multi_impl::FFT;
    } else if (is_cblas_enabled || is_cublas_enabled) {
        return etl::conv_multi_impl::BLAS;
    }

    if (avx) {
        return etl::conv_multi_impl::AVX;
    } else if (sse) {
        return etl::conv_multi_impl::SSE;
    }

    return etl::conv_multi_impl::STD;
}

/*!
 * \brief Select the implementation of the conv multi of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv_multi_impl select_default_conv_valid_multi_multi_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    static constexpr bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

    if (avx) {
        return etl::conv_multi_impl::AVX;
    } else if (sse) {
        return etl::conv_multi_impl::SSE;
    }

    if (is_cblas_enabled || is_cublas_enabled) {
        return etl::conv_multi_impl::BLAS;
    } else if (is_mkl_enabled && conv_valid_fft) {
        return etl::conv_multi_impl::FFT;
    }

    return etl::conv_multi_impl::STD;
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv_multi_impl select_conv_valid_multi_impl() {
    if (local_context().conv_multi_selector.forced) {
        auto forced = local_context().conv_multi_selector.impl;

        switch (forced) {
            //CUDNN cannot always be used
            case conv_multi_impl::CUDNN:
                if (!is_cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //AVX cannot always be used
            case conv_multi_impl::AVX:
                if (!avx_enabled) {
                    std::cerr << "Forced selection to AVX conv implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

            //SSE cannot always be used
            case conv_multi_impl::SSE:
                if (!sse3_enabled) {
                    std::cerr << "Forced selection to SSE conv implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

                // Although it may be suboptimal the forced selection can
                // always be achieved
            default:
                return forced;
        }
    }

    return select_default_conv_valid_multi<I, K, C>();
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv_multi_impl select_conv_valid_multi_multi_impl() {
    if (local_context().conv_multi_selector.forced) {
        auto forced = local_context().conv_multi_selector.impl;

        switch (forced) {
            //AVX cannot always be used
            case conv_multi_impl::AVX:
                if (!avx_enabled) {
                    std::cerr << "Forced selection to AVX conv implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

            //SSE cannot always be used
            case conv_multi_impl::SSE:
                if (!sse3_enabled) {
                    std::cerr << "Forced selection to SSE conv implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

                // Although it may be suboptimal the forced selection can
                // always be achieved
            default:
                return forced;
        }
    }

    return select_default_conv_valid_multi_multi_impl<I, K, C>();
}

/*!
 * \brief Select the implementation of the conv multi of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv_multi_impl select_default_conv_full_multi_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    static constexpr bool cudnn = is_cudnn_enabled;

    if(cudnn && decay_traits<I>::dimensions() == 2){
        //TODO Should only be used with (very?) large sizes
        return etl::conv_multi_impl::CUDNN;
    }

    if (vectorize_impl && vec_enabled) {
        return etl::conv_multi_impl::VEC;
    }

    return etl::conv_multi_impl::STD;
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv_multi_impl select_conv_full_multi_impl() {
    if (local_context().conv_multi_selector.forced) {
        auto forced = local_context().conv_multi_selector.impl;

        switch (forced) {
            //CUDNN cannot always be used
            case conv_multi_impl::CUDNN:
                if (!is_cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_full_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case conv_multi_impl::VEC:
                if (!vec_enabled) {
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_full_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

                // Although it may be suboptimal the forced selection can
                // always be achieved
            default:
                return forced;
        }
    }

    return select_default_conv_full_multi_impl<I, K, C>();
}

/*!
 * \brief Select the implementation of the conv multi of I and K in C
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv_multi_impl select_default_conv_same_multi_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr order input_order  = decay_traits<I>::storage_order;
    static constexpr order kernel_order = decay_traits<K>::storage_order;
    static constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    if (vec_enabled) {
        return etl::conv_multi_impl::VEC;
    }

    return etl::conv_multi_impl::STD;
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
inline etl::conv_multi_impl select_conv_same_multi_impl() {
    auto default_impl = select_default_conv_same_multi_impl<I, K, C>();

    if (local_context().conv_multi_selector.forced) {
        auto forced = local_context().conv_multi_selector.impl;

        switch (forced) {
            //VEC cannot always be used
            case conv_multi_impl::VEC:
                if (!vec_enabled) {
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;


                // Although it may be suboptimal the forced selection can
                // always be achieved
            default:
                return forced;
        }
    }

    return default_impl;
}

/*!
 * \brief Test if ETL should run in parallel for the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return true to run in paralle, false otherwise
 */
template <typename I, typename K, typename C>
inline bool select_parallel(const I& /*input*/, const K& kernel, C&& conv) {
    if ((is_parallel && !local_context().serial) || local_context().parallel) {
        return size(conv) >= conv1_parallel_threshold_conv && size(kernel) >= conv1_parallel_threshold_kernel;
    } else {
        return false;
    }
}

} //end of namespace detail

} //end of namespace etl
