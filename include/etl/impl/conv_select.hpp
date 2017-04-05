//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

    static constexpr bool vec = vectorize_impl && vec_enabled;
    static constexpr bool mkl = mkl_enabled;
    static constexpr bool cufft = cufft_enabled;

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
                if (!mkl_enabled) {
                    std::cerr << "Forced selection to MKL fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!cufft_enabled) {
                    std::cerr << "Forced selection to CUFFT fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //VEC cannot always be used
            case conv_impl::VEC:
                if (!vec_enabled || !vectorize_impl) {
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

    // Full has more options
    if (TT == conv_type::FULL) {
        if (cufft_enabled) {
            return etl::conv_impl::FFT_CUFFT;
        } else if (mkl_enabled) {
            return etl::conv_impl::FFT_MKL;
        } else if (cudnn_enabled) {
            return etl::conv_impl::CUDNN;
        }
    }

    if (vectorize_impl && vec_enabled) {
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
                if (!vec_enabled || !vectorize_impl) {
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUDNN cannot always be used
            case conv_impl::CUDNN:
                if (!cudnn_enabled) {
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //MKL cannot always be used
            case conv_impl::FFT_MKL:
                if (!mkl_enabled) {
                    std::cerr << "Forced selection to MKL conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!cufft_enabled) {
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

    if (vec_enabled && vectorize_impl) {
        return etl::conv_impl::VEC;
    } else if(cudnn_enabled && (TT == conv_type::VALID || TT == conv_type::FULL) && decay_traits<I>::dimensions() == 2){
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
                if (!mkl_enabled) {
                    std::cerr << "Forced selection to MKL fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!cufft_enabled) {
                    std::cerr << "Forced selection to CUFFT fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUDNN cannot always be used
            case conv_impl::CUDNN:
                if (!cudnn_enabled) {
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //VEC cannot always be used
            case conv_impl::VEC:
                if (!vec_enabled || !vectorize_impl) {
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

    static constexpr bool cudnn = cudnn_enabled;

    if(cudnn){
        return etl::conv4_impl::CUDNN;
    }

    if (conv4_prefer_blas) {
        if (cublas_enabled || mkl_enabled) {
            return etl::conv4_impl::BLAS_MKL;
        } else if (vec_enabled && vectorize_impl) {
            return etl::conv4_impl::BLAS_VEC;
        }
    } else {
        if (vec_enabled && vectorize_impl) {
            return etl::conv4_impl::VEC;
        } else if (cublas_enabled || mkl_enabled) {
            return etl::conv4_impl::BLAS_MKL;
        }
    }

    return etl::conv4_impl::STD;
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
            //VEC cannot always be used
            case conv4_impl::BLAS_VEC:
            case conv4_impl::VEC:
                if (!vec_enabled || !vectorize_impl) {                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>();                                                             // COVERAGE_EXCLUDE_LINE
                }                                                                                                                  // COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
            case conv4_impl::BLAS_MKL:
                if (!cblas_enabled) {                                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>();                                                               // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUDNN cannot always be used
            case conv4_impl::CUDNN:
                if (!cudnn_enabled) {                                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>();                                                               // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

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

    static constexpr bool cudnn = cudnn_enabled;

    // CUDNN is always faster than the others
    if(cudnn){
        return etl::conv4_impl::CUDNN;
    }

    // CUFFT is generally faster than the other, but anyway in GPU mode, CUDNN should be available
    if(cufft_enabled){
        return etl::conv4_impl::FFT_CUFFT;
    }

    // MKL is generally faster than VEC
    // This could be improved for small batch size where VEC is interesting
    if (mkl_enabled) {
        return etl::conv4_impl::FFT_MKL;
    }

    // If possible, use vectorized implementations
    if(vectorize_impl && vec_enabled){
        return etl::conv4_impl::VEC;
    }

    // If nothing else if available
    return etl::conv4_impl::FFT_STD;
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
            //VEC cannot always be used
            case conv4_impl::VEC:
                if (!sse3_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUDNN cannot always be used
            case conv4_impl::CUDNN:
                if (!cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUFFT cannot always be used
            case conv4_impl::FFT_CUFFT:
                if (!cufft_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to FFT_CUFFT conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //MKL cannot always be used
            case conv4_impl::FFT_MKL:
                if (!mkl_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
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

    static constexpr bool cudnn = cudnn_enabled;

    if(cudnn && decay_traits<I>::dimensions() == 2){
        //TODO Should only be used with (very?) large sizes
        return etl::conv_multi_impl::CUDNN;
    }

    if (mkl_enabled && conv_valid_fft) {
        return etl::conv_multi_impl::VALID_FFT_MKL;
    } else if (cblas_enabled) {
        return etl::conv_multi_impl::BLAS_MKL;
    }

    if (vectorize_impl && vec_enabled) {
        // TODO When to use VEC and BLAS_VEC ?
        return etl::conv_multi_impl::BLAS_VEC;
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

    if (vectorize_impl && vec_enabled) {
        // TODO When to use VEC and BLAS_VEC ?
        return etl::conv_multi_impl::BLAS_VEC;
    }

    if (cblas_enabled) {
        return etl::conv_multi_impl::BLAS_MKL;
    } else if (mkl_enabled && conv_valid_fft) {
        return etl::conv_multi_impl::VALID_FFT_MKL;
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
                if (!cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //MKL cannot always be used
            case conv_multi_impl::VALID_FFT_MKL:
                if (!mkl_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to MKL conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
            case conv_multi_impl::BLAS_MKL:
                if (!cblas_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case conv_multi_impl::VEC:
            case conv_multi_impl::BLAS_VEC:
                if (!vec_enabled || !vectorize_impl) {
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl;
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
            //BLAS cannot always be used
            case conv_multi_impl::BLAS_MKL:
                if (!cblas_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case conv_multi_impl::BLAS_VEC:
            case conv_multi_impl::VEC:
                if (!vec_enabled || !vectorize_impl) {
                    std::cerr << "Forced selection to VEC conv implementation, but not possible for this expression" << std::endl;
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

    static constexpr bool cudnn = cudnn_enabled;

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
                if (!cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_full_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case conv_multi_impl::VEC:
                if (!vec_enabled || !vectorize_impl) {
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

    if (vec_enabled && vectorize_impl) {
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
                if (!vec_enabled || !vectorize_impl) {
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
    if ((is_parallel && !local_context().serial) || (parallel_support && local_context().parallel)) {
        return size(conv) >= conv1_parallel_threshold_conv && size(kernel) >= conv1_parallel_threshold_kernel;
    } else {
        return false;
    }
}

} //end of namespace detail

} //end of namespace etl
