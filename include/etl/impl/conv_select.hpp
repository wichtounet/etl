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
etl::conv_impl select_default_conv1_impl_new() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    if(TT == conv_type::FULL){
        if(impl::cufft::conv1_possible<I, K, C> && !local_context().cpu){
            //TODO This should only be done for some sizes
            return etl::conv_impl::FFT_CUFFT;
        } else if(impl::blas::conv1_possible<I, K, C>){
            //TODO This should only be done for some sizes
            return etl::conv_impl::FFT_MKL;
        }
    }

    if (impl::vec::conv1_possible<vector_mode, I, K, C>) {
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
                if (!impl::blas::conv1_possible<I, K, C>) {
                    std::cerr << "Forced selection to MKL fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!impl::cufft::conv1_possible<I, K, C> || local_context().cpu) {
                    std::cerr << "Forced selection to CUFFT fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //VEC cannot always be used
            case conv_impl::VEC:
                if (!impl::vec::conv1_possible<vector_mode, I, K, C>) {
                    std::cerr << "Forced selection to VEC conv1 implementation, but not possible for this expression" << std::endl;
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

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_impl::STD;
    }

    // Full has more options
    if (TT == conv_type::FULL) {
        if (impl::cudnn::conv_possible<I, K, C> && !local_context().cpu) {
            return etl::conv_impl::CUDNN;
        } else if (impl::cufft::conv2_possible<I, K, C> && !local_context().cpu) {
            return etl::conv_impl::FFT_CUFFT;
        } else if (impl::blas::conv2_possible<I, K, C>) {
            return etl::conv_impl::FFT_MKL;
        }
    }

    if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
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
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {
                    std::cerr << "Forced selection to VEC conv2 implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUDNN cannot always be used
            case conv_impl::CUDNN:
                if (!impl::cudnn::conv_possible<I, K, C> || local_context().cpu) {
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //MKL cannot always be used
            case conv_impl::FFT_MKL:
                if (!impl::blas::conv2_possible<I, K, C>) {
                    std::cerr << "Forced selection to MKL conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!impl::cufft::conv2_possible<I, K, C> || local_context().cpu) {
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
constexpr etl::conv_impl select_default_conv_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_impl::STD;
    }

    if(impl::cudnn::conv_possible<I, K, C> && (TT == conv_type::VALID || TT == conv_type::FULL) && is_2d<I> && !local_context().cpu){
        return etl::conv_impl::CUDNN;
    } else if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
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
inline etl::conv_impl select_conv_impl() {
    auto default_impl = select_default_conv_impl<TT, I, K, C>();

    //COVERAGE_EXCLUDE_BEGIN
    if (local_context().conv_selector.forced) {
        auto forced = local_context().conv_selector.impl;

        switch (forced) {
            //MKL cannot always be used
            case conv_impl::FFT_MKL:
                if (!impl::blas::conv2_possible<I, K, C>) {
                    std::cerr << "Forced selection to MKL fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUFFT cannot always be used
            case conv_impl::FFT_CUFFT:
                if (!impl::cufft::conv2_possible<I, K, C> || local_context().cpu) {
                    std::cerr << "Forced selection to CUFFT fft_conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //CUDNN cannot always be used
            case conv_impl::CUDNN:
                if (!impl::cudnn::conv_possible<I, K, C> || local_context().cpu) {
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl;
                    return default_impl;
                }

                return forced;

            //VEC cannot always be used
            case conv_impl::VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {
                    std::cerr << "Forced selection to VEC conv2 implementation, but not possible for this expression" << std::endl;
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
constexpr etl::conv4_impl select_default_conv4_valid_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv4_impl::STD;
    }

    if(impl::cudnn::conv_possible<I, K, C> && !local_context().cpu){
        return etl::conv4_impl::CUDNN;
    }

    // Small kernels
    if(k1 == k2 && k1 <= 5){
        if(impl::vec::conv2_possible<vector_mode, I, K, C> && i1 == i2 && i1 > 100){
            return etl::conv4_impl::VEC;
        } else {
            if (cblas_enabled) {
                return etl::conv4_impl::BLAS_MKL;
            } else if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
                return etl::conv4_impl::BLAS_VEC;
            }
        }
    }

    if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
        return etl::conv4_impl::VEC;
    } else if (cblas_enabled) {
        return etl::conv4_impl::BLAS_MKL;
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
inline etl::conv4_impl select_conv4_valid_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
    if (local_context().conv4_selector.forced) {
        auto forced = local_context().conv4_selector.impl;

        switch (forced) {
            //VEC cannot always be used
        case etl::conv4_impl::BLAS_VEC:
        case etl::conv4_impl::VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC conv4 implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>(i1, i2, k1, k2);                                                             // COVERAGE_EXCLUDE_LINE
                }                                                                                                                  // COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
        case etl::conv4_impl::BLAS_MKL:
                if (!cblas_enabled) {                                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>(i1, i2, k1, k2);                                                               // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUDNN cannot always be used
        case etl::conv4_impl::CUDNN:
                if (!impl::cudnn::conv_possible<I, K, C> || local_context().cpu) {                                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>(i1, i2, k1, k2);                                                               // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_valid_impl<I, K, C>(i1, i2, k1, k2);
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
constexpr etl::conv4_impl select_default_conv4_valid_filter_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv4_impl::STD;
    }

    // Small kernels
    if(k1 == k2 && k1 <= 5){
        if(impl::vec::conv2_possible<vector_mode, I, K, C> && i1 == i2 && i1 > 100){
            return etl::conv4_impl::VEC;
        } else {
            if (cblas_enabled) {
                return etl::conv4_impl::BLAS_MKL;
            } else if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
                return etl::conv4_impl::BLAS_VEC;
            }
        }
    }

    if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
        return etl::conv4_impl::VEC;
    } else if (cblas_enabled) {
        return etl::conv4_impl::BLAS_MKL;
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
inline etl::conv4_impl select_conv4_valid_filter_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
    if (local_context().conv4_selector.forced) {
        auto forced = local_context().conv4_selector.impl;

        switch (forced) {
            //VEC cannot always be used
        case etl::conv4_impl::BLAS_VEC:
        case etl::conv4_impl::VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC conv4_valid_filter implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_filter_impl<I, K, C>(i1, i2, k1, k2);                                        // COVERAGE_EXCLUDE_LINE
                }                                                                                                                  // COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
        case etl::conv4_impl::BLAS_MKL:
                if (!cblas_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_filter_impl<I, K, C>(i1, i2, k1, k2);                                         // COVERAGE_EXCLUDE_LINE
                }                                                                                                                   // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_valid_filter_impl<I, K, C>(i1, i2, k1, k2);
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
constexpr etl::conv4_impl select_default_conv4_valid_back_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv4_impl::STD;
    }

    // Small kernels
    if(k1 == k2 && k1 <= 5){
        if(i1 == i2 && i1 > 100){
            if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
                return etl::conv4_impl::VEC;
            }
        } else {
            if (cblas_enabled) {
                return etl::conv4_impl::BLAS_MKL;
            } else if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
                return etl::conv4_impl::BLAS_VEC;
            }
        }
    }

    if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
        return etl::conv4_impl::VEC;
    } else if (cblas_enabled) {
        return etl::conv4_impl::BLAS_MKL;
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
inline etl::conv4_impl select_conv4_valid_back_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
    if (local_context().conv4_selector.forced) {
        auto forced = local_context().conv4_selector.impl;

        switch (forced) {
            //VEC cannot always be used
        case etl::conv4_impl::BLAS_VEC:
        case etl::conv4_impl::VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC conv4_valid_back implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_back_impl<I, K, C>(i1, i2, k1, k2);                                                             // COVERAGE_EXCLUDE_LINE
                }                                                                                                                  // COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
        case etl::conv4_impl::BLAS_MKL:
                if (!cblas_enabled) {                                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_back_impl<I, K, C>(i1, i2, k1, k2);                                                               // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_valid_back_impl<I, K, C>(i1, i2, k1, k2);
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
etl::conv4_impl select_default_conv4_full_impl(size_t k1, size_t k2) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv4_impl::STD;
    }

    // CUDNN is always faster than the others
    if(impl::cudnn::conv_possible<I, K, C> && !local_context().cpu){
        return etl::conv4_impl::CUDNN;
    }

    // CUFFT is generally faster than the other, but anyway in GPU mode, CUDNN should be available
    if(impl::cufft::conv2_possible<I, K, C> && !local_context().cpu){
        return etl::conv4_impl::FFT_CUFFT;
    }

    // MKL is generally faster than VEC
    // This could be improved for small batch size where VEC is interesting
    if (impl::blas::conv2_possible<I, K, C>) {
        if(impl::vec::conv2_possible<vector_mode, I, K, C> && k1 == k2 && (k2 == 3 || k2 == 5)){
            return etl::conv4_impl::VEC;
        }

        return etl::conv4_impl::FFT_MKL;
    }

    // If possible, use vectorized implementations
    if(impl::vec::conv2_possible<vector_mode, I, K, C>){
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
inline etl::conv4_impl select_conv4_full_impl(size_t k1, size_t k2) {
    if (local_context().conv4_selector.forced) {
        auto forced = local_context().conv4_selector.impl;

        switch (forced) {
            //VEC cannot always be used
            case etl::conv4_impl::VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {                                                                  // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC conv4_full implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>(k1, k2);                                                              // COVERAGE_EXCLUDE_LINE
                }                                                                                                                        // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUDNN cannot always be used
            case etl::conv4_impl::CUDNN:
                if (!impl::cudnn::conv_possible<I, K, C> || local_context().cpu) {                                                                          // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>(k1, k2);                                                          // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUFFT cannot always be used
            case etl::conv4_impl::FFT_CUFFT:
                if (!impl::cufft::conv2_possible<I, K, C> || local_context().cpu) {                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to FFT_CUFFT conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>(k1, k2);                                                              // COVERAGE_EXCLUDE_LINE
                }                                                                                                                        // COVERAGE_EXCLUDE_LINE

                return forced;

            //MKL cannot always be used
            case etl::conv4_impl::FFT_MKL:
                if (!impl::blas::conv2_possible<I, K, C>) {                                                                            // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to FFT_MKL conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>(k1, k2);                                                            // COVERAGE_EXCLUDE_LINE
                }                                                                                                                      // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_full_impl<I, K, C>(k1, k2);
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
constexpr etl::conv_multi_impl select_default_conv_valid_multi(bool no_gpu) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    if(impl::cudnn::conv_possible<I, K, C> && is_2d<I> && !no_gpu){
        //TODO Should only be used with (very?) large sizes
        return etl::conv_multi_impl::CUDNN;
    }

    if (mkl_enabled && conv_valid_fft) {
        return etl::conv_multi_impl::VALID_FFT_MKL;
    } else if (impl::blas::blas_conv2_possible<I, K, C>) {
        return etl::conv_multi_impl::BLAS_MKL;
    }

    if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
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
constexpr etl::conv_multi_impl select_default_conv_valid_multi_multi_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
        // TODO When to use VEC and BLAS_VEC ?
        return etl::conv_multi_impl::BLAS_VEC;
    }

    if (impl::blas::blas_conv2_possible<I, K, C>) {
        return etl::conv_multi_impl::BLAS_MKL;
    } else if (mkl_enabled && conv_valid_fft) {
        return etl::conv_multi_impl::VALID_FFT_MKL;
    }

    return etl::conv_multi_impl::STD;
}

#ifdef ETL_MANUAL_SELECT

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
etl::conv_multi_impl select_conv_valid_multi_impl() {
    if (local_context().conv_multi_selector.forced) {
        auto forced = local_context().conv_multi_selector.impl;

        switch (forced) {
            //CUDNN cannot always be used
            case conv_multi_impl::CUDNN:
                if (!impl::cudnn::conv_possible<I, K, C>) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //MKL cannot always be used
            case conv_multi_impl::VALID_FFT_MKL:
                if (!mkl_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to MKL conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
            case conv_multi_impl::BLAS_MKL:
                if (!impl::blas::blas_conv2_possible<I, K, C>) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case conv_multi_impl::VEC:
                if (!vec_enabled || !vectorize_impl) {
                    std::cerr << "Forced selection to VEC conv_valid_multi implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

            case conv_multi_impl::BLAS_VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {
                    std::cerr << "Forced selection to BLAS_VEC conv_valid_multi implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

                // Although it may be suboptimal the forced selection can
                // always be achieved
            default:
                return forced;
        }
    }

    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
etl::conv_multi_impl select_conv_valid_multi_multi_impl() {
    if (local_context().conv_multi_selector.forced) {
        auto forced = local_context().conv_multi_selector.impl;

        switch (forced) {
            //BLAS cannot always be used
            case conv_multi_impl::BLAS_MKL:
                if (!impl::blas::blas_conv2_possible<I, K, C>) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case conv_multi_impl::BLAS_VEC:
            case conv_multi_impl::VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {
                    std::cerr << "Forced selection to VEC conv_valid_multi_multi implementation, but not possible for this expression" << std::endl;
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

#else

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
constexpr etl::conv_multi_impl select_conv_valid_multi_impl() {
    return select_default_conv_valid_multi<I, K, C>(false);
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <typename I, typename K, typename C>
constexpr etl::conv_multi_impl select_conv_valid_multi_multi_impl() {
    return select_default_conv_valid_multi_multi_impl<I, K, C>();
}

#endif

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
