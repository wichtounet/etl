//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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

namespace detail {

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
constexpr etl::conv4_impl select_default_conv4_valid_impl(bool no_gpu, size_t i1, size_t i2, size_t k1, size_t k2) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv4_impl::STD;
    }

    if(impl::cudnn::conv_possible<I, K, C> && !no_gpu){
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
constexpr etl::conv4_impl select_default_conv4_full_impl(bool no_gpu, size_t k1, size_t k2) {
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
    if(impl::cudnn::conv_possible<I, K, C> && !no_gpu){
        return etl::conv4_impl::CUDNN;
    }

    // CUFFT is generally faster than the other, but anyway in GPU mode, CUDNN should be available
    if(impl::cufft::conv2_possible<I, K, C> && !no_gpu){
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

#ifdef ETL_MANUAL_SELECT

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
                    return select_default_conv4_valid_impl<I, K, C>(local_context().cpu, i1, i2, k1, k2);                                                             // COVERAGE_EXCLUDE_LINE
                }                                                                                                                  // COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
        case etl::conv4_impl::BLAS_MKL:
                if (!cblas_enabled) {                                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>(local_context().cpu, i1, i2, k1, k2);                                                               // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUDNN cannot always be used
        case etl::conv4_impl::CUDNN:
                if (!impl::cudnn::conv_possible<I, K, C> || local_context().cpu) {                                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_valid_impl<I, K, C>(local_context().cpu, i1, i2, k1, k2);                                                               // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_valid_impl<I, K, C>(local_context().cpu, i1, i2, k1, k2);
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
                    return select_default_conv4_full_impl<I, K, C>(local_context().cpu, k1, k2);                                                              // COVERAGE_EXCLUDE_LINE
                }                                                                                                                        // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUDNN cannot always be used
            case etl::conv4_impl::CUDNN:
                if (!impl::cudnn::conv_possible<I, K, C> || local_context().cpu) {                                                                          // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>(local_context().cpu, k1, k2);                                                          // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            //CUFFT cannot always be used
            case etl::conv4_impl::FFT_CUFFT:
                if (!impl::cufft::conv2_possible<I, K, C> || local_context().cpu) {                                                                             // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to FFT_CUFFT conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>(local_context().cpu, k1, k2);                                                              // COVERAGE_EXCLUDE_LINE
                }                                                                                                                        // COVERAGE_EXCLUDE_LINE

                return forced;

            //MKL cannot always be used
            case etl::conv4_impl::FFT_MKL:
                if (!impl::blas::conv2_possible<I, K, C>) {                                                                            // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to FFT_MKL conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_full_impl<I, K, C>(local_context().cpu, k1, k2);                                                            // COVERAGE_EXCLUDE_LINE
                }                                                                                                                      // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_full_impl<I, K, C>(local_context().cpu, k1, k2);
}

#else

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
constexpr etl::conv4_impl select_conv4_valid_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
    return select_default_conv4_valid_impl<I, K, C>(false, i1, i2, k1, k2);
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
constexpr etl::conv4_impl select_conv4_valid_filter_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
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
constexpr etl::conv4_impl select_conv4_valid_back_impl(size_t i1, size_t i2, size_t k1, size_t k2) {
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
constexpr etl::conv4_impl select_conv4_full_impl(size_t k1, size_t k2) {
    return select_default_conv4_full_impl<I, K, C>(false, k1, k2);
}

#endif

} //end of namespace detail
} //end of namespace etl
