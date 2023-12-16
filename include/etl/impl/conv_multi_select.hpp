//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains selectors for convolution implementations.
 */

#pragma once

namespace etl::detail {

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

    if (impl::cudnn::conv_possible<I, K, C> && is_2d<I> && !no_gpu) {
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
                if (!impl::cudnn::conv_possible<I, K, C>) {                                                                          // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);                                            // COVERAGE_EXCLUDE_LINE
                }                                                                                                                    // COVERAGE_EXCLUDE_LINE

                return forced;

            //MKL cannot always be used
            case conv_multi_impl::VALID_FFT_MKL:
                if (!mkl_enabled) {                                                                                                // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to MKL conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);                                          // COVERAGE_EXCLUDE_LINE
                }                                                                                                                  // COVERAGE_EXCLUDE_LINE

                return forced;

            //BLAS cannot always be used
            case conv_multi_impl::BLAS_MKL:
                if (!impl::blas::blas_conv2_possible<I, K, C>) {                                                                    // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu);                                           // COVERAGE_EXCLUDE_LINE
                }                                                                                                                   // COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case conv_multi_impl::VEC:
                if (!vec_enabled || !vectorize_impl) {
                    std::cerr << "Forced selection to VEC conv_valid_multi implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu); // COVERAGE_EXCLUDE_LINE
                }

                return forced;

            case conv_multi_impl::BLAS_VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {
                    std::cerr << "Forced selection to BLAS_VEC conv_valid_multi implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi<I, K, C>(local_context().cpu); // COVERAGE_EXCLUDE_LINE
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
                if (!impl::blas::blas_conv2_possible<I, K, C>) {                                                                    // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_valid_multi_multi_impl<I, K, C>();                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                   // COVERAGE_EXCLUDE_LINE

                return forced;

            //VEC cannot always be used
            case conv_multi_impl::BLAS_VEC:
            case conv_multi_impl::VEC:
                if (!impl::vec::conv2_possible<vector_mode, I, K, C>) {
                    std::cerr << "Forced selection to VEC conv_valid_multi_multi implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_valid_multi_multi_impl<I, K, C>(); // COVERAGE_EXCLUDE_LINE
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

} //end of namespace etl::detail
