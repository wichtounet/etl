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

namespace etl::detail {

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
constexpr etl::conv_impl select_default_conv1_impl_new(bool no_gpu) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    if (TT == conv_type::FULL) {
        if (impl::cufft::conv1_possible<I, K, C> && !no_gpu) {
            return etl::conv_impl::FFT_CUFFT;
        } else if (impl::blas::conv1_possible<I, K, C>) {
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
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
constexpr etl::conv_impl select_default_conv2_impl_new(bool no_gpu) {
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
        if (impl::cudnn::conv_possible<I, K, C> && !no_gpu) {
            return etl::conv_impl::CUDNN;
        } else if (impl::cufft::conv2_possible<I, K, C> && !no_gpu) {
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
 *
 * This does not take the local context into account.
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
constexpr etl::conv_impl select_default_conv_impl(bool no_gpu) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    constexpr order input_order  = decay_traits<I>::storage_order;
    constexpr order kernel_order = decay_traits<K>::storage_order;
    constexpr order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_impl::STD;
    }

    if (impl::cudnn::conv_possible<I, K, C> && (TT == conv_type::VALID || TT == conv_type::FULL) && is_2d<I> && !no_gpu) {
        return etl::conv_impl::CUDNN;
    } else if (impl::vec::conv2_possible<vector_mode, I, K, C>) {
        return etl::conv_impl::VEC;
    } else {
        return etl::conv_impl::STD;
    }
}

#ifdef ETL_MANUAL_SELECT

/*!
 * \brief Select the implementation of the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_conv1_impl_new() {
    auto default_impl = select_default_conv1_impl_new<TT, I, K, C>(local_context().cpu);

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
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_conv2_impl_new() {
    auto default_impl = select_default_conv2_impl_new<TT, I, K, C>(local_context().cpu);

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
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
inline etl::conv_impl select_conv_impl() {
    auto default_impl = select_default_conv_impl<TT, I, K, C>(local_context().cpu);

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

#else

/*!
 * \brief Select the implementation of the conv of I and K in C
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
constexpr etl::conv_impl select_conv1_impl_new() {
    return select_default_conv1_impl_new<TT, I, K, C>(false);
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
constexpr etl::conv_impl select_conv2_impl_new() {
    return select_default_conv2_impl_new<TT, I, K, C>(false);
}

/*!
 * \brief Select the implementation of the conv of I and K in C
 *
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return the implementation to be used
 */
template <conv_type TT, typename I, typename K, typename C>
constexpr etl::conv_impl select_conv_impl() {
    return select_default_conv_impl<TT, I, K, C>(false);
}

#endif

} //end of namespace etl::detail
