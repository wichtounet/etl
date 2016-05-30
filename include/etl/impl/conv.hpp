//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the convolution implementations.
 *
 * The functions are responsible for selecting the most efficient
 * implementation for each case, based on what is available. The selection of
 * parallel versus serial is also done at this level. The implementation
 * functions should never be used directly, only functions of this header can
 * be used directly.
 *
 * Ideas for improvements:
 *  * Parallel dispatching for SSE/AVX implementation is not perfect, it should be done inside the micro kernel main loop
 */

#pragma once

//Include the implementations
#include "etl/impl/std/conv.hpp"
#include "etl/impl/sse/conv.hpp"
#include "etl/impl/avx/conv.hpp"
#include "etl/impl/reduc/conv_mmul.hpp"
#include "etl/impl/reduc/conv_multi.hpp"
#include "etl/impl/cudnn/conv.hpp"

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
inline etl::conv_impl select_default_conv_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr const order input_order  = decay_traits<I>::storage_order;
    static constexpr const order kernel_order = decay_traits<K>::storage_order;
    static constexpr const order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_impl::STD;
    }

    static constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;
    static constexpr const bool cudnn = is_cudnn_enabled;

    if(cudnn && TT == conv_type::VALID && decay_traits<I>::dimensions() == 2){
        //TODO Should only be used with (very?) large sizes
        return etl::conv_impl::CUDNN;
    }

    if (avx) {
        return etl::conv_impl::AVX;
    } else if (sse) {
        return etl::conv_impl::SSE;
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
    if (local_context().conv_selector.forced) {
        auto forced = local_context().conv_selector.impl;

        switch (forced) {
            //CUDNN cannot always be used
            case conv_impl::CUDNN:
                if (!is_cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_impl<TT, I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //AVX cannot always be used
            case conv_impl::AVX:
                if (!avx_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to AVX sum implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_impl<TT, I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //SSE cannot always be used
            case conv_impl::SSE:
                if (!sse3_enabled) {                                                                                              //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to SSE sum implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_conv_impl<TT, I, K, C>();                                                                   //COVERAGE_EXCLUDE_LINE
                }                                                                                                                 //COVERAGE_EXCLUDE_LINE

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_conv_impl<TT, I, K, C>();
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
inline etl::conv4_impl select_default_conv4_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr const order input_order  = decay_traits<I>::storage_order;
    static constexpr const order kernel_order = decay_traits<K>::storage_order;
    static constexpr const order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv4_impl::STD;
    }

    static constexpr const bool cudnn = is_cudnn_enabled;

    if(cudnn){
        return etl::conv4_impl::CUDNN;
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
inline etl::conv4_impl select_conv4_impl() {
    if (local_context().conv4_selector.forced) {
        auto forced = local_context().conv4_selector.impl;

        switch (forced) {
            //CUDNN cannot always be used
            case conv4_impl::CUDNN:
                if (!is_cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            default:
                return forced;
        }
    }

    return select_default_conv4_impl<I, K, C>();
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
inline etl::conv_multi_impl select_default_conv_multi_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    static constexpr const order input_order  = decay_traits<I>::storage_order;
    static constexpr const order kernel_order = decay_traits<K>::storage_order;
    static constexpr const order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    static constexpr const bool cudnn = is_cudnn_enabled;

    if(cudnn && decay_traits<I>::dimensions() == 2){
        //TODO Should only be used with (very?) large sizes
        return etl::conv_multi_impl::CUDNN;
    }

    if (is_mkl_enabled && conv_valid_fft) {
        return etl::conv_multi_impl::FFT;
    } else if (is_cblas_enabled || is_cublas_enabled) {
        return etl::conv_multi_impl::BLAS;
    } else {
        return etl::conv_multi_impl::STD;
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
inline etl::conv_multi_impl select_conv_multi_impl() {
    if (local_context().conv_multi_selector.forced) {
        auto forced = local_context().conv_multi_selector.impl;

        switch (forced) {
            //CUDNN cannot always be used
            case conv_multi_impl::CUDNN:
                if (!is_cudnn_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

                // Although it may be suboptimal the forced selection can
                // always be achieved
            default:
                return forced;
        }
    }

    return select_default_conv_multi_impl<I, K, C>();
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

    static constexpr const order input_order  = decay_traits<I>::storage_order;
    static constexpr const order kernel_order = decay_traits<K>::storage_order;
    static constexpr const order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    static constexpr const bool cudnn = is_cudnn_enabled;

    if(cudnn && decay_traits<I>::dimensions() == 2){
        //TODO Should only be used with (very?) large sizes
        return etl::conv_multi_impl::CUDNN;
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

                // Although it may be suboptimal the forced selection can
                // always be achieved
            default:
                return forced;
        }
    }

    return select_default_conv_full_multi_impl<I, K, C>();
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

/*!
 * \brief The functor impl for 1D full conv
 */
struct conv1_full_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl    = select_conv_impl<conv_type::FULL, I, K, C>();
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::avx::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::sse::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::standard::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 1D same conv
 */
struct conv1_same_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl    = select_conv_impl<conv_type::SAME, I, K, C>();
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::avx::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::sse::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::standard::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 1D valid conv
 */
struct conv1_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl    = select_conv_impl<conv_type::VALID, I, K, C>();
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::avx::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::sse::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last) {
                impl::standard::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

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
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::FULL, I, K, C>();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_full(input, kernel, conv);
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
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::SAME, I, K, C>();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_same(input, kernel, conv);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_same(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_same(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
struct conv2_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_valid(input, kernel, conv);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_valid(input, kernel, conv);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
struct conv2_valid_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_valid_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_valid_flipped(input, kernel, conv);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct conv4_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid(input, kernel, conv);
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
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_full(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct conv4_valid_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_valid_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_flipped(input, kernel, conv);
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
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_full_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_full_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
struct conv2_valid_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::BLAS) {
            impl::reduc::blas_conv2_valid_multi(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::FFT) {
            impl::reduc::fft_conv2_valid_multi(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_valid_multi(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
struct conv2_valid_multi_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::BLAS) {
            impl::reduc::blas_conv2_valid_multi_flipped(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::FFT) {
            impl::reduc::fft_conv2_valid_multi_flipped(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_valid_multi_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
struct conv3_valid_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        impl::standard::conv3_valid_multi(input, kernel, conv);
    }
};

/*!
 * \brief The functor impl for 2D valid conv, with multiple kernels
 */
struct conv3_valid_multi_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::BLAS) {
            cpp_unreachable("Unimplemented");
        } else if (impl == etl::conv_multi_impl::FFT) {
            impl::reduc::fft_conv3_valid_multi_flipped(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::STD) {
            impl::standard::conv3_valid_multi_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D+ conv.
 */
struct conv_deep_valid_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if(decay_traits<I>::dimensions() == 3)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            conv(i) = conv_2d_valid(input(i), kernel(i));
        }
    }

    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if((decay_traits<I>::dimensions() > 3))>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            apply(input(i), kernel(i), conv(i));
        }
    }
};
/*!
 * \brief The functor impl for 2D+ conv.
 */
struct conv_deep_same_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if(decay_traits<I>::dimensions() == 3)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            conv(i) = conv_2d_same(input(i), kernel(i));
        }
    }

    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if((decay_traits<I>::dimensions() > 3))>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            apply(input(i), kernel(i), conv(i));
        }
    }
};

/*!
 * \brief The functor impl for 2D+ conv.
 */
struct conv_deep_full_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if(decay_traits<I>::dimensions() == 3)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            conv(i) = conv_2d_full(input(i), kernel(i));
        }
    }

    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if((decay_traits<I>::dimensions() > 3))>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (std::size_t i = 0; i < dim<0>(input); ++i) {
            apply(input(i), kernel(i), conv(i));
        }
    }
};

/*!
 * \brief The functor impl for 2D full conv, with multiple kernels
 */
struct conv2_full_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv_full_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_full_multi(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_full_multi(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D full conv, with multiple kernels
 */
struct conv2_full_multi_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv_full_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_full_multi_flipped(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_full_multi_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }
};

/*!
 * \brief The functor impl for 2D same conv, with multiple kernels
 */
struct conv2_same_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        impl::standard::conv2_same_multi(input, kernel, conv);
    }
};

} //end of namespace detail

} //end of namespace etl
