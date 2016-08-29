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

//TODO conv_X_multi should follow the same principle as the other convolutions
//and have dedicated SSE/AVX/... kernels

#pragma once

//Include the implementations
#include "etl/impl/std/conv.hpp"
#include "etl/impl/sse/conv.hpp"
#include "etl/impl/avx/conv.hpp"
#include "etl/impl/reduc/conv_multi.hpp"
#include "etl/impl/cudnn/conv.hpp"

namespace etl {

/*!
 * \brief Constexpr min between two values
 */
template<std::size_t A, std::size_t B>
struct c_min {
    static constexpr size_t value = A < B ? A : B; ///< The resulting value
};

/*!
 * \brief Return safely the D dimension of E.
 *
 * Once C++ offers a real static_if, this needs to be removed
 *
 * \return the Dth dimension of E
 */
template <std::size_t D, typename E>
constexpr std::size_t safe_dim() noexcept {
    return decay_traits<E>::template dim<c_min<D, etl::dimensions<E>() - 1>::value>();
}

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
    static constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;
    static constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;

    if(cudnn){
        return etl::conv4_impl::CUDNN;
    } else if(avx){
        return etl::conv4_impl::AVX;
    } else if(sse){
        return etl::conv4_impl::SSE;
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
            //SSE cannot always be used
            case conv4_impl::SSE:
                if (!sse3_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to SSE conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

            //AVX cannot always be used
            case conv4_impl::AVX:
                if (!avx_enabled) {                                                                                               // COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to AVX conv implementation, but not possible for this expression" << std::endl; // COVERAGE_EXCLUDE_LINE
                    return select_default_conv4_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }                                                                                                                 // COVERAGE_EXCLUDE_LINE

                return forced;

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
inline etl::conv_multi_impl select_default_conv_valid_multi() {
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
    static constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

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
    static constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

    if(cudnn && decay_traits<I>::dimensions() == 2){
        //TODO Should only be used with (very?) large sizes
        return etl::conv_multi_impl::CUDNN;
    }

    if (avx) {
        return etl::conv_multi_impl::AVX;
    } else if (sse) {
        return etl::conv_multi_impl::SSE;
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

            //AVX cannot always be used
            case conv_multi_impl::AVX:
                if (!avx_enabled) {
                    std::cerr << "Forced selection to AVX conv implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_full_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

            //SSE cannot always be used
            case conv_multi_impl::SSE:
                if (!sse3_enabled) {
                    std::cerr << "Forced selection to SSE conv implementation, but not possible for this expression" << std::endl;
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

    static constexpr const order input_order  = decay_traits<I>::storage_order;
    static constexpr const order kernel_order = decay_traits<K>::storage_order;
    static constexpr const order output_order = decay_traits<C>::storage_order;

    //Only the standard implementation is able to handle column major
    if (input_order == order::ColumnMajor || kernel_order == order::ColumnMajor || output_order == order::ColumnMajor) {
        return etl::conv_multi_impl::STD;
    }

    static constexpr const bool sse = vectorize_impl && vector_mode == vector_mode_t::SSE3;
    static constexpr const bool avx = vectorize_impl && vector_mode == vector_mode_t::AVX;

    if (avx) {
        return etl::conv_multi_impl::AVX;
    } else if (sse) {
        return etl::conv_multi_impl::SSE;
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
    if (local_context().conv_multi_selector.forced) {
        auto forced = local_context().conv_multi_selector.impl;

        switch (forced) {
            //AVX cannot always be used
            case conv_multi_impl::AVX:
                if (!avx_enabled) {
                    std::cerr << "Forced selection to AVX conv implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_same_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

            //SSE cannot always be used
            case conv_multi_impl::SSE:
                if (!sse3_enabled) {
                    std::cerr << "Forced selection to SSE conv implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_same_multi_impl<I, K, C>();                                                                   // COVERAGE_EXCLUDE_LINE
                }

                return forced;

                // Although it may be suboptimal the forced selection can
                // always be achieved
            default:
                return forced;
        }
    }

    return select_default_conv_same_multi_impl<I, K, C>();
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
        } else if (impl == etl::conv_impl::FFT_STD) {
            impl::standard::conv1_full_fft(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_MKL) {
            impl::blas::conv1_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_CUFFT) {
            impl::cufft::conv1_full(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv1_full";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 1, "Invalid number of dimensions for input of conv1_full");
        static_assert(etl::dimensions<K>() == 1, "Invalid number of dimensions for kernel of conv1_full");
        static_assert(etl::dimensions<C>() == 1, "Invalid number of dimensions for conv of conv1_full");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0) + etl::dim(kernel, 0) - 1, "Invalid dimensions for conv1_full");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv1_full");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 1, "Invalid number of dimensions for input of conv1_full");
        static_assert(etl::dimensions<K>() == 1, "Invalid number of dimensions for kernel of conv1_full");
        static_assert(etl::dimensions<C>() == 1, "Invalid number of dimensions for conv of conv1_full");

        static_assert(etl::dim<0, C>() == etl::dim<0, I>() + etl::dim<0, K>() - 1, "Invalid dimensions for conv1_full");
        static_assert(etl::dim<0, I>() >= etl::dim<0, K>(), "Invalid dimensions for conv1_full");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d == 0, "Invalid dimensions access");

        return etl::dim(input, d) + etl::dim(kernel, d) - 1;
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D == 0, "Invalid dimension access");

        return etl::dim<0, I>() + etl::dim<0, K>() - 1;
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

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv1_same";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 1, "Invalid number of dimensions for input of conv1_same");
        static_assert(etl::dimensions<K>() == 1, "Invalid number of dimensions for kernel of conv1_same");
        static_assert(etl::dimensions<C>() == 1, "Invalid number of dimensions for conv of conv1_same");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0), "Invalid dimensions for conv1_same");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv1_same");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 1, "Invalid number of dimensions for input of conv1_same");
        static_assert(etl::dimensions<K>() == 1, "Invalid number of dimensions for kernel of conv1_same");
        static_assert(etl::dimensions<C>() == 1, "Invalid number of dimensions for conv of conv1_same");

        static_assert(etl::dim<0, C>() == etl::dim<0, I>(), "Invalid dimensions for conv1_same");
        static_assert(etl::dim<0, I>() >= etl::dim<0, K>(), "Invalid dimensions for conv1_same");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d == 0, "Invalid dimensions access");
        cpp_unused(kernel);

        return etl::dim(input, d);
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D == 0, "Invalid dimension access");

        return etl::dim<0, I>();
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

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv1_valid";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 1, "Invalid number of dimensions for input of conv1_valid");
        static_assert(etl::dimensions<K>() == 1, "Invalid number of dimensions for kernel of conv1_valid");
        static_assert(etl::dimensions<C>() == 1, "Invalid number of dimensions for conv of conv1_valid");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0) - etl::dim(kernel, 0) + 1, "Invalid dimensions for conv1_valid");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv1_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 1, "Invalid number of dimensions for input of conv1_valid");
        static_assert(etl::dimensions<K>() == 1, "Invalid number of dimensions for kernel of conv1_valid");
        static_assert(etl::dimensions<C>() == 1, "Invalid number of dimensions for conv of conv1_valid");

        static_assert(etl::dim<0, C>() == etl::dim<0, I>() - etl::dim<0, K>() + 1, "Invalid dimensions for conv1_valid");
        static_assert(etl::dim<0, I>() >= etl::dim<0, K>(), "Invalid dimensions for conv1_valid");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d == 0, "Invalid dimensions access");

        return etl::dim(input, d) - etl::dim(kernel, d) + 1;
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D == 0, "Invalid dimension access");

        return etl::dim<0, I>() - etl::dim<0, K>() + 1;
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

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_full(i, k, c);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_full(i, k, c);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_full(i, k, c);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_STD) {
            impl::standard::conv2_full_fft(i, k, c);
        } else if (impl == etl::conv_impl::FFT_MKL) {
            impl::blas::conv2_full(i, k, c);
        } else if (impl == etl::conv_impl::FFT_CUFFT) {
            impl::cufft::conv2_full(i, k, c);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_full";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_full");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_full");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_full");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0) + etl::dim(kernel, 0) - 1, "Invalid dimensions for conv2_full");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 1) + etl::dim(kernel, 1) - 1, "Invalid dimensions for conv2_full");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv2_full");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_full");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_full");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_full");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_full");

        static_assert(etl::dim<0,C>() == etl::dim<0,I>() + etl::dim<0,K>() - 1, "Invalid dimensions for conv2_full");
        static_assert(etl::dim<1,C>() == etl::dim<1,I>() + etl::dim<1,K>() - 1, "Invalid dimensions for conv2_full");
        static_assert(etl::dim<0,I>() >= etl::dim<0,K>(), "Invalid dimensions for conv2_full");
        static_assert(etl::dim<1,I>() >= etl::dim<1,K>(), "Invalid dimensions for conv2_full");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 2, "Invalid dimensions access");

        return etl::dim(input, d) + etl::dim(kernel, d) - 1;
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 2, "Invalid dimension access");

        return etl::dim<D, I>() + etl::dim<D, K>() - 1;
    }
};

/*!
 * \brief The functor impl for 2D full conv
 */
struct conv2_full_flipped_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::FULL, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_full_flipped(i, k, c);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_full_flipped(i, k, c);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_full_flipped(i, k, c);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_full_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_full_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv2_full_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv2_full_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 2, "Invalid dimensions access");

        return etl::dim(input, d) + etl::dim(kernel, d) - 1;
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 2, "Invalid dimension access");

        return etl::dim<D, I>() + etl::dim<D, K>() - 1;
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

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_same(i, k, c);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_same(i, k, c);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_same(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_same";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_same");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_same");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_same");

        cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0), "Invalid dimensions for conv2_same");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 1), "Invalid dimensions for conv2_same");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv2_same");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_same");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_same");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_same");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_same");

        static_assert(etl::dim<0,C>() == etl::dim<0,I>(), "Invalid dimensions for conv2_same");
        static_assert(etl::dim<1,C>() == etl::dim<1,I>(), "Invalid dimensions for conv2_same");
        static_assert(etl::dim<0,I>() >= etl::dim<0,K>(), "Invalid dimensions for conv2_same");
        static_assert(etl::dim<1,I>() >= etl::dim<1,K>(), "Invalid dimensions for conv2_same");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 2, "Invalid dimensions access");
        cpp_unused(kernel);

        return etl::dim(input, d);
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 2, "Invalid dimension access");

        return etl::dim<D, I>();
    }
};

/*!
 * \brief The functor impl for 2D same conv
 */
struct conv2_same_flipped_impl : conv2_same_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::SAME, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_same_flipped(i, k, c);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_same_flipped(i, k, c);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_same_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_same";
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
template<size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0>
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

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_valid<S1, S2, P1, P2>(i, k, c);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_valid<S1, S2, P1, P2>(i, k, c);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid<S1, S2, P1, P2>(i, k, c);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid<S1, S2, P1, P2>(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_valid";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_valid");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_valid");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_valid");

        cpp_assert(etl::dim(conv, 0) == (etl::dim(input, 0) - etl::dim(kernel, 0)) / S1 + 1, "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(conv, 1) == (etl::dim(input, 1) - etl::dim(kernel, 1)) / S2 + 1, "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv2_valid");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_valid");
        static_assert(etl::dimensions<K>() == 2, "Invalid number of dimensions for kernel of conv2_valid");
        static_assert(etl::dimensions<C>() == 2, "Invalid number of dimensions for conv of conv2_valid");

        static_assert(etl::dim<0,C>() == (etl::dim<0,I>() - etl::dim<0,K>()) / S1 + 1, "Invalid dimensions for conv2_valid");
        static_assert(etl::dim<1,C>() == (etl::dim<1,I>() - etl::dim<1,K>()) / S2 + 1, "Invalid dimensions for conv2_valid");
        static_assert(etl::dim<0,I>() >= etl::dim<0,K>(), "Invalid dimensions for conv2_valid");
        static_assert(etl::dim<1,I>() >= etl::dim<1,K>(), "Invalid dimensions for conv2_valid");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 2, "Invalid dimensions access");

        if(d == 0){
            return (etl::dim(input, 0) - etl::dim(kernel, 0)) / S1 + 1;
        } else {
            return (etl::dim(input, 1) - etl::dim(kernel, 1)) / S2 + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 2, "Invalid dimension access");

        return D == 0 ? (etl::dim<D, I>() - etl::dim<D, K>()) / S1 + 1
                      : (etl::dim<D, I>() - etl::dim<D, K>()) / S2 + 1;
    }
};

/*!
 * \brief The functor impl for 2D valid conv
 */
template<size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0>
struct conv2_valid_flipped_impl : conv2_valid_impl<S1, S2, P1, P2> {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        etl::conv_impl impl = select_conv_impl<conv_type::VALID, I, K, C>();

        auto i = input.direct();
        auto k = kernel.direct();
        auto c = conv.direct();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_valid_flipped<S1, S2, P1, P2>(i, k, c);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_valid_flipped<S1, S2, P1, P2>(i, k, c);
        } else if (impl == etl::conv_impl::CUDNN) {
            impl::cudnn::conv2_valid_flipped<S1, S2, P1, P2>(i, k, c);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid_flipped<S1, S2, P1, P2>(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_valid_flipped";
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
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_valid(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_valid(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_valid";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_valid");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_valid");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_valid");

        // TODO Not complete

        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 2) - etl::dim(kernel, 2) + 1, "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(conv, 3) == etl::dim(input, 3) - etl::dim(kernel, 3) + 1, "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(input, 2) >= etl::dim(kernel, 2), "Invalid dimensions for conv4_valid");
        cpp_assert(etl::dim(input, 3) >= etl::dim(kernel, 3), "Invalid dimensions for conv4_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 4, "Invalid number of dimensions for input of conv4_valid");
        static_assert(etl::dimensions<K>() == 4, "Invalid number of dimensions for kernel of conv4_valid");
        static_assert(etl::dimensions<C>() == 4, "Invalid number of dimensions for conv of conv4_valid");

        // TODO Not complete

        static_assert(etl::dim<2,C>() == etl::dim<2,I>() - etl::dim<2,K>() + 1, "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<3,C>() == etl::dim<3,I>() - etl::dim<3,K>() + 1, "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<2,I>() >= etl::dim<2,K>(), "Invalid dimensions for conv4_valid");
        static_assert(etl::dim<3,I>() >= etl::dim<3,K>(), "Invalid dimensions for conv4_valid");
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
            return etl::dim(kernel, 0);
        } else {
            return etl::dim(input, d) - etl::dim(kernel, d) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<0,I>()
            :  D == 1 ? etl::dim<0,K>()
            : etl::dim<D,I>() - etl::dim<D,K>() + 1;
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
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_valid_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_valid_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_valid_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv4_valid_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv4_valid_impl::template check<I, K, C>();
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
            return etl::dim(kernel, 0);
        } else {
            return etl::dim(input, d) - etl::dim(kernel, d) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<0,I>()
            :  D == 1 ? etl::dim<0,K>()
            : etl::dim<D,I>() - etl::dim<D,K>() + 1;
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct conv4_valid_filter_impl {
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
            impl::cudnn::conv4_valid_filter(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_valid_filter(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_valid_filter(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_valid_filter";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv4_valid_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv4_valid_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 1);
        } else if(d == 1){
            return etl::dim(input, 1);
        } else {
            return etl::dim(input, d) - etl::dim(kernel, d) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<1,K>()
            :  D == 1 ? etl::dim<1,I>()
            : etl::dim<D,I>() - etl::dim<D,K>() + 1;
    }
};

/*!
 * \brief The functor impl for 4D valid conv
 */
struct conv4_valid_filter_flipped_impl {
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
            impl::cudnn::conv4_valid_filter_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_valid_filter_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_valid_filter_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::STD) {
            impl::standard::conv4_valid_filter_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv4_valid_filter_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv4_valid_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv4_valid_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 1);
        } else if(d == 1){
            return etl::dim(input, 1);
        } else {
            return etl::dim(input, d) - etl::dim(kernel, d) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 4, "Invalid dimension access");

        return D == 0 ? etl::dim<1,K>()
            :  D == 1 ? etl::dim<1,I>()
            : etl::dim<D,I>() - etl::dim<D,K>() + 1;
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
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::FFT_STD) {
            impl::standard::conv4_full_fft(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::FFT_MKL) {
            impl::blas::conv4_full(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::FFT_CUFFT) {
            impl::cufft::conv4_full(input.direct(), kernel.direct(), conv.direct());
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

        // TODO Not complete

        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 2) + etl::dim(kernel, 2) - 1, "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(conv, 3) == etl::dim(input, 3) + etl::dim(kernel, 3) - 1, "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(input, 2) >= etl::dim(kernel, 2), "Invalid dimensions for conv4_full");
        cpp_assert(etl::dim(input, 3) >= etl::dim(kernel, 3), "Invalid dimensions for conv4_full");

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

        // TODO Not complete

        static_assert(etl::dim<2,C>() == etl::dim<2,I>() + etl::dim<2,K>() - 1, "Invalid dimensions for conv4_full");
        static_assert(etl::dim<3,C>() == etl::dim<3,I>() + etl::dim<3,K>() - 1, "Invalid dimensions for conv4_full");
        static_assert(etl::dim<2,I>() >= etl::dim<2,K>(), "Invalid dimensions for conv4_full");
        static_assert(etl::dim<3,I>() >= etl::dim<3,K>(), "Invalid dimensions for conv4_full");
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
        auto impl = select_conv4_impl<I, K, C>();

        if (impl == etl::conv4_impl::CUDNN) {
            impl::cudnn::conv4_full_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::AVX) {
            impl::avx::conv4_full_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv4_impl::SSE) {
            impl::sse::conv4_full_flipped(input.direct(), kernel.direct(), conv.direct());
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
        auto impl = select_conv_valid_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::BLAS) {
            impl::reduc::blas_conv2_valid_multi(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::FFT) {
            impl::reduc::fft_conv2_valid_multi(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_valid_multi(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::AVX) {
            impl::avx::conv2_valid_multi(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::SSE) {
            impl::sse::conv2_valid_multi(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_valid_multi";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_valid_multi");
        static_assert(etl::dimensions<K>() == 3, "Invalid number of dimensions for kernel of conv2_valid_multi");
        static_assert(etl::dimensions<C>() == 3, "Invalid number of dimensions for conv of conv2_valid_multi");

        cpp_assert(etl::dim(conv, 0) == etl::dim(kernel, 0), "Invalid dimensions for conv2_valid_multi");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 0) - etl::dim(kernel, 1) + 1, "Invalid dimensions for conv2_valid_multi");
        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 1) - etl::dim(kernel, 2) + 1, "Invalid dimensions for conv2_valid_multi");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_valid_multi");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 2), "Invalid dimensions for conv2_valid_multi");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_valid_multi");
        static_assert(etl::dimensions<K>() == 3, "Invalid number of dimensions for kernel of conv2_valid_multi");
        static_assert(etl::dimensions<C>() == 3, "Invalid number of dimensions for conv of conv2_valid_multi");

        static_assert(etl::dim<0,C>() == etl::dim<0,K>(), "Invalid dimensions for conv2_valid_multi");
        static_assert(etl::dim<1,C>() == etl::dim<0,I>() - etl::dim<1,K>() + 1, "Invalid dimensions for conv2_valid_multi");
        static_assert(etl::dim<2,C>() == etl::dim<1,I>() - etl::dim<2,K>() + 1, "Invalid dimensions for conv2_valid_multi");
        static_assert(etl::dim<0,I>() >= etl::dim<1,K>(), "Invalid dimensions for conv2_valid_multi");
        static_assert(etl::dim<1,I>() >= etl::dim<2,K>(), "Invalid dimensions for conv2_valid_multi");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 4, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 0);
        } else {
            return etl::dim(input, d - 1) - etl::dim(kernel, d) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 3, "Invalid dimension access");

        return D == 0 ? etl::dim<0,K>()
            : etl::safe_dim<D - 1,I>() - etl::dim<D,K>() + 1;
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
        auto impl = select_conv_valid_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::BLAS) {
            impl::reduc::blas_conv2_valid_multi_flipped(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::FFT) {
            impl::reduc::fft_conv2_valid_multi_flipped(input, kernel, conv);
        } else if (impl == etl::conv_multi_impl::CUDNN) {
            impl::cudnn::conv2_valid_multi_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::AVX) {
            impl::avx::conv2_valid_multi_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::SSE) {
            impl::sse::conv2_valid_multi_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_valid_multi_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_valid_multi_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv2_valid_multi_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv2_valid_multi_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 3, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 0);
        } else {
            return etl::dim(input, d - 1) - etl::dim(kernel, d) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 3, "Invalid dimension access");

        return D == 0 ? etl::dim<0,K>()
            : etl::safe_dim<D - 1,I>() - etl::dim<D,K>() + 1;
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
        for (std::size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        for (std::size_t i = 0; i < etl::dim<0>(input); ++i) {
            apply(input(i), kernel(i), conv(i));
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv_deep_valid";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        constexpr const size_t n = etl::dimensions<I>();

        static_assert(etl::dimensions<I>() == n, "Invalid number of dimensions for input of conv_deep_valid");
        static_assert(etl::dimensions<K>() == n, "Invalid number of dimensions for kernel of conv_deep_valid");
        static_assert(etl::dimensions<C>() == n, "Invalid number of dimensions for conv of conv_deep_valid");

        cpp_assert(etl::dim(conv, n - 1) == etl::dim(input, n - 1) - etl::dim(kernel, n - 1) + 1, "Invalid dimensions for conv_deep_valid");
        cpp_assert(etl::dim(conv, n - 2) == etl::dim(input, n - 2) - etl::dim(kernel, n - 2) + 1, "Invalid dimensions for conv_deep_valid");
        cpp_assert(etl::dim(input, n - 1) >= etl::dim(kernel, n - 1), "Invalid dimensions for conv_deep_valid");
        cpp_assert(etl::dim(input, n - 2) >= etl::dim(kernel, n - 2), "Invalid dimensions for conv_deep_valid");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        constexpr const size_t n = etl::dimensions<I>();

        static_assert(etl::dimensions<I>() == n, "Invalid number of dimensions for input of conv_deep_valid");
        static_assert(etl::dimensions<K>() == n, "Invalid number of dimensions for kernel of conv_deep_valid");
        static_assert(etl::dimensions<C>() == n, "Invalid number of dimensions for conv of conv_deep_valid");

        static_assert(etl::dim<n-1,C>() == etl::dim<n-1,I>() - etl::dim<n-1,K>() + 1, "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-2,C>() == etl::dim<n-2,I>() - etl::dim<n-2,K>() + 1, "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-1,I>() >= etl::dim<n-1,K>(), "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-2,I>() >= etl::dim<n-2,K>(), "Invalid dimensions for conv_deep_valid");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        if(d < etl::dimensions<I>() - 2){
            return etl::dim(input, d);
        } else {
            return etl::dim(input, d) - etl::dim(kernel) + 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        return D < etl::dimensions<I>() - 2 ? etl::dim<D,I>()
            : etl::dim<D,I>() - etl::dim<D, K>() + 1;
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
        for (std::size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        for (std::size_t i = 0; i < etl::dim<0>(input); ++i) {
            apply(input(i), kernel(i), conv(i));
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv_deep_same";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        constexpr const size_t n = etl::dimensions<I>();

        static_assert(etl::dimensions<I>() == n, "Invalid number of dimensions for input of conv_deep_same");
        static_assert(etl::dimensions<K>() == n, "Invalid number of dimensions for kernel of conv_deep_same");
        static_assert(etl::dimensions<C>() == n, "Invalid number of dimensions for conv of conv_deep_same");

        cpp_assert(etl::dim(conv, n - 1) == etl::dim(input, n - 1), "Invalid dimensions for conv_deep_same");
        cpp_assert(etl::dim(conv, n - 2) == etl::dim(input, n - 2), "Invalid dimensions for conv_deep_same");
        cpp_assert(etl::dim(input, n - 1) >= etl::dim(kernel, n - 1), "Invalid dimensions for conv_deep_same");
        cpp_assert(etl::dim(input, n - 2) >= etl::dim(kernel, n - 2), "Invalid dimensions for conv_deep_same");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        constexpr const size_t n = etl::dimensions<I>();

        static_assert(etl::dimensions<I>() == n, "Invalid number of dimensions for input of conv_deep_valid");
        static_assert(etl::dimensions<K>() == n, "Invalid number of dimensions for kernel of conv_deep_valid");
        static_assert(etl::dimensions<C>() == n, "Invalid number of dimensions for conv of conv_deep_valid");

        static_assert(etl::dim<n-1,C>() == etl::dim<n-1,I>(), "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-2,C>() == etl::dim<n-2,I>(), "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-1,I>() >= etl::dim<n-1,K>(), "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-2,I>() >= etl::dim<n-2,K>(), "Invalid dimensions for conv_deep_valid");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_unused(kernel);
        return etl::dim(input, d);
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        return etl::dim<D,I>();
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
        for (std::size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        for (std::size_t i = 0; i < etl::dim<0>(input); ++i) {
            apply(input(i), kernel(i), conv(i));
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv_deep_full";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        constexpr const size_t n = etl::dimensions<I>();

        static_assert(etl::dimensions<I>() == n, "Invalid number of dimensions for input of conv_deep_full");
        static_assert(etl::dimensions<K>() == n, "Invalid number of dimensions for kernel of conv_deep_full");
        static_assert(etl::dimensions<C>() == n, "Invalid number of dimensions for conv of conv_deep_full");

        cpp_assert(etl::dim(conv, n - 1) == etl::dim(input, n - 1) + etl::dim(kernel, n - 1) - 1, "Invalid dimensions for conv_deep_full");
        cpp_assert(etl::dim(conv, n - 2) == etl::dim(input, n - 2) + etl::dim(kernel, n - 2) - 1, "Invalid dimensions for conv_deep_full");
        cpp_assert(etl::dim(input, n - 1) >= etl::dim(kernel, n - 1), "Invalid dimensions for conv_deep_full");
        cpp_assert(etl::dim(input, n - 2) >= etl::dim(kernel, n - 2), "Invalid dimensions for conv_deep_full");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        constexpr const size_t n = etl::dimensions<I>();

        static_assert(etl::dimensions<I>() == n, "Invalid number of dimensions for input of conv_deep_valid");
        static_assert(etl::dimensions<K>() == n, "Invalid number of dimensions for kernel of conv_deep_valid");
        static_assert(etl::dimensions<C>() == n, "Invalid number of dimensions for conv of conv_deep_valid");

        static_assert(etl::dim<n-1,C>() == etl::dim<n-1,I>() + etl::dim<n-1,K>() - 1, "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-2,C>() == etl::dim<n-2,I>() + etl::dim<n-2,K>() - 1, "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-1,I>() >= etl::dim<n-1,K>(), "Invalid dimensions for conv_deep_valid");
        static_assert(etl::dim<n-2,I>() >= etl::dim<n-2,K>(), "Invalid dimensions for conv_deep_valid");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        if(d < etl::dimensions<I>() - 2){
            return etl::dim(input, d);
        } else {
            return etl::dim(input, d) + etl::dim(kernel) - 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        return D < etl::dimensions<I>() - 2 ? etl::dim<D,I>()
            : etl::dim<D,I>() + etl::dim<D, K>() - 1;
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
        } else if (impl == etl::conv_multi_impl::AVX){
            impl::avx::conv2_full_multi(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::SSE){
            impl::sse::conv2_full_multi(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_full_multi(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_full_multi";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_full_multi");
        static_assert(etl::dimensions<K>() == 3, "Invalid number of dimensions for kernel of conv2_full_multi");
        static_assert(etl::dimensions<C>() == 3, "Invalid number of dimensions for conv of conv2_full_multi");

        cpp_assert(etl::dim(conv, 0) == etl::dim(kernel, 0), "Invalid dimensions for conv2_full_multi");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 0) + etl::dim(kernel, 1) - 1, "Invalid dimensions for conv2_full_multi");
        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 1) + etl::dim(kernel, 2) - 1, "Invalid dimensions for conv2_full_multi");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_full_multi");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 2), "Invalid dimensions for conv2_full_multi");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_full_multi");
        static_assert(etl::dimensions<K>() == 3, "Invalid number of dimensions for kernel of conv2_full_multi");
        static_assert(etl::dimensions<C>() == 3, "Invalid number of dimensions for conv of conv2_full_multi");

        static_assert(etl::dim<0,C>() == etl::dim<0,K>(), "Invalid dimensions for conv2_full_multi");
        static_assert(etl::dim<1,C>() == etl::dim<0,I>() + etl::dim<1,K>() - 1, "Invalid dimensions for conv2_full_multi");
        static_assert(etl::dim<2,C>() == etl::dim<1,I>() + etl::dim<2,K>() - 1, "Invalid dimensions for conv2_full_multi");
        static_assert(etl::dim<0,I>() >= etl::dim<1,K>(), "Invalid dimensions for conv2_full_multi");
        static_assert(etl::dim<1,I>() >= etl::dim<2,K>(), "Invalid dimensions for conv2_full_multi");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 3, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 0);
        } else {
            return etl::dim(input, d - 1) + etl::dim(kernel, d) - 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 3, "Invalid dimension access");

        return D == 0 ? etl::dim<0,K>()
            : etl::safe_dim<D - 1,I>() + etl::dim<D,K>() - 1;
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
        } else if (impl == etl::conv_multi_impl::AVX){
            impl::avx::conv2_full_multi_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::SSE){
            impl::sse::conv2_full_multi_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_full_multi_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_full_multi_flipped";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        conv2_full_multi_impl::check(input, kernel, conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        conv2_full_multi_impl::template check<I, K, C>();
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 3, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 0);
        } else {
            return etl::dim(input, d - 1) + etl::dim(kernel, d) - 1;
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 3, "Invalid dimension access");

        return D == 0 ? etl::dim<0,K>()
            : etl::safe_dim<D - 1,I>() + etl::dim<D,K>() - 1;
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
        auto impl = select_conv_same_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::AVX){
            impl::avx::conv2_same_multi(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::SSE){
            impl::sse::conv2_same_multi(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_same_multi(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_same_multi";
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_same_multi");
        static_assert(etl::dimensions<K>() == 3, "Invalid number of dimensions for kernel of conv2_same_multi");
        static_assert(etl::dimensions<C>() == 3, "Invalid number of dimensions for conv of conv2_same_multi");

        cpp_assert(etl::dim(conv, 0) == etl::dim(kernel, 0), "Invalid dimensions for conv2_same_multi");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 0), "Invalid dimensions for conv2_same_multi");
        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 1), "Invalid dimensions for conv2_same_multi");
        cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 1), "Invalid dimensions for conv2_same_multi");
        cpp_assert(etl::dim(input, 1) >= etl::dim(kernel, 2), "Invalid dimensions for conv2_same_multi");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check(){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_same_multi");
        static_assert(etl::dimensions<K>() == 3, "Invalid number of dimensions for kernel of conv2_same_multi");
        static_assert(etl::dimensions<C>() == 3, "Invalid number of dimensions for conv of conv2_same_multi");

        static_assert(etl::dim<0,C>() == etl::dim<0,K>(), "Invalid dimensions for conv2_same_multi");
        static_assert(etl::dim<1,C>() == etl::dim<0,I>(), "Invalid dimensions for conv2_same_multi");
        static_assert(etl::dim<2,C>() == etl::dim<1,I>(), "Invalid dimensions for conv2_same_multi");
        static_assert(etl::dim<0,I>() >= etl::dim<1,K>(), "Invalid dimensions for conv2_same_multi");
        static_assert(etl::dim<1,I>() >= etl::dim<2,K>(), "Invalid dimensions for conv2_same_multi");
    }

    /*!
     * \brief Returns the dth dimension of the result of the convolution
     */
    template <typename I, typename K>
    static size_t dim(size_t d, const I& input, const K& kernel){
        cpp_assert(d < 3, "Invalid dimensions access");

        if(d == 0){
            return etl::dim(kernel, 0);
        } else {
            return etl::dim(input, d - 1);
        }
    }

    /*!
     * \brief Returns the Dth dimension of the result of the convolution
     */
    template <size_t D, typename I, typename K>
    static constexpr size_t dim(){
        static_assert(D < 3, "Invalid dimension access");

        return D == 0 ? etl::dim<0,K>()
            : etl::safe_dim<D - 1,I>();
    }
};

/*!
 * \brief The functor impl for 2D same conv, with multiple flipped kernels
 */
struct conv2_same_multi_flipped_impl : conv2_same_multi_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        auto impl = select_conv_same_multi_impl<I, K, C>();

        if (impl == etl::conv_multi_impl::AVX){
            impl::avx::conv2_same_multi_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::SSE){
            impl::sse::conv2_same_multi_flipped(input.direct(), kernel.direct(), conv.direct());
        } else if (impl == etl::conv_multi_impl::STD){
            impl::standard::conv2_same_multi_flipped(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Returns the description of the operation
     */
    static constexpr const char* desc(){
        return "conv2_same_multi";
    }
};

} //end of namespace detail

} //end of namespace etl
