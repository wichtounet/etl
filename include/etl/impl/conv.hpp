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

namespace etl {

/*!
 * \brief Enumeration describing the different types of convolution
 */
enum class conv_type {
    VALID, ///< Valid convolution
    SAME,  ///< Same convolution
    FULL   ///< Full convolution
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
template <typename I, typename K, typename C>
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
template <typename I, typename K, typename C>
inline etl::conv_impl select_conv_impl() {
    if(local_context().conv_selector.forced){
        auto forced = local_context().conv_selector.impl;

        switch (forced) {
            //AVX cannot always be used
            case conv_impl::AVX:
                if(!avx_enabled){
                    std::cerr << "Forced selection to AVX sum implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_impl<I, K, C>();
                }

                return forced;

            //SSE cannot always be used
            case conv_impl::SSE:
                if(!sse3_enabled){
                    std::cerr << "Forced selection to SSE sum implementation, but not possible for this expression" << std::endl;
                    return select_default_conv_impl<I, K, C>();
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_conv_impl<I, K, C>();
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
    if((parallel && !local_context().serial) || local_context().parallel){
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
        etl::conv_impl impl = select_conv_impl<I, K, C>();
        selected_apply(input, kernel, std::forward<C>(conv), impl);
    }

    /*!
     * \brief Apply the convolution with the given implementation
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     * \param impl The implementation to use
     */
    template <typename I, typename K, typename C>
    static void selected_apply(const I& input, const K& kernel, C&& conv, etl::conv_impl impl) {
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::avx::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::sse::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::standard::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv));
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
        etl::conv_impl impl = select_conv_impl<I, K, C>();
        selected_apply(input, kernel, std::forward<C>(conv), impl);
    }

    /*!
     * \brief Apply the convolution with the given implementation
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     * \param impl The implementation to use
     */
    template <typename I, typename K, typename C>
    static void selected_apply(const I& input, const K& kernel, C&& conv, etl::conv_impl impl) {
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::avx::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::sse::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::standard::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv));
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
        etl::conv_impl impl = select_conv_impl<I, K, C>();
        selected_apply(input, kernel, std::forward<C>(conv), impl);
    }

    /*!
     * \brief Apply the convolution with the given implementation
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     * \param impl The implementation to use
     */
    template <typename I, typename K, typename C>
    static void selected_apply(const I& input, const K& kernel, C&& conv, etl::conv_impl impl) {
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::AVX) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::avx::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::SSE) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::sse::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            dispatch_1d(parallel_dispatch, [&](std::size_t first, std::size_t last){
                impl::standard::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv));
        }
    }
};

//Should only be used by the benchmark
template <typename I, typename K, typename C>
void conv1_full_direct(const I& input, const K& kernel, C&& conv, etl::conv_impl impl){
    conv1_full_impl::selected_apply(input, kernel, std::forward<C>(conv), impl);
}

//Should only be used by the benchmark
template <typename I, typename K, typename C>
void conv1_same_direct(const I& input, const K& kernel, C&& conv, etl::conv_impl impl){
    conv1_same_impl::selected_apply(input, kernel, std::forward<C>(conv), impl);
}

//Should only be used by the benchmark
template <typename I, typename K, typename C>
void conv1_valid_direct(const I& input, const K& kernel, C&& conv, etl::conv_impl impl){
    conv1_valid_impl::selected_apply(input, kernel, std::forward<C>(conv), impl);
}

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
        etl::conv_impl impl = select_conv_impl<I, K, C>();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_full(input, kernel, conv);
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
        etl::conv_impl impl = select_conv_impl<I, K, C>();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_same(input, kernel, conv);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_same(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_same(input, kernel, conv);
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
        etl::conv_impl impl = select_conv_impl<I, K, C>();

        if (impl == etl::conv_impl::AVX) {
            impl::avx::conv2_valid(input, kernel, conv);
        } else if (impl == etl::conv_impl::SSE) {
            impl::sse::conv2_valid(input, kernel, conv);
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv2_valid(input, kernel, conv);
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

} //end of namespace detail

} //end of namespace etl
