//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for 1D convolution operations
 */

#pragma once

namespace etl {

namespace detail {

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
        const auto impl = select_conv1_impl_new<conv_type::FULL, I, K, C>();

//CPP17: if constexpr
#ifdef ETL_PARALLEL_SUPPORT
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::VEC) {
            engine_dispatch_1d([&](size_t first, size_t last) {
                impl::vec::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv), parallel_dispatch);
        } else if (impl == etl::conv_impl::STD) {
            engine_dispatch_1d([&](size_t first, size_t last) {
                impl::standard::conv1_full(input, kernel, conv, first, last);
            }, 0, size(conv), parallel_dispatch);
        } else if (impl == etl::conv_impl::FFT_STD) {
            impl::standard::conv1_full_fft(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_MKL) {
            impl::blas::conv1_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_CUFFT) {
            impl::cufft::conv1_full(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
#else
        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv1_full(input, kernel, conv, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv1_full(input, kernel, conv, 0, size(conv));
        } else if (impl == etl::conv_impl::FFT_STD) {
            impl::standard::conv1_full_fft(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_MKL) {
            impl::blas::conv1_full(input, kernel, conv);
        } else if (impl == etl::conv_impl::FFT_CUFFT) {
            impl::cufft::conv1_full(input, kernel, conv);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
#endif
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
        const auto impl = select_conv1_impl_new<conv_type::SAME, I, K, C>();

//CPP17: if constexpr
#ifdef ETL_PARALLEL_SUPPORT
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::VEC) {
            engine_dispatch_1d([&](size_t first, size_t last) {
                impl::vec::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv), parallel_dispatch);
        } else if (impl == etl::conv_impl::STD) {
            engine_dispatch_1d([&](size_t first, size_t last) {
                impl::standard::conv1_same(input, kernel, conv, first, last);
            }, 0, size(conv), parallel_dispatch);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
#else
        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv1_same(input, kernel, conv, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv1_same(input, kernel, conv, 0, size(conv));
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
#endif
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
        const auto impl = select_conv1_impl_new<conv_type::VALID, I, K, C>();

//CPP17: if constexpr
#ifdef ETL_PARALLEL_SUPPORT
        bool parallel_dispatch = select_parallel(input, kernel, conv);

        if (impl == etl::conv_impl::VEC) {
            engine_dispatch_1d([&](size_t first, size_t last) {
                impl::vec::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv), parallel_dispatch);
        } else if (impl == etl::conv_impl::STD) {
            engine_dispatch_1d([&](size_t first, size_t last) {
                impl::standard::conv1_valid(input, kernel, conv, first, last);
            }, 0, size(conv), parallel_dispatch);
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
#else
        if (impl == etl::conv_impl::VEC) {
            impl::vec::conv1_valid(input, kernel, conv, 0, size(conv));
        } else if (impl == etl::conv_impl::STD) {
            impl::standard::conv1_valid(input, kernel, conv, 0, size(conv));
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
#endif
    }
};

} //end of namespace detail

} //end of namespace etl
