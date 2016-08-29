//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

} //end of namespace detail

} //end of namespace etl
