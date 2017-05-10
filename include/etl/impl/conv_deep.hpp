//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for "deep" convolution operations
 */

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief The functor impl for 2D+ conv.
 */
struct conv2_valid_deep_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if(decay_traits<I>::dimensions() == 3)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        constexpr size_t n = etl::dimensions<I>();

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
        constexpr size_t n = etl::dimensions<I>();

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
struct conv2_same_deep_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if(decay_traits<I>::dimensions() == 3)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        constexpr size_t n = etl::dimensions<I>();

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
        constexpr size_t n = etl::dimensions<I>();

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
struct conv2_full_deep_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C, cpp_enable_if(decay_traits<I>::dimensions() == 3)>
    static void apply(const I& input, const K& kernel, C&& conv) {
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        for (size_t i = 0; i < etl::dim<0>(input); ++i) {
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
        constexpr size_t n = etl::dimensions<I>();

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
        constexpr size_t n = etl::dimensions<I>();

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

} //end of namespace detail

} //end of namespace etl
