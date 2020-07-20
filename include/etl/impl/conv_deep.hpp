//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains descriptors for "deep" convolution operations
 */

#pragma once

namespace etl::detail {

/*!
 * \brief The functor impl for 2D+ conv.
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_deep_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        if constexpr (is_3d<I>) {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                conv(i) = conv_2d_valid<S1, S2, P1, P2>(input(i), kernel(i));
            }
        } else {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                apply(input(i), kernel(i), conv(i));
            }
        }
    }
};

/*!
 * \brief The functor impl for 2D+ conv.
 */
template <size_t S1, size_t S2, size_t P1, size_t P2>
struct conv2_valid_flipped_deep_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        if constexpr (is_3d<I>) {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                conv(i) = conv_2d_valid_flipped<S1, S2, P1, P2>(input(i), kernel(i));
            }
        } else {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                apply(input(i), kernel(i), conv(i));
            }
        }
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
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        if constexpr (is_3d<I>) {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                conv(i) = conv_2d_same(input(i), kernel(i));
            }
        } else {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                apply(input(i), kernel(i), conv(i));
            }
        }
    }
};

/*!
 * \brief The functor impl for 2D+ conv.
 */
struct conv2_same_flipped_deep_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        if constexpr (is_3d<I>) {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                conv(i) = conv_2d_same_flipped(input(i), kernel(i));
            }
        } else {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                apply(input(i), kernel(i), conv(i));
            }
        }
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
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        if constexpr (is_3d<I>) {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                conv(i) = conv_2d_full(input(i), kernel(i));
            }
        } else {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                apply(input(i), kernel(i), conv(i));
            }
        }
    }
};

/*!
 * \brief The functor impl for 2D+ conv.
 */
struct conv2_full_flipped_deep_impl {
    /*!
     * \brief Apply the convolution
     * \param input The input expression
     * \param kernel The kernel expression
     * \param conv The output expression
     */
    template <typename I, typename K, typename C>
    static void apply(const I& input, const K& kernel, C&& conv) {
        if constexpr (is_3d<I>) {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                conv(i) = conv_2d_full_flipped(input(i), kernel(i));
            }
        } else {
            for (size_t i = 0; i < etl::dim<0>(input); ++i) {
                apply(input(i), kernel(i), conv(i));
            }
        }
    }
};

} //end of namespace etl::detail
