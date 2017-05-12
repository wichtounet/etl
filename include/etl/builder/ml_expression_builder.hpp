//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains some special helpers for machine learning.
 *
 * This is mostly a simpler set of names and functions to achieve
 * machine learning features.
*/

#pragma once

namespace etl {

namespace ml {

/*!
 * \brief Forward convolution for a batch of images with a set of kernels.
 *
 * This will compute the 2D convolutions of each image with each given
 * kernels. The results accross channels will be accumulated together.
 *
 * The 4D matrix a is assumed to be of [N, C, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hj, Wj] dimensions.
 * The 4D matrix c is assumed to be of [N, K, (Hi - Hj + 2 * P1) / S1 + 1, (Wi - Hj + 2 * P2) / S2 + 1] dimensions.
 *
 * \param a An expression containing the batch of images
 * \param b An expression containing the set of kernels
 *
 * \tparam S1 The stride in the first dimension
 * \tparam S2 The stride in the second dimension
 * \tparam P1 The padding of the first dimension
 * \tparam P2 The padding of the second dimension
 *
 * \return an expression representing the result of the forward convolution
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
conv_4d_valid_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true>
convolution_forward(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    return conv_4d_valid_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true>{a, b};
}

/*!
 * \brief Forward convolution for a batch of images with a set of kernels.
 *
 * This will compute the 2D convolutions of each image with each given
 * kernels. The results accross channels will be accumulated together.
 *
 * The 4D matrix a is assumed to be of [N, C, Hi, Wi] dimensions.
 * The 4D matrix b is assumed to be of [K, C, Hj, Wj] dimensions.
 * The 4D matrix c is assumed to be of [N, K, (Hi - Hj + 2 * P1) / S1 + 1, (Wi - Hj + 2 * P2) / S2 + 1] dimensions.
 *
 * \param a An expression containing the batch of images
 * \param b An expression containing the set of kernels
 *
 * \param s1 The stride in the first dimension
 * \param s2 The stride in the second dimension
 * \param p1 The padding of the first dimension
 * \param p2 The padding of the second dimension
 *
 * \return an expression representing the result of the forward convolution
 */
template <typename A, typename B>
dyn_conv_4d_valid_expr<detail::build_type<A>, detail::build_type<B>, true>
convolution_forward(A&& a, B&& b, size_t s1, size_t s2, size_t p1 = 0, size_t p2 = 0) {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    return dyn_conv_4d_valid_expr<detail::build_type<A>, detail::build_type<B>, true>{a, b, s1, s2, p1, p2};
}

} //end of namespace ml
} //end of namespace etl
