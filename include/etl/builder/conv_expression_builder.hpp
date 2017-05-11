//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file conv_expression_builder.hpp
 * \brief Contains all the operators and functions to build convolution expressions.
*/

#pragma once

#include "etl/expression_helpers.hpp"

namespace etl {

/*!
 * \brief Construct a matrix to compute a convolution by matrix-matrix multiplication
 * \param a The vector to transform (the input of the convolution)
 * \param h The size of kernel
 * \return a matrix expression for convolution
 */
template <typename A>
auto convmtx(A&& a, size_t h) -> detail::stable_transform_helper<A, dyn_convmtx_transformer> {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 1, "Convolutional matrix only works in 1D");

    return detail::stable_transform_helper<A, dyn_convmtx_transformer>{dyn_convmtx_transformer<detail::build_type<A>>(a, h)};
}

/*!
 * \brief Construct a matrix to compute a 2D convolution by matrix-matrix multiplication
 * \param a The 2D matrix to transform (the input of the convolution)
 * \param k1 The first dimension of the kernel
 * \param k2 The second dimension of the kernel
 * \return a matrix expression for convolution
 */
template <typename A>
auto convmtx2(A&& a, size_t k1, size_t k2) -> detail::stable_transform_helper<A, dyn_convmtx2_transformer> {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2, "Convolutional matrix only works in 2D");

    return detail::stable_transform_helper<A, dyn_convmtx2_transformer>{dyn_convmtx2_transformer<detail::build_type<A>>(a, k1, k2)};
}

/*!
 * \brief Construct a matrix to compute a 2D convolution by matrix-matrix multiplication
 * \param a The 2D matrix to transform (the input of the convolution)
 * \tparam K1 The first dimension of the kernel
 * \tparam K2 The second dimension of the kernel
 * \return a matrix expression for convolution
 */
template <size_t K1, size_t K2, typename A>
convmtx_2d_expr<detail::build_type<A>, K1, K2> convmtx2_direct(A&& a) {
    static_assert(is_etl_expr<A>::value, "Convolution matrices only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2, "Convolutional matrix only works in 2D");

    return convmtx_2d_expr<detail::build_type<A>, K1, K2>{a};
}

} //end of namespace etl
