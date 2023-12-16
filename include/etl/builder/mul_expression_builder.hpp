//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file mul_expression_builder.hpp
 * \brief Contains all the operators and functions to build multiplication expressions.
 */

#pragma once

namespace etl {

/*!
 * \brief Multiply two matrices together lazily (expression templates)
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <etl_2d A, etl_2d B>
auto lazy_mul(A&& a, B&& b) -> detail::stable_transform_binary_helper<A, B, mm_mul_transformer> {
    return detail::stable_transform_binary_helper<A, B, mm_mul_transformer>{mm_mul_transformer<detail::build_type<A>, detail::build_type<B>>(a, b)};
}

} //end of namespace etl
