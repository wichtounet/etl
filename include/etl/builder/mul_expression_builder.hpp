//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file mul_expression_builder.hpp
 * \brief Contains all the operators and functions to build multiplication expressions.
*/

#pragma once

#include "etl/expression_helpers.hpp"

namespace etl {

/*!
 * \brief Multiply two matrices together
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_2d<A>::value, is_2d<B>::value, !is_element_wise_mul_default)>
auto operator*(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

/*!
 * \brief Multiply a vector and a matrix together
 * \param a The left hand side vector
 * \param b The right hand side matrix
 * \return An expression representing the vector-matrix multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_1d<A>::value, is_2d<B>::value, !is_element_wise_mul_default)>
auto operator*(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, vm_mul_expr> {
    return {a, b};
}

/*!
 * \brief Multiply a matrix and a vector together
 * \param a The left hand side matrix
 * \param b The right hand side vector
 * \return An expression representing the matrix-vector multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_2d<A>::value, is_1d<B>::value, !is_element_wise_mul_default)>
auto operator*(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, mv_mul_expr> {
    return {a, b};
}

/*!
 * \brief Multiply two matrices together
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_2d<A>::value, is_2d<B>::value)>
auto mul(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

/*!
 * \brief Multiply two matrices together and store the result in c
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \param c The expression used to store the result
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B, typename C, cpp_enable_if(is_2d<A>::value, is_2d<B>::value, is_2d<C>::value)>
auto mul(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");

    c = mul(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Multiply two matrices together lazily (expression templates)
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B>
auto lazy_mul(A&& a, B&& b) -> detail::stable_transform_binary_helper<A, B, mm_mul_transformer> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return detail::stable_transform_binary_helper<A, B, mm_mul_transformer>{mm_mul_transformer<detail::build_type<A>, detail::build_type<B>>(a, b)};
}

/*!
 * \brief Multiply a vector and a matrix together
 * \param a The left hand side vector
 * \param b The right hand side matrix
 * \return An expression representing the vector-matrix multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_1d<A>::value, is_2d<B>::value)>
auto mul(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, vm_mul_expr> {
    return {a, b};
}

/*!
 * \brief Multiply a vector and a matrix together and store the result in c
 * \param a The left hand side vector
 * \param b The right hand side matrix
 * \param c The expression used to store the result
 * \return An expression representing the vector-matrix multiplication of a and b
 */
template <typename A, typename B, typename C, cpp_enable_if(is_1d<A>::value, is_2d<B>::value)>
auto mul(A&& a, B&& b, C&& c){
    c = mul(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Multiply a matrix and a vector together
 * \param a The left hand side matrix
 * \param b The right hand side vector
 * \return An expression representing the matrix-vector multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_2d<A>::value, is_1d<B>::value)>
auto mul(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, mv_mul_expr> {
    return {a, b};
}

/*!
 * \brief Multiply a matrix and a vector together and store the result in c
 * \param a The left hand side matrix
 * \param b The right hand side vector
 * \param c The expression used to store the result
 * \return An expression representing the matrix-vector multiplication of a and b
 */
template <typename A, typename B, typename C, cpp_enable_if(is_2d<A>::value, is_1d<B>::value)>
auto mul(A&& a, B&& b, C&& c) {
    c = mul(a, b);
    return std::forward<C>(c);
}

/*!
 * \brief Multiply two matrices together using strassen
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B>
auto strassen_mul(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, strassen_mm_mul_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return {a, b};
}

/*!
 * \brief Multiply two matrices together using strassen and store the result in c
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \param c The expression used to store the result
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B, typename C>
auto strassen_mul(A&& a, B&& b, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2 && decay_traits<C>::dimensions() == 2, "Matrix multiplication only works in 2D");

    c = mul(a,b);
    return std::forward<C>(c);
}

/*!
 * \brief Outer product multiplication of two matrices
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B>
auto outer(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, outer_product_expr> {
    return {std::forward<A>(a), std::forward<B>(b)};
}

/*!
 * \brief Outer product multiplication of two matrices and store the result in c
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \param c The expression used to store the result
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B, typename C>
auto outer(A&& a, B&& b, C&& c){
    c = outer(std::forward<A>(a), std::forward<B>(b));
    return std::forward<C>(c);
}

/*!
 * \brief Batch Outer product multiplication of two matrices
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the outer product multiplication of a and b
 */
template <typename A, typename B>
auto batch_outer(A&& a, B&& b) -> detail::temporary_binary_helper<A, B, batch_outer_product_expr> {
    return {std::forward<A>(a), std::forward<B>(b)};
}

/*!
 * \brief Batch Outer product multiplication of two matrices and store the result in c
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \param c The expression used to store the result
 * \return An expression representing the outer product multiplication of a and b
 */
template <typename A, typename B, typename C>
auto batch_outer(A&& a, B&& b, C&& c){
    c = batch_outer(std::forward<A>(a), std::forward<B>(b));
    return std::forward<C>(c);
}

} //end of namespace etl
