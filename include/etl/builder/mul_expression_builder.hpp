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

#ifndef ETL_ELEMENT_WISE_MULTIPLICATION

/*!
 * \brief Multiply two matrices together
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_2d<A>::value, is_2d<B>::value)>
gemm_expr<A, B, detail::mm_mul_impl> operator*(A&& a, B&& b) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return gemm_expr<A, B, detail::mm_mul_impl>{a, b};
}

/*!
 * \brief Multiply a vector and a matrix together
 * \param a The left hand side vector
 * \param b The right hand side matrix
 * \return An expression representing the vector-matrix multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_1d<A>::value, is_2d<B>::value)>
gevm_expr<A, B, detail::vm_mul_impl> operator*(A&& a, B&& b) {
    return gevm_expr<A, B, detail::vm_mul_impl>{a, b};
}

/*!
 * \brief Multiply a matrix and a vector together
 * \param a The left hand side matrix
 * \param b The right hand side vector
 * \return An expression representing the matrix-vector multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_2d<A>::value, is_1d<B>::value)>
gemv_expr<A, B, detail::mv_mul_impl> operator*(A&& a, B&& b) {
    return gemv_expr<A, B, detail::mv_mul_impl>{a, b};
}

#endif

/*!
 * \brief Multiply two matrices together
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_2d<A>::value, is_2d<B>::value)>
gemm_expr<A, B, detail::mm_mul_impl> mul(A&& a, B&& b) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return gemm_expr<A, B, detail::mm_mul_impl>{a, b};
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
    return c;
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
gevm_expr<A, B, detail::vm_mul_impl> mul(A&& a, B&& b) {
    return gevm_expr<A, B, detail::vm_mul_impl>{a, b};
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
    return c;
}

/*!
 * \brief Multiply a matrix and a vector together
 * \param a The left hand side matrix
 * \param b The right hand side vector
 * \return An expression representing the matrix-vector multiplication of a and b
 */
template <typename A, typename B, cpp_enable_if(is_2d<A>::value, is_1d<B>::value)>
gemv_expr<A, B, detail::mv_mul_impl> mul(A&& a, B&& b){
    return gemv_expr<A, B, detail::mv_mul_impl>{a, b};
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
    return c;
}

/*!
 * \brief Multiply two matrices together using strassen
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B>
gemm_expr<A, B, detail::strassen_mm_mul_impl> strassen_mul(A&& a, B&& b) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value, "Matrix multiplication only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() == 2 && decay_traits<B>::dimensions() == 2, "Matrix multiplication only works in 2D");

    return gemm_expr<A, B, detail::strassen_mm_mul_impl>{a, b};
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
    return c;
}

/*!
 * \brief Outer product multiplication of two matrices
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the matrix-matrix multiplication of a and b
 */
template <typename A, typename B>
outer_product_expr<A, B> outer(A&& a, B&& b) {
    return outer_product_expr<A, B>{a, b};
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
    c = outer(a, b);
    return c;
}

/*!
 * \brief Batch Outer product multiplication of two matrices
 * \param a The left hand side matrix
 * \param b The right hand side matrix
 * \return An expression representing the outer product multiplication of a and b
 */
template <typename A, typename B>
batch_outer_product_expr<A, B> batch_outer(A&& a, B&& b) {
    return batch_outer_product_expr<A, B>{a, b};
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
    c = batch_outer(a, b);
    return c;
}

} //end of namespace etl
