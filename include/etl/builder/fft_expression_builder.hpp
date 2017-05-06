//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file fft_expression_builder.hpp
 * \brief Contains all the operators and functions to build FFT expressions.
*/

#pragma once

#include "etl/expression_helpers.hpp"

namespace etl {

//Helpers to compute the type of the result

namespace detail {

/*!
 * \brief The output value type of an FFT based on the input
 */
template <typename A>
using fft_value_type = std::conditional_t<is_complex<A>::value, value_t<A>, std::complex<value_t<A>>>;

/*!
 * \brief The output value type of an Inverse FFT based on the input
 */
template <typename A>
using ifft_value_type = std::conditional_t<is_complex<A>::value, value_t<A>, std::complex<value_t<A>>>;

/*!
 * \brief The output value type of an Inverse FFT real based on the input
 */
template <typename A>
using ifft_real_value_type = std::conditional_t<is_complex<A>::value, typename value_t<A>::value_type, value_t<A>>;

} //end of namespace detail

/*!
 * \brief Creates an expression representing the 1D Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 1D FFT of a
 */
template <typename A>
fft_expr<A, detail::fft_value_type<A>, detail::fft1_impl> fft_1d(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return fft_expr<A, detail::fft_value_type<A>, detail::fft1_impl>{a};
}

/*!
 * \brief Creates an expression representing the 1D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the 1D FFT of a
 */
template <typename A, typename C>
auto fft_1d(A&& a, C&& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = fft_1d(a);
    return c;
}

/*!
 * \brief Creates an expression representing the 1D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 1D inverse FFT of a
 */
template <typename A>
fft_expr<A, detail::ifft_value_type<A>, detail::ifft1_impl> ifft_1d(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return fft_expr<A, detail::ifft_value_type<A>, detail::ifft1_impl>{a};
}

/*!
 * \brief Creates an expression representing the 1D inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the 1D inverse FFT of a
 */
template <typename A, typename C>
auto ifft_1d(A&& a, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = ifft_1d(a);
    return c;
}

/*!
 * \brief Creates an expression representing the real part of the 1D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the real part of the 1D inverse FFT of a
 */
template <typename A>
fft_expr<A, detail::ifft_real_value_type<A>, detail::ifft1_real_impl> ifft_1d_real(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return fft_expr<A, detail::ifft_real_value_type<A>, detail::ifft1_real_impl>{a};
}

/*!
 * \brief Creates an expression representing the real part of the 1D inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the real part of the 1D inverse FFT of a
 */
template <typename A, typename C>
auto ifft_1d_real(A&& a, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = ifft_1d_real(a);
    return c;
}

/*!
 * \brief Creates an expression representing the 2D Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 2D FFT of a
 */
template <typename A>
fft_expr<A, detail::fft_value_type<A>, detail::fft2_impl> fft_2d(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return fft_expr<A, detail::fft_value_type<A>, detail::fft2_impl>{a};
}

/*!
 * \brief Creates an expression representing the 2D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the 2D FFT of a
 */
template <typename A, typename C>
auto fft_2d(A&& a, C&& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = fft_2d(a);
    return c;
}

/*!
 * \brief Creates an expression representing the 2D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 2D inverse FFT of a
 */
template <typename A>
fft_expr<A, detail::ifft_value_type<A>, detail::ifft2_impl> ifft_2d(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return fft_expr<A, detail::ifft_value_type<A>, detail::ifft2_impl>{a};
}

/*!
 * \brief Creates an expression representing the 2D inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the 2D inverse FFT of a
 */
template <typename A, typename C>
auto ifft_2d(A&& a, C&& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = ifft_2d(a);
    return c;
}

/*!
 * \brief Creates an expression representing the real part of the 2D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the real part of the 2D inverse FFT of a
 */
template <typename A>
fft_expr<A, detail::ifft_real_value_type<A>, detail::ifft2_real_impl> ifft_2d_real(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return fft_expr<A, detail::ifft_real_value_type<A>, detail::ifft2_real_impl>{a};
}

/*!
 * \brief Creates an expression representing the real part of the 2D inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the real part of the 2D inverse FFT of a
 */
template <typename A, typename C>
auto ifft_2d_real(A&& a, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = ifft_2d_real(a);
    return c;
}

/*!
 * \brief Creates an expression representing several 1D Fast-Fourrier-Transform of the given expression
 *
 * Only the last dimension is used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \return an expression representing several 1D FFT of a
 */
template <typename A>
fft_expr<A, detail::fft_value_type<A>, detail::fft1_many_impl> fft_1d_many(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2, "fft_many requires at least 2D matrices");

    return fft_expr<A, detail::fft_value_type<A>, detail::fft1_many_impl>{a};
}

/*!
 * \brief Creates an expression representing several 1D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 *
 * Only the last dimension is used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param c The result
 * \return an expression representing several 1D FFT of a
 */
template <typename A, typename C>
auto fft_1d_many(A&& a, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2 && decay_traits<C>::dimensions() >= 2, "fft_many requires at least 2D matrices");
    validate_assign(c, a);

    c = fft_1d_many(a);
    return c;
}

/*!
 * \brief Creates an expression representing several 1D Inverse Fast-Fourrier-Transform of the given expression
 *
 * Only the last dimension is used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \return an expression representing several 1D FFT of a
 */
template <typename A>
fft_expr<A, detail::ifft_value_type<A>, detail::ifft1_many_impl> ifft_1d_many(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2, "ifft_many requires at least 2D matrices");

    return fft_expr<A, detail::ifft_value_type<A>, detail::ifft1_many_impl>{a};
}

/*!
 * \brief Creates an expression representing several 1D Inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 *
 * Only the last dimension is used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param c The result
 * \return an expression representing several 1D FFT of a
 */
template <typename A, typename C>
auto ifft_1d_many(A&& a, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2 && decay_traits<C>::dimensions() >= 2, "ifft_many requires at least 2D matrices");
    validate_assign(c, a);

    c = ifft_1d_many(a);
    return c;
}

/*!
 * \brief Creates an expression representing several 2D Fast-Fourrier-Transform of the given expression
 *
 * Only the last two dimensions are used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \return an expression representing several 2D FFT of a
 */
template <typename A>
fft_expr<A, detail::fft_value_type<A>, detail::fft2_many_impl> fft_2d_many(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3, "fft_many requires at least 3D matrices");

    return fft_expr<A, detail::fft_value_type<A>, detail::fft2_many_impl>{a};
}

/*!
 * \brief Creates an expression representing several 2D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 *
 * Only the last two dimensions are used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param c The result
 * \return an expression representing several 2D FFT of a
 */
template <typename A, typename C>
auto fft_2d_many(A&& a, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3 && decay_traits<C>::dimensions() >= 3, "fft_many requires at least 3D matrices");
    validate_assign(c, a);

    c = fft_2d_many(a);
    return c;
}

/*!
 * \brief Creates an expression representing several 2D Fast-Fourrier-Transform of the given expression
 *
 * Only the last two dimensions are used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \return an expression representing several 2D FFT of a
 */
template <typename A>
fft_expr<A, detail::ifft_value_type<A>, detail::ifft2_many_impl> ifft_2d_many(A&& a) {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3, "ifft_many requires at least 3D matrices");

    return fft_expr<A, detail::ifft_value_type<A>, detail::ifft2_many_impl>{a};
}

/*!
 * \brief Creates an expression representing several 2D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 *
 * Only the last two dimensions are used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param c The result
 * \return an expression representing several 2D FFT of a
 */
template <typename A, typename C>
auto ifft_2d_many(A&& a, C&& c) {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3 && decay_traits<C>::dimensions() >= 3, "ifft_many requires at least 3D matrices");
    validate_assign(c, a);

    c = ifft_2d_many(a);
    return c;
}

} //end of namespace etl
