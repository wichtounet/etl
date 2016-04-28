//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

template <typename T, typename A, template <typename, typename> class OP>
using temporary_unary_helper_type_alpha = OP<T, detail::build_type<A>>;

/*!
 * \brief Creates an expression representing the 1D Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 1D FFT of a
 */
template <typename A>
auto fft_1d(A&& a) -> temporary_unary_helper_type_alpha<detail::fft_value_type<A>, A, fft1_expr_alpha> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return temporary_unary_helper_type_alpha<detail::fft_value_type<A>, A, fft1_expr_alpha>{a};
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

    return c = fft_1d(a);
}

/*!
 * \brief Creates an expression representing the 1D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 1D inverse FFT of a
 */
template <typename A>
auto ifft_1d(A&& a) -> temporary_unary_helper_type_alpha<detail::ifft_value_type<A>, A, ifft1_expr_alpha> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return temporary_unary_helper_type_alpha<detail::ifft_value_type<A>, A, ifft1_expr_alpha>{a};
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

    return c = ifft_1d(a);
}

/*!
 * \brief Creates an expression representing the real part of the 1D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the real part of the 1D inverse FFT of a
 */
template <typename A>
auto ifft_1d_real(A&& a) -> temporary_unary_helper_type_alpha<detail::ifft_real_value_type<A>, A, ifft1_real_expr_alpha> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return temporary_unary_helper_type_alpha<detail::ifft_real_value_type<A>, A, ifft1_real_expr_alpha>{a};
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

    return c = ifft_1d_real(a);
}

/*!
 * \brief Creates an expression representing the 2D Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 2D FFT of a
 */
template <typename A>
auto fft_2d(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft2_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft2_expr>{a};
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

    return c = fft_2d(a);
}

/*!
 * \brief Creates an expression representing the 2D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 2D inverse FFT of a
 */
template <typename A>
auto ifft_2d(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, ifft2_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, ifft2_expr>{a};
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

    return c = ifft_2d(a);
}

/*!
 * \brief Creates an expression representing the real part of the 2D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the real part of the 2D inverse FFT of a
 */
template <typename A>
auto ifft_2d_real(A&& a) -> detail::temporary_unary_helper_type<detail::ifft_real_value_type<A>, A, ifft2_real_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::ifft_real_value_type<A>, A, ifft2_real_expr>{a};
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

    return c = ifft_2d_real(a);
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
auto fft_1d_many(A&& a) -> temporary_unary_helper_type_alpha<detail::fft_value_type<A>, A, fft1_many_expr_alpha> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2, "fft_many requires at least 2D matrices");

    return temporary_unary_helper_type_alpha<detail::fft_value_type<A>, A, fft1_many_expr_alpha>{a};
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

    return c = fft_1d_many(a);
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
auto ifft_1d_many(A&& a) -> temporary_unary_helper_type_alpha<detail::fft_value_type<A>, A, ifft1_many_expr_alpha> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2, "ifft_many requires at least 2D matrices");

    return temporary_unary_helper_type_alpha<detail::fft_value_type<A>, A, ifft1_many_expr_alpha>{a};
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

    return c = ifft_1d_many(a);
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
auto fft_2d_many(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft2_many_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3, "fft_many requires at least 3D matrices");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft2_many_expr>{a};
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

    return c = fft_2d_many(a);
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
auto ifft_2d_many(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, ifft2_many_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3, "ifft_many requires at least 3D matrices");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, ifft2_many_expr>{a};
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

    return c = ifft_2d_many(a);
}

} //end of namespace etl
