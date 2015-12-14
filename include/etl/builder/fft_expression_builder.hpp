//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
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

template <typename A>
using fft_value_type = std::conditional_t<is_complex<A>::value, value_t<A>, std::complex<value_t<A>>>;

template <typename A>
using ifft_value_type = std::conditional_t<is_complex<A>::value, value_t<A>, std::complex<value_t<A>>>;

template <typename A>
using ifft_real_value_type = std::conditional_t<is_complex<A>::value, typename value_t<A>::value_type, value_t<A>>;

} //end of namespace detail

template <typename A>
auto fft_1d(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft1_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft1_expr>{a};
}

template <typename A, typename C>
auto fft_1d(A&& a, C&& c) -> detail::forced_temporary_unary_helper_type<detail::fft_value_type<A>, A, C, fft1_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    return {a, c};
}

template <typename A>
auto ifft_1d(A&& a) -> detail::temporary_unary_helper_type<detail::ifft_value_type<A>, A, ifft1_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::ifft_value_type<A>, A, ifft1_expr>{a};
}

template <typename A, typename C>
auto ifft_1d(A&& a, C&& c) -> detail::forced_temporary_unary_helper_type<detail::ifft_value_type<A>, A, C, ifft1_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    return {a, c};
}

template <typename A>
auto ifft_1d_real(A&& a) -> detail::temporary_unary_helper_type<detail::ifft_real_value_type<A>, A, ifft1_real_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::ifft_real_value_type<A>, A, ifft1_real_expr>{a};
}

template <typename A, typename C>
auto ifft_1d_real(A&& a, C&& c) -> detail::forced_temporary_unary_helper_type<detail::ifft_real_value_type<A>, A, C, ifft1_real_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    return {a, c};
}

template <typename A>
auto fft_2d(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft2_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft2_expr>{a};
}

template <typename A, typename C>
auto fft_2d(A&& a, C&& c) -> detail::forced_temporary_unary_helper_type<detail::fft_value_type<A>, A, C, fft2_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    return {a, c};
}

template <typename A>
auto ifft_2d(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, ifft2_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, ifft2_expr>{a};
}

template <typename A, typename C>
auto ifft_2d(A&& a, C&& c) -> detail::forced_temporary_unary_helper_type<detail::fft_value_type<A>, A, C, ifft2_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    return {a, c};
}

template <typename A>
auto ifft_2d_real(A&& a) -> detail::temporary_unary_helper_type<detail::ifft_real_value_type<A>, A, ifft2_real_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");

    return detail::temporary_unary_helper_type<detail::ifft_real_value_type<A>, A, ifft2_real_expr>{a};
}

template <typename A, typename C>
auto ifft_2d_real(A&& a, C&& c) -> detail::forced_temporary_unary_helper_type<detail::ifft_real_value_type<A>, A, C, ifft2_real_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    return {a, c};
}

template <typename A>
auto fft_1d_many(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft1_many_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2, "fft_many requires at least 2D matrices");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft1_many_expr>{a};
}

template <typename A, typename C>
auto fft_1d_many(A&& a, C&& c) -> detail::forced_temporary_unary_helper_type<detail::fft_value_type<A>, A, C, fft1_many_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2 && decay_traits<C>::dimensions() >= 2, "fft_many requires at least 2D matrices");
    validate_assign(c, a);

    return {a, c};
}

template <typename A>
auto fft_2d_many(A&& a) -> detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft2_many_expr> {
    static_assert(is_etl_expr<A>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3, "fft_many requires at least 3D matrices");

    return detail::temporary_unary_helper_type<detail::fft_value_type<A>, A, fft2_many_expr>{a};
}

template <typename A, typename C>
auto fft_2d_many(A&& a, C&& c) -> detail::forced_temporary_unary_helper_type<detail::fft_value_type<A>, A, C, fft2_many_expr> {
    static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3 && decay_traits<C>::dimensions() >= 3, "fft_many requires at least 3D matrices");
    validate_assign(c, a);

    return {a, c};
}

} //end of namespace etl
