//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains all the operators and functions to build expressions
 * representing mathematical functions.
 */

#pragma once

namespace etl {

/*!
 * \brief Apply square root on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the square root of each value of the given expression
 */
template <typename E>
auto sqrt(E&& value) -> detail::unary_helper<E, sqrt_unary_op> {
    static_assert(is_etl_expr<E>, "etl::sqrt can only be used on ETL expressions");
    return detail::unary_helper<E, sqrt_unary_op>{value};
}

/*!
 * \brief Apply inverse square root on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the inverse square root of each value of the given expression
 */
template <typename E>
auto invsqrt(E&& value) -> detail::unary_helper<E, invsqrt_unary_op> {
    static_assert(is_etl_expr<E>, "etl::invsqrt can only be used on ETL expressions");
    return detail::unary_helper<E, invsqrt_unary_op>{value};
}

/*!
 * \brief Apply cubic root on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the cubic root of each value of the given expression
 */
template <typename E>
auto cbrt(E&& value) -> detail::unary_helper<E, cbrt_unary_op> {
    static_assert(is_etl_expr<E>, "etl::cbrt can only be used on ETL expressions");
    return detail::unary_helper<E, cbrt_unary_op>{value};
}

/*!
 * \brief Apply inverse cubic root on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the inverse cubic root of each value of the given expression
 */
template <typename E>
auto invcbrt(E&& value) -> detail::unary_helper<E, invcbrt_unary_op> {
    static_assert(is_etl_expr<E>, "etl::invcbrt can only be used on ETL expressions");
    return detail::unary_helper<E, invcbrt_unary_op>{value};
}

/*!
 * \brief Apply logarithm (base e) on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the logarithm (base e) of each value of the given expression
 */
template <typename E>
auto log(E&& value) -> detail::unary_helper<E, log_unary_op> {
    static_assert(is_etl_expr<E>, "etl::log can only be used on ETL expressions");
    return detail::unary_helper<E, log_unary_op>{value};
}

/*!
 * \brief Apply logarithm (base 2) on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the logarithm (base 2) of each value of the given expression
 */
template <typename E>
auto log2(E&& value) -> detail::unary_helper<E, log2_unary_op> {
    static_assert(is_etl_expr<E>, "etl::log2 can only be used on ETL expressions");
    return detail::unary_helper<E, log2_unary_op>{value};
}

/*!
 * \brief Apply logarithm (base 10) on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the logarithm (base 10) of each value of the given expression
 */
template <typename E>
auto log10(E&& value) -> detail::unary_helper<E, log10_unary_op> {
    static_assert(is_etl_expr<E>, "etl::log10 can only be used on ETL expressions");
    return detail::unary_helper<E, log10_unary_op>{value};
}

/*!
 * \brief Apply tangent on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the tangent of each value of the given expression
 */
template <typename E>
auto tan(E&& value) -> detail::unary_helper<E, tan_unary_op> {
    static_assert(is_etl_expr<E>, "etl::tan can only be used on ETL expressions");
    return detail::unary_helper<E, tan_unary_op>{value};
}

/*!
 * \brief Apply cosinus on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the cosinus of each value of the given expression
 */
template <typename E>
auto cos(E&& value) -> detail::unary_helper<E, cos_unary_op> {
    static_assert(is_etl_expr<E>, "etl::cos can only be used on ETL expressions");
    return detail::unary_helper<E, cos_unary_op>{value};
}

/*!
 * \brief Apply sinus on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the sinus of each value of the given expression
 */
template <typename E>
auto sin(E&& value) -> detail::unary_helper<E, sin_unary_op> {
    static_assert(is_etl_expr<E>, "etl::sin can only be used on ETL expressions");
    return detail::unary_helper<E, sin_unary_op>{value};
}

/*!
 * \brief Apply hyperbolic tangent on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the hyperbolic tangent of each value of the given expression
 */
template <typename E>
auto tanh(E&& value) -> detail::unary_helper<E, tanh_unary_op> {
    static_assert(is_etl_expr<E>, "etl::tanh can only be used on ETL expressions");
    return detail::unary_helper<E, tanh_unary_op>{value};
}

/*!
 * \brief Apply hyperbolic cosinus on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the hyperbolic cosinus of each value of the given expression
 */
template <typename E>
auto cosh(E&& value) -> detail::unary_helper<E, cosh_unary_op> {
    static_assert(is_etl_expr<E>, "etl::cosh can only be used on ETL expressions");
    return detail::unary_helper<E, cosh_unary_op>{value};
}

/*!
 * \brief Apply hyperbolic sinus on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the hyperbolic sinus of each value of the given expression
 */
template <typename E>
auto sinh(E&& value) -> detail::unary_helper<E, sinh_unary_op> {
    static_assert(is_etl_expr<E>, "etl::sinh can only be used on ETL expressions");
    return detail::unary_helper<E, sinh_unary_op>{value};
}

/*!
 * \brief Apply exponential on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the exponential of each value of the given expression
 */
template <typename E>
auto exp(E&& value) -> detail::unary_helper<E, exp_unary_op> {
    static_assert(is_etl_expr<E>, "etl::exp can only be used on ETL expressions");
    return detail::unary_helper<E, exp_unary_op>{value};
}

/*!
 * \brief Apply sign on each value of the given expression
 * \param value The ETL expression
 * \return an expression representing the sign of each value of the given expression
 */
template <typename E>
auto sign(E&& value) -> detail::unary_helper<E, sign_unary_op> {
    static_assert(is_etl_expr<E>, "etl::sign can only be used on ETL expressions");
    return detail::unary_helper<E, sign_unary_op>{value};
}

/*!
 * \brief Performs the identiy function on the ETL expression.
 * \param value The ETL expression
 * \return The same value, perfectly forwardd
 */
template <typename E>
decltype(auto) identity(E&& value) {
    return std::forward<E>(value);
}

/*!
 * \brief Return the derivative of the identiy function for the given value.
 * \param value The ETL expression
 * \return 1.0
 */
template <typename E>
auto identity_derivative([[maybe_unused]] E&& value) {
    return 1.0;
}

//Note: Use of decltype here should not be necessary, but g++ does
//not like it without it for some reason

/*!
 * \brief Return the logistic sigmoid of the given ETL expression.
 * \param value The ETL expression
 * \return An ETL expression representing the logistic sigmoid of the input.
 */
template <typename E>
auto sigmoid(const E& value) -> detail::unary_helper<E, sigmoid_unary_op> {
    static_assert(is_etl_expr<E>, "etl::fast_sigmoid can only be used on ETL expressions");
    return detail::unary_helper<E, sigmoid_unary_op>{value};
}

/*!
 * \brief Return the relu activation of the given ETL expression.
 * \param value The ETL expression
 * \return An ETL expression representing the relu activation of the input.
 */
template <typename E>
auto relu(const E& value) -> detail::unary_helper<E, relu_unary_op> {
    static_assert(is_etl_expr<E>, "etl::relu can only be used on ETL expressions");
    return detail::unary_helper<E, relu_unary_op>{value};
}

/*!
 * \brief Return the derivative of the logistic sigmoid of the given ETL expression.
 * \param value The ETL expression
 * \return An ETL expression representing the derivative of the logistic sigmoid of the input.
 */
template <typename E>
auto sigmoid_derivative(E&& value) -> decltype(sigmoid(value) >> (1.0 - sigmoid(value))) {
    static_assert(is_etl_expr<E>, "etl::sigmoid_derivative can only be used on ETL expressions");
    return sigmoid(value) >> (1.0 - sigmoid(value));
}

/*!
 * \brief Return a fast approximation of the logistic sigmoid of the given ETL expression.
 *
 * This function is faster than the sigmoid function and has an acceptable precision.
 *
 * \param value The ETL expression
 * \return An ETL expression representing a fast approximation of the logistic sigmoid of the input.
 */
template <typename E>
auto fast_sigmoid(const E& value) -> detail::unary_helper<E, fast_sigmoid_unary_op> {
    static_assert(is_etl_expr<E>, "etl::fast_sigmoid can only be used on ETL expressions");
    return detail::unary_helper<E, fast_sigmoid_unary_op>{value};
}

/*!
 * \brief Return an hard approximation of the logistic sigmoid of the given ETL expression.
 *
 * This function is much faster than the sigmoid, but it's precision is very low.
 *
 * \param x The ETL expression
 * \return An ETL expression representing an hard approximation of the logistic sigmoid of the input.
 */
template <typename E>
auto hard_sigmoid(E&& x) -> decltype(etl::clip(x * 0.2 + 0.5, 0.0, 1.0)) {
    static_assert(is_etl_expr<E>, "etl::hard_sigmoid can only be used on ETL expressions");
    return etl::clip(x * 0.2 + 0.5, 0.0, 1.0);
}

/*!
 * \brief Return the softmax function of the given ETL expression.
 * \param e The ETL expression
 * \return An ETL expression representing the softmax function of the input.
 */
template <typename E>
auto softmax(E&& e) {
    static_assert(is_etl_expr<E>, "etl::softmax can only be used on ETL expressions");

    if constexpr (etl::is_1d<E>) {
        return exp(e) / sum(exp(e));
    } else {
        return batch_softmax_expr<detail::build_type<E>, false>{e};
    }
}

/*!
 * \brief Returns the softmax function of the given ETL expression.
 * This version is implemented so that numerical stability is preserved.
 * \param e The ETL expression
 * \return An ETL expression representing the softmax function of the input.
 */
template <typename E>
auto stable_softmax(E&& e) {
    static_assert(is_etl_expr<E>, "etl::softmax can only be used on ETL expressions");

    if constexpr (etl::is_1d<E>) {
        auto m = max(e);
        return exp(e - m) / sum(exp(e - m));
    } else {
        return batch_softmax_expr<detail::build_type<E>, true>{e};
    }
}

/*!
 * \brief Return the derivative of the softmax function of the given ETL expression.
 * \param e The ETL expression
 * \return An ETL expression representing the derivative of the softmax function of the input.
 */
template <typename E>
auto softmax_derivative([[maybe_unused]] E&& e) {
    return 1.0;
}

/*!
 * \brief Return the softplus of the given ETL expression.
 * \param value The ETL expression
 * \return An ETL expression representing the softplus of the input.
 */
template <typename E>
auto softplus(E&& value) -> detail::unary_helper<E, softplus_unary_op> {
    static_assert(is_etl_expr<E>, "etl::softplus can only be used on ETL expressions");
    return detail::unary_helper<E, softplus_unary_op>{value};
}

/*!
 * \brief Apply Bernoulli sampling to the values of the expression
 * \param value the expression to sample
 * \return an expression representing the Bernoulli sampling of the given expression
 */
template <typename E>
auto bernoulli(const E& value) -> detail::unary_helper<E, bernoulli_unary_op> {
    static_assert(is_etl_expr<E>, "etl::bernoulli can only be used on ETL expressions");
    return detail::unary_helper<E, bernoulli_unary_op>{value};
}

/*!
 * \brief Apply Bernoulli sampling to the values of the expression
 * \param value the expression to sample
 * \return an expression representing the Bernoulli sampling of the given expression
 */
template <typename E, typename G>
auto bernoulli(G& g, E&& value) {
    return detail::make_stateful_unary_expr<E, bernoulli_unary_g_op<G, value_t<E>>>(value, g);
}

/*!
 * \brief Apply Bernoulli sampling to the values of the expression
 * \param value the expression to sample
 * \return an expression representing the Bernoulli sampling of the given expression
 */
template <typename E>
auto state_bernoulli(const E& value) {
    static_assert(is_etl_expr<E>, "etl::bernoulli can only be used on ETL expressions");
    return detail::make_stateful_unary_expr<E, state_bernoulli_unary_op<value_t<E>>>(value);
}

/*!
 * \brief Apply Bernoulli sampling to the values of the expression
 * \param value the expression to sample
 * \return an expression representing the Bernoulli sampling of the given expression
 */
template <typename E>
auto state_bernoulli(const E& value, const std::shared_ptr<void*> & states) {
    static_assert(is_etl_expr<E>, "etl::bernoulli can only be used on ETL expressions");
    return detail::make_stateful_unary_expr<E, state_bernoulli_unary_op<value_t<E>>>(value, states);
}

/*!
 * \brief Apply Bernoulli sampling to the values of the expression
 * \param value the expression to sample
 * \return an expression representing the Bernoulli sampling of the given expression
 */
template <typename E, typename G, cpp_enable_iff(is_etl_expr<E>)>
auto state_bernoulli(G& g, E&& value) {
    return detail::make_stateful_unary_expr<E, state_bernoulli_unary_g_op<G, value_t<E>>>(value, g);
}

/*!
 * \brief Apply Bernoulli sampling to the values of the expression
 * \param value the expression to sample
 * \return an expression representing the Bernoulli sampling of the given expression
 */
template <typename E, typename G, cpp_enable_iff(is_etl_expr<E>)>
auto state_bernoulli(G& g, E&& value, const std::shared_ptr<void*> & states) {
    return detail::make_stateful_unary_expr<E, state_bernoulli_unary_g_op<G, value_t<E>>>(value, g, states);
}

/*!
 * \brief Apply Reverse Bernoulli sampling to the values of the expression
 * \param value the expression to sample
 * \return an expression representing the Reverse Bernoulli sampling of the given expression
 */
template <typename E>
auto r_bernoulli(const E& value) -> detail::unary_helper<E, reverse_bernoulli_unary_op> {
    static_assert(is_etl_expr<E>, "etl::r_bernoulli can only be used on ETL expressions");
    return detail::unary_helper<E, reverse_bernoulli_unary_op>{value};
}

/*!
 * \brief Apply Reverse Bernoulli sampling to the values of the expression
 * \param value the expression to sample
 * \return an expression representing the Reverse Bernoulli sampling of the given expression
 */
template <typename E, typename G>
auto r_bernoulli(G& g, E&& value) {
    return detail::make_stateful_unary_expr<E, reverse_bernoulli_unary_g_op<G, value_t<E>>>(value, g);
}

/*!
 * \brief Return the derivative of the tanh function of the given ETL expression.
 * \param value The ETL expression
 * \return An ETL expression representing the derivative of the tanh function of the input.
 */
template <typename E>
auto tanh_derivative(E&& value) -> decltype(1.0 - (tanh(value) >> tanh(value))) {
    static_assert(is_etl_expr<E>, "etl::tanh_derivative can only be used on ETL expressions");
    return 1.0 - (tanh(value) >> tanh(value));
}

/*!
 * \brief Return the derivative of the relu function of the given ETL expression.
 * \param value The ETL expression
 * \return An ETL expression representing the derivative of the relu function of the input.
 */
template <typename E>
auto relu_derivative(const E& value) -> detail::unary_helper<E, relu_derivative_op> {
    static_assert(is_etl_expr<E>, "etl::relu_derivative can only be used on ETL expressions");
    return detail::unary_helper<E, relu_derivative_op>{value};
}

} //end of namespace etl
