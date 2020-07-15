//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the Mean Squared Error reduction
 */

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Returns the Mean Squared Error Loss
 * \param output The outputs
 * \param labels The labels
 * \return The MSE Loss of the output and labels
 */
template <typename O, typename L>
value_t<O> mse_loss(const size_t n, const O& output, const L& labels, value_t<O> scale) {
    return scale * sum((output - labels) >> (output - labels));
}

/*!
 * \brief Returns the Mean Squared Error Error
 * \param output The outputs
 * \param labels The labels
 * \return The MSE Error of the output and labels
 */
template <typename O, typename L>
value_t<O> mse_error(const size_t n, const O& output, const L& labels, value_t<O> scale) {
    return scale * asum(labels - output);
}

/*!
 * \brief Returns the Mean Squared Error Loss and Error
 * \param output The outputs
 * \param labels The labels
 * \return The MSE Error of the output and labels
 */
template <typename O, typename L>
std::pair<value_t<O>, value_t<O>> mse(const size_t n, const O& output, const L& labels, value_t<O> alpha, value_t<O> beta) {
    return std::make_pair(mse_loss(n, output, labels, alpha), mse_error(n, output, labels, beta));
}

} //end of namespace etl::impl::standard
