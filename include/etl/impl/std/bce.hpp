//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the Binary Cross Entropy reduction
 */

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Returns the Binary Cross Entropy Loss
 * \param output The outputs
 * \param labels The labels
 * \return The BCE Loss of the output and labels
 */
template <typename O, typename L>
value_t<O> bce_loss(const O& output, const L& labels, value_t<O> scale) {
    return scale * sum((labels >> log(output)) + ((1.0 - labels) >> log(1.0 - output)));
}

/*!
 * \brief Returns the Binary Cross Entropy Error
 * \param output The outputs
 * \param labels The labels
 * \return The BCE Error of the output and labels
 */
template <typename O, typename L>
value_t<O> bce_error(const O& output, const L& labels, value_t<O> scale) {
    return scale * asum(labels - output);
}

/*!
 * \brief Returns the Binary Cross Entropy Loss and Error
 * \param output The outputs
 * \param labels The labels
 * \return The BCE Error of the output and labels
 */
template <typename O, typename L>
std::pair<value_t<O>, value_t<O>> bce(const O& output, const L& labels, value_t<O> alpha, value_t<O> beta) {
    return std::make_pair(bce_loss(output, labels, alpha), bce_error(output, labels, beta));
}

} //end of namespace etl::impl::standard
