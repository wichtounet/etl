//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the Categorical Cross Entropy reduction
 */

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Compute the Categorical Cross Entropy loss of the input in the given expression
 * \param input The input expression
 * \return the sum
 */
template <typename O, typename L>
value_t<O> cce_loss(const O& output, const L& labels, value_t<O> scale) {
    return scale * etl::sum(log(output) >> labels);
}

/*!
 * \brief Compute the Categorical Cross Entropy error of the input in the given expression
 * \param input The input expression
 * \return the sum
 */
template <typename O, typename L>
value_t<O> cce_error(const O& output, const L& labels, value_t<O> scale) {
    return scale * sum(min(abs(argmax(labels) - argmax(output)), 1.0));
}

/*!
 * \brief Returns the Binary Cross Entropy Loss and Error
 * \param output The outputs
 * \param labels The labels
 * \return The CCE Error of the output and labels
 */
template <typename O, typename L>
std::pair<value_t<O>, value_t<O>> cce(const O& output, const L& labels, value_t<O> alpha, value_t<O> beta) {
    return std::make_pair(cce_loss(output, labels, alpha), cce_error(output, labels, beta));
}

} //end of namespace etl::impl::standard
