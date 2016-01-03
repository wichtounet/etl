//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <cmath>

#include "cpp_utils/tmp.hpp"

namespace etl {

namespace math {

/*!
 * \brief Return the logistic sigmoid of x
 * \param x The value
 * \return The logistic sigmoid of x
 */
inline float logistic_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/*!
 * \brief Return the logistic sigmoid of x
 * \param x The value
 * \return The logistic sigmoid of x
 */
inline double logistic_sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/*!
 * \brief Return the softplus of x
 * \param x The value
 * \return The softplus of x
 */
inline float softplus(float x) {
    return std::log1p(std::exp(x));
}

/*!
 * \brief Return the softplus of x
 * \param x The value
 * \return The softplus of x
 */
inline double softplus(double x) {
    return std::log1p(std::exp(x));
}

/*!
 * \brief Return the sign of x
 * \param v The value
 * \return The sign of x
 */
template <typename W>
inline constexpr double sign(W v) noexcept {
    return v == W(0) ? W(0) : (v > W(0) ? W(1) : W(-1));
}

} //end of namespace math

} //end of namespace etl
