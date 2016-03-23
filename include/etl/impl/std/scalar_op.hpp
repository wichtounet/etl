//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the scalar operations
 */

#pragma once

namespace etl {

namespace impl {

namespace standard {

/*!
 * \brief Add the rhs scalar value to each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename TT>
void scalar_add(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] += rhs;
    }
}

/*!
 * \brief Subtract the rhs scalar value from each element of lhs
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename TT>
void scalar_sub(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] -= rhs;
    }
}

/*!
 * \brief Multiply each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename TT>
void scalar_mul(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] *= rhs;
    }
}

/*!
 * \brief Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename TT>
void scalar_div(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] /= rhs;
    }
}

/*!
 * \brief Modulo Divide each element of lhs by the rhs scalar value
 * \param lhs The matrix
 * \param rhs The scalar
 */
template <typename TT>
void scalar_mod(TT&& lhs, value_t<TT> rhs) {
    auto m = lhs.memory_start();

    for (std::size_t i = 0; i < size(lhs); ++i) {
        m[i] %= rhs;
    }
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
