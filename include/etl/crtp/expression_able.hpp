//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file expression_able.hpp
 * \brief Use CRTP technique to inject functions creating new expressions.
 */

#pragma once

namespace etl {

/*!
 * \brief CRTP class to inject functions creating new expressions.
 *
 * All the functions returns new expressions, no modificatio of the expression is done.j
 */
template <typename D>
struct expression_able {
    using derived_t = D; ///< The derived type

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Scale the expression by a scalar factor or another expression.
     * \return A new expression representing the scaling of this expression.
     */
    template <typename E>
    auto scale(E&& e) {
        return etl::scale(as_derived(), std::forward<E>(e));
    }

    /*!
     * \brief Flip the matrix horizontally and vertically.
     * \return A new expression representing the horizontal and vertical flipping of the matrix.
     */
    auto fflip() {
        return etl::fflip(as_derived());
    }

    /*!
     * \brief Flip the matrix horizontally.
     * \return A new expression representing the horizontal flipping of the matrix.
     */
    auto hflip() {
        return etl::hflip(as_derived());
    }

    /*!
     * \brief Flip the matrix vertically.
     * \return A new expression representing the vertical flipping of the matrix.
     */
    auto vflip() {
        return etl::vflip(as_derived());
    }

    /*!
     * \brief Transpose the matrix
     * \return A new expression representing the transposition of this expression.
     */
    auto transpose() {
        return etl::transpose(as_derived());
    }

    /*!
     * \brief Extract the real part of a complex expression
     * \return A new expression representing only real part of this expression.
     */
    auto real() {
        return etl::real(as_derived());
    }

    /*!
     * \brief Extract the imag part of a complex expression
     * \return A new expression representing only imag part of this expression.
     */
    auto imag() {
        return etl::imag(as_derived());
    }

    /*!
     * \brief Returns a new expression containg the conjugate of each value of the expression.
     * \return A new expression containing the conjugate of each complex value of the expression.
     */
    auto conj() {
        return etl::conj(as_derived());
    }
};

} //end of namespace etl
