//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file value_testable.hpp
 * \brief Use CRTP technique to inject functions that test the values of the expressions or the value classes.
 */

#pragma once

namespace etl {

template <typename E>
bool is_diagonal(E&& expr);

/*!
 * \brief CRTP class to inject functions testing values of the expressions.
 *
 * This CRTP class injects test for is_finite and is_zero.
 */
template <typename D>
struct value_testable {
    using derived_t = D; ///< The derived type

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    /*!
     * \brief Indicates if the expression contains only finite values.
     * \return true if the sequence only contains finite values, false otherwise.
     */
    bool is_finite() const noexcept {
        return std::all_of(as_derived().begin(), as_derived().end(), static_cast<bool (*)(value_t<derived_t>)>(std::isfinite));
    }

    /*!
     * \brief Indicates if the expression contains only zero values.
     * \return true if the sequence only contains zero values, false otherwise.
     */
    bool is_zero() const noexcept {
        return std::all_of(as_derived().begin(), as_derived().end(), [](value_t<derived_t> v) { return v == value_t<derived_t>(0); });
    }

    /*!
     * \brief Indicates if the expression is diagonal.
     * \return true if the expression is diagonal, false otherwise.
     */
    bool is_diagonal() const noexcept {
        return etl::is_diagonal(as_derived());
    }

    /*!
     * \brief Indicates if the expression is uniform, i.e. all elements are of the same value
     * \return true if the expression is uniform, false otherwise.
     */
    bool is_uniform() const noexcept {
        return etl::is_uniform(as_derived());
    }
};

} //end of namespace etl
