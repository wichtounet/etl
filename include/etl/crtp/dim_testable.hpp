//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <algorithm>

/*
 * \file dim_testable.hpp
 * \brief Use CRTP technique to inject functions that test the dimensions.
 */

namespace etl {

/*!
 * \brief CRTP class to inject functions testing the dimensions.
 */
template <typename D>
struct dim_testable {
    using derived_t = D;

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
     * \brief Indicates if the expressions is of square dimensions (only for 2d expression)
     * \return true if the expressions is square, false otherwise.
     */
    bool is_square() const noexcept {
        return etl::is_square(as_derived());
    }

    /*!
     * \brief Indicates if the expressions is of rectangular dimensions (only for 2d expression)
     * \return true if the expressions is rectangular, false otherwise.
     */
    bool is_rectangular() const noexcept {
        return etl::is_rectangular(as_derived());
    }

    /*!
     * \brief Indicates if the expressions is of square dimensions, ignoring the first dimension (only for 3d expression)
     * \return true if the expressions is sub square, false otherwise.
     */
    bool is_sub_square() const noexcept {
        return etl::is_sub_square(as_derived());
    }

    /*!
     * \brief Indicates if the expressions is of rectangular dimensions, ignoring the first dimension (only for 3d expression)
     * \return true if the expressions is sub rectangular, false otherwise.
     */
    bool is_sub_rectangular() const noexcept {
        return etl::is_sub_rectangular(as_derived());
    }

    /*!
     * \brief Indicates if the given expression is a symmetric matrix or not.
     * \return true if the given expression is a symmetric matrix, false otherwise.
     */
    bool is_symmetric() const noexcept {
        return etl::is_symmetric(as_derived());
    }
};

} //end of namespace etl
