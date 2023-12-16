//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*
 * \file dim_testable.hpp
 * \brief Use CRTP technique to inject functions that test the dimensions.
 */

#pragma once

namespace etl {

template <typename E>
bool is_symmetric(E&& expr);

template <typename E>
bool is_triangular(E&& expr);

template <typename E>
bool is_lower_triangular(E&& expr);

template <typename E>
bool is_upper_triangular(E&& expr);

template <typename E>
bool is_strictly_lower_triangular(E&& expr);

template <typename E>
bool is_strictly_upper_triangular(E&& expr);

template <typename E>
bool is_uni_lower_triangular(E&& expr);

template <typename E>
bool is_uni_upper_triangular(E&& expr);

/*!
 * \brief CRTP class to inject functions testing the dimensions.
 */
template <typename D>
struct dim_testable {
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

    /*!
     * \brief Indicates if the given expression is a lower triangular matrix or not.
     * \return true if the given expression is a lower triangular matrix, false otherwise.
     */
    bool is_lower_triangular() const noexcept {
        return etl::is_lower_triangular(as_derived());
    }

    /*!
     * \brief Indicates if the given expression is a uni lower triangular matrix or not.
     * \return true if the given expression is a uni lower triangular matrix, false otherwise.
     */
    bool is_uni_lower_triangular() const noexcept {
        return etl::is_uni_lower_triangular(as_derived());
    }

    /*!
     * \brief Indicates if the given expression is a strictly lower triangular matrix or not.
     * \return true if the given expression is a strictly lower triangular matrix, false otherwise.
     */
    bool is_strictly_lower_triangular() const noexcept {
        return etl::is_strictly_lower_triangular(as_derived());
    }

    /*!
     * \brief Indicates if the given expression is a upper triangular matrix or not.
     * \return true if the given expression is a upper triangular matrix, false otherwise.
     */
    bool is_upper_triangular() const noexcept {
        return etl::is_upper_triangular(as_derived());
    }

    /*!
     * \brief Indicates if the given expression is a uni upper triangular matrix or not.
     * \return true if the given expression is a uni upper triangular matrix, false otherwise.
     */
    bool is_uni_upper_triangular() const noexcept {
        return etl::is_uni_upper_triangular(as_derived());
    }

    /*!
     * \brief Indicates if the given expression is a strictly upper triangular matrix or not.
     * \return true if the given expression is a strictly upper triangular matrix, false otherwise.
     */
    bool is_strictly_upper_triangular() const noexcept {
        return etl::is_strictly_upper_triangular(as_derived());
    }

    /*!
     * \brief Indicates if the given expression is a triangular matrix or not.
     * \return true if the given expression is a triangular matrix, false otherwise.
     */
    bool is_triangular() const noexcept {
        return etl::is_triangular(as_derived());
    }
};

} //end of namespace etl
