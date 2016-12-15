//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

/*!
 * \file comparable.hpp
 * \brief Use CRTP technique to inject comparison operators to expressions and value classes.
 */

namespace etl {

/*!
 * \brief CRTP class to inject comparison operators.
 */
template <typename D>
struct comparable {
    using derived_t = D; ///< The derived type

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Returns a const reference to the derived object, i.e. the object using the CRTP injector.
     * \return a const reference to the derived object.
     */
    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

    /*!
     * \brief Compare the expression with another expression.
     *
     * \return true if the expressions contains the same sequence of values, false othwerise.
     */
    template <typename E>
    bool operator==(E&& rhs){
        auto& lhs = as_derived();

        // Both expressions must have the same number of dimensions
        if (etl::dimensions(lhs) != etl::dimensions(rhs)) {
            return false;
        }

        // The dimensions must be the same
        for(std::size_t i = 0; i < etl::dimensions(rhs); ++i){
            if(etl::dim(lhs, i) != etl::dim(rhs, i)){
                return false;
            }
        }

        // At this point, the values are necessary for the comparison
        etl::force(lhs);
        etl::force(std::forward<E>(rhs));

        // Note: Ideally, we should use std::equal, but this is significantly
        // faster to compile

        for(size_t i = 0; i < etl::size(lhs); ++i){
            if(lhs[i] != rhs[i]){
                return false;
            }
        }

        return true;
    }

    /*!
     * \brief Compare the expression with another expression for inequality.
     *
     * \return false if the expressions contains the same sequence of values, true othwerise.
     */
    template <typename E>
    bool operator!=(E&& rhs){
        return !(as_derived() == std::forward<E>(rhs));
    }
};

} //end of namespace etl
