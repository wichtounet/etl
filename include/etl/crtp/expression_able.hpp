//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/compat.hpp"

/*!
 * \file expression_able.hpp
 * \brief Use CRTP technique to inject functions creating new expressions.
 */

namespace etl {

/*!
 * \brief CRTP class to inject functions creating new expressions.
 *
 * All the functions returns new expressions, no modificatio of the expression is done.j
 */
template<typename D>
struct expression_able {
    using derived_t = D;

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
    template<typename E>
    auto scale(E&& e){
        return etl::scale(as_derived(), std::forward<E>(e));
    }

    /*!
     * \brief Flip the matrix horizontally and vertically.
     * \return A new expression representing the horizontal and vertical flipping of the matrix.
     */
    ETL_DEBUG_AUTO_TRICK auto fflip(){
        return etl::fflip(as_derived());
    }

    /*!
     * \brief Flip the matrix horizontally.
     * \return A new expression representing the horizontal flipping of the matrix.
     */
    ETL_DEBUG_AUTO_TRICK auto hflip(){
        return etl::hflip(as_derived());
    }

    /*!
     * \brief Flip the matrix vertically.
     * \return A new expression representing the vertical flipping of the matrix.
     */
    ETL_DEBUG_AUTO_TRICK auto vflip(){
        return etl::vflip(as_derived());
    }

    /*!
     * \brief Transpose the matrix
     * \return A new expression representing the transposition of this expression.
     */
    ETL_DEBUG_AUTO_TRICK auto transpose(){
        return etl::transpose(as_derived());
    }
};

} //end of namespace etl
