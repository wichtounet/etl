//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file outer.hpp
 * \brief Selector for outer product implementations
 */

#pragma once

//Include the implementations
#include "etl/impl/std/outer.hpp"
#include "etl/impl/blas/outer.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Enumeration describing the different implementations of
 * outer product
 */
enum class outer_imple {
    STD,  ///< Standard implementation
    BLAS, ///< BLAS implementation
};

/*!
 * \brief Select the outer product implementation for an expression of type A and B
 * \tparam A The type of a expression
 * \tparam B The type of b expression
 * \tparam C The type of c expression
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
cpp14_constexpr outer_imple select_outer_impl() {
    if(all_dma<A, B, C>::value){
        if (is_cblas_enabled) {
            return outer_imple::BLAS;
        } else {
            return outer_imple::STD;
        }
    }

    return outer_imple::STD;
}

/*!
 * \brief Functor for outer product
 */
struct outer_product_impl {
    /*!
     * \brief Apply the functor to a and b and store the result in c
     * \param a the left hand side
     * \param b the left hand side
     * \param c the result
     */
    template <typename A, typename B, typename C>
    static void apply(const A& a, const B& b, C&& c) {
        cpp14_constexpr auto impl = select_outer_impl<A, B, C>();

        if (impl == outer_imple::BLAS) {
            return etl::impl::blas::outer(a, b, c);
        } else {
            return etl::impl::standard::outer(a, b, c);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
