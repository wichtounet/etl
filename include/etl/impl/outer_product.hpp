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
 * \brief Select the outer product implementation for an expression of type A and B
 *
 * This does not take the local context into account
 *
 * \tparam A The type of a expression
 * \tparam B The type of b expression
 * \tparam C The type of c expression
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
cpp14_constexpr etl::outer_impl select_default_outer_impl() {
    if(all_dma<A, B, C>::value){
        if (is_cblas_enabled) {
            return etl::outer_impl::BLAS;
        } else {
            return etl::outer_impl::STD;
        }
    }

    return etl::outer_impl::STD;
}

/*!
 * \brief Select the outer product implementation for an expression of type A and B
 * \tparam A The type of a expression
 * \tparam B The type of b expression
 * \tparam C The type of c expression
 * \return The implementation to use
 */
template <typename A, typename B, typename C>
cpp14_constexpr etl::outer_impl select_outer_impl() {
    if(local_context().outer_selector.forced){
        auto forced = local_context().outer_selector.impl;

        switch (forced) {
            //AVX cannot always be used
            case outer_impl::BLAS:
                if(!is_cblas_enabled || !all_dma<A, B, C>::value){
                    std::cerr << "Forced selection to BLAS outer implementation, but not possible for this expression" << std::endl;
                    return select_default_outer_impl<A, B, C>();
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_outer_impl<A, B, C>();
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

        if (impl == etl::outer_impl::BLAS) {
            return etl::impl::blas::outer(a, b, c);
        } else {
            return etl::impl::standard::outer(a, b, c);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
