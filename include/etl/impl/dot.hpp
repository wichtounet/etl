//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file dot.hpp
 * \brief Selector for the dot product
 */

#pragma once

//Include the implementations
#include "etl/impl/std/dot.hpp"
#include "etl/impl/blas/dot.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Enumeration describing the different implementations of dot
 */
enum class dot_imple {
    STD,  ///< Standard implementation
    BLAS, ///< BLAS implementation
};

/*!
 * \brief Select the dot implementation for an expression of type A and B
 * \tparam A The type of lhs expression
 * \tparam B The type of rhs expression
 * \return The implementation to use
 */
template <typename A, typename B>
cpp14_constexpr dot_imple select_dot_impl() {
    if(all_dma<A, B>::value){
        if (is_cblas_enabled) {
            return dot_imple::BLAS;
        } else {
            return dot_imple::STD;
        }
    }

    return dot_imple::STD;
}

/*!
 * \brief Functor for dot product
 */
struct dot_impl {
    /*!
     * \brief Apply the functor to a and b
     * \param a the left hand side
     * \param b the left hand side
     * \return the dot product of a and b
     */
    template <typename A, typename B>
    static value_t<A> apply(const A& a, const B& b) {
        cpp14_constexpr auto impl = select_dot_impl<A, B>();

        if (impl == dot_imple::BLAS) {
            return etl::impl::blas::dot(a, b);
        } else {
            return etl::impl::standard::dot(a, b);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
