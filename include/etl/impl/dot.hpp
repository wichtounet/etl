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
#include "etl/impl/sse/dot.hpp"
#include "etl/impl/avx/dot.hpp"
#include "etl/impl/blas/dot.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Select the dot implementation for an expression of type A and B
 *
 * This does not take the local context into account
 *
 * \tparam A The type of lhs expression
 * \tparam B The type of rhs expression
 * \return The implementation to use
 */
template <typename A, typename B>
cpp14_constexpr etl::dot_impl select_default_dot_impl() {
    if (all_dma<A, B>::value) {
        if (is_cblas_enabled) {
            return etl::dot_impl::BLAS;
        } else if (avx_enabled) {
            return etl::dot_impl::AVX;
        } else if (sse3_enabled) {
            return etl::dot_impl::SSE;
        } else {
            return etl::dot_impl::STD;
        }
    }

    return etl::dot_impl::STD;
}

/*!
 * \brief Select the dot implementation for an expression of type A and B
 * \tparam A The type of lhs expression
 * \tparam B The type of rhs expression
 * \return The implementation to use
 */
template <typename A, typename B>
etl::dot_impl select_dot_impl() {
    if (local_context().dot_selector.forced) {
        auto forced = local_context().dot_selector.impl;

        switch (forced) {
            //BLAS cannot always be used
            case dot_impl::BLAS:
                if (!is_cblas_enabled || !all_dma<A, B>::value) {
                    std::cerr << "Forced selection to BLAS dot implementation, but not possible for this expression" << std::endl;
                    return select_default_dot_impl<A, B>();
                }

                return forced;

            //AVX cannot always be used
            case dot_impl::AVX:
                if (!sse3_enabled || !all_dma<A, B>::value) {
                    std::cerr << "Forced selection to AVX dot implementation, but not possible for this expression" << std::endl;
                    return select_default_dot_impl<A, B>();
                }

                return forced;

            //SSE cannot always be used
            case dot_impl::SSE:
                if (!sse3_enabled || !all_dma<A, B>::value) {
                    std::cerr << "Forced selection to SSE dot implementation, but not possible for this expression" << std::endl;
                    return select_default_dot_impl<A, B>();
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_dot_impl<A, B>();
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
        auto impl = select_dot_impl<A, B>();

        if (impl == etl::dot_impl::BLAS) {
            return etl::impl::blas::dot(a, b);
        } else if (impl == etl::dot_impl::AVX) {
            return etl::impl::avx::dot(a.direct(), b.direct());
        } else if (impl == etl::dot_impl::SSE) {
            return etl::impl::sse::dot(a.direct(), b.direct());
        } else {
            return etl::impl::standard::dot(a, b);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
