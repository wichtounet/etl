//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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
#include "etl/impl/cublas/dot.hpp"
#include "etl/impl/vec/dot.hpp"

namespace etl::detail {

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
constexpr etl::dot_impl select_default_dot_impl() {
    if (all_dma<A, B> && cblas_enabled) {
        return etl::dot_impl::BLAS;
    }

    if (vec_enabled && all_vectorizable<vector_mode, A, B> && std::same_as<default_intrinsic_type<value_t<A>>, default_intrinsic_type<value_t<B>>>) {
        return etl::dot_impl::VEC;
    }

    return etl::dot_impl::STD;
}

#ifdef ETL_MANUAL_SELECT

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
            //CUBLAS cannot always be used
            case dot_impl::CUBLAS:
                if (!cublas_enabled || !all_dma<A, B>) {
                    std::cerr << "Forced selection to CUBLAS dot implementation, but not possible for this expression" << std::endl;
                    return select_default_dot_impl<A, B>();
                }

                return forced;

            //BLAS cannot always be used
            case dot_impl::BLAS:
                if (!cblas_enabled || !all_dma<A, B>) {
                    std::cerr << "Forced selection to BLAS dot implementation, but not possible for this expression" << std::endl;
                    return select_default_dot_impl<A, B>();
                }

                return forced;

            //VEC cannot always be used
            case dot_impl::VEC:
                if (!vec_enabled || !decay_traits<A>::template vectorizable<vector_mode> || !decay_traits<B>::template vectorizable<vector_mode>) {
                    std::cerr << "Forced selection to VEC dot implementation, but not possible for this expression" << std::endl;
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

#else

/*!
 * \brief Select the dot implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename A, typename B>
constexpr etl::dot_impl select_dot_impl() {
    return select_default_dot_impl<A, B>();
}

#endif

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
        constexpr_select auto impl = select_dot_impl<A, B>();

        if
            constexpr_select(impl == etl::dot_impl::BLAS) {
                inc_counter("impl:blas");
                return etl::impl::blas::dot(a, b);
            }
        else if
            constexpr_select(impl == etl::dot_impl::CUBLAS) {
                inc_counter("impl:cublas");
                return etl::impl::cublas::dot(a, b);
            }
        else if
            constexpr_select(impl == etl::dot_impl::VEC) {
                inc_counter("impl:vec");
                return etl::impl::vec::dot(a, b);
            }
        else {
            inc_counter("impl:std");
            return etl::impl::standard::dot(a, b);
        }
    }
};

} //end of namespace etl::detail
