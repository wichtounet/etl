//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementations of inplace matrix transposition
 */

#pragma once

//Include the implementations
#include "etl/impl/std/transpose.hpp"
#include "etl/impl/blas/transpose.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Select the transpose implementation for an expression of type A and C
 *
 * This does not take the local context into account.
 *
 * \tparam A The type of rhs expression
 * \tparam C The type of lhs expression
 * \return The implementation to use
 */
template <typename A, typename C>
cpp14_constexpr transpose_impl select_default_transpose_impl() {
    if(all_dma<A, C>::value && all_floating<A, C>::value){
        if (is_mkl_enabled) {
            return transpose_impl::MKL;
        } else {
            return transpose_impl::STD;
        }
    }

    return transpose_impl::STD;
}

/*!
 * \brief Select the transpose implementation for an expression of type A and C
 * \tparam A The type of rhs expression
 * \tparam C The type of lhs expression
 * \return The implementation to use
 */
template <typename A, typename C>
transpose_impl select_transpose_impl() {
    if(local_context().transpose_selector.forced){
        auto forced = local_context().transpose_selector.impl;

        switch (forced) {
            //MKL cannot always be used
            case transpose_impl::MKL:
                if(!is_mkl_enabled || !all_dma<A, C>::value || !all_floating<A, C>::value){
                    std::cerr << "Forced selection to MKL transpose implementation, but not possible for this expression" << std::endl;
                    return select_default_transpose_impl<A, C>();
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_transpose_impl<A, C>();
}

struct inplace_square_transpose {
    template <typename C>
    static void apply(C&& c) {
        const auto impl = select_transpose_impl<C, C>();

        if(impl == transpose_impl::MKL){
            etl::impl::blas::inplace_square_transpose(c);
        } else {
            etl::impl::standard::inplace_square_transpose(c);
        }
    }
};

struct inplace_rectangular_transpose {
    template <typename C>
    static void apply(C&& c) {
        const auto impl = select_transpose_impl<C, C>();

        if(impl == transpose_impl::MKL){
            etl::impl::blas::inplace_rectangular_transpose(c);
        } else {
            etl::impl::standard::inplace_rectangular_transpose(c);
        }
    }
};

struct transpose {
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        const auto impl = select_transpose_impl<A, C>();

        if(impl == transpose_impl::MKL){
            etl::impl::blas::transpose(a, c);
        } else {
            etl::impl::standard::transpose(a, c);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
