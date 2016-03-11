//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file scalar_op.hpp
 * \brief Implementations of vector/matrix scalar operations.
 */

#pragma once

//Include the implementations
#include "etl/impl/std/scalar_op.hpp"
#include "etl/impl/blas/scalar_op.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Select the scalar implementation for an expression of type A
 *
 * The local context is not taken into account
 *
 * \tparam A The type of expression
 * \return The implementation to use
 */
template <typename A>
cpp14_constexpr scalar_impl select_default_scalar_impl() {
    if(all_floating<A>::value){
        if (is_cblas_enabled) {
            return scalar_impl::BLAS;
        } else {
            return scalar_impl::STD;
        }
    }

    return scalar_impl::STD;
}

/*!
 * \brief Select the scalar implementation for an expression of type A
 * \tparam A The type of expression
 * \return The implementation to use
 */
template <typename A>
scalar_impl select_scalar_impl() {
    if(local_context().scalar_selector.forced){
        auto forced = local_context().scalar_selector.impl;

        switch (forced) {
            //BLAS cannot always be used
            case scalar_impl::BLAS:
                if(!is_cblas_enabled || !all_floating<A>::value){
                    std::cerr << "Forced selection to BLAS scalar implementation, but not possible for this expression" << std::endl;
                    return select_default_scalar_impl<A>();
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_scalar_impl<A>();
}

struct scalar_add {
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        auto impl = select_scalar_impl<T>();

        if (impl == scalar_impl::BLAS) {
            return etl::impl::blas::scalar_add(lhs, rhs);
        } else {
            return etl::impl::standard::scalar_add(lhs, rhs);
        }
    }
};

struct scalar_sub {
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        auto impl = select_scalar_impl<T>();

        if (impl == scalar_impl::BLAS) {
            return etl::impl::blas::scalar_sub(lhs, rhs);
        } else {
            return etl::impl::standard::scalar_sub(lhs, rhs);
        }
    }
};

struct scalar_mul {
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        auto impl = select_scalar_impl<T>();

        if (impl == scalar_impl::BLAS) {
            return etl::impl::blas::scalar_mul(lhs, rhs);
        } else {
            return etl::impl::standard::scalar_mul(lhs, rhs);
        }
    }
};

struct scalar_div {
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        auto impl = select_scalar_impl<T>();

        if (impl == scalar_impl::BLAS) {
            return etl::impl::blas::scalar_div(lhs, rhs);
        } else {
            return etl::impl::standard::scalar_div(lhs, rhs);
        }
    }
};

struct scalar_mod {
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        etl::impl::standard::scalar_mod(lhs, rhs);
    }
};

} //end of namespace detail

} //end of namespace etl
