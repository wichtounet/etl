//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementations of vector/matrix scalar operations.
 */

#pragma once

//Include the implementations
#include "etl/impl/std/scalar_op.hpp"
#include "etl/impl/blas/scalar_op.hpp"
#include "etl/impl/cublas/scalar_op.hpp"

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
template <typename A, bool Simple>
constexpr scalar_impl select_default_scalar_impl(bool gpu) {
    if (is_floating<A>) {
        if (cublas_enabled && gpu){
            return scalar_impl::CUBLAS;
        } else if (cblas_enabled && !Simple) {
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
template <typename A, bool Simple>
scalar_impl select_scalar_impl(bool gpu) {
    if (local_context().scalar_selector.forced) {
        auto forced = local_context().scalar_selector.impl;

        switch (forced) {
            //BLAS cannot always be used
            case scalar_impl::BLAS:
                if (!cblas_enabled || !is_floating<A>) {
                    std::cerr << "Forced selection to BLAS scalar implementation, but not possible for this expression" << std::endl;
                    return select_default_scalar_impl<A, Simple>(gpu);
                }

                return forced;

            //CUBLAS cannot always be used
            case scalar_impl::CUBLAS:
                if (!cublas_enabled || !is_floating<A>) {
                    std::cerr << "Forced selection to CUBLAS scalar implementation, but not possible for this expression" << std::endl;
                    return select_default_scalar_impl<A, Simple>(gpu);
                }

                return forced;


            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_scalar_impl<A, Simple>(gpu);
}

/*!
 * \brief Functor to add a scalar to each element of an expression
 */
struct scalar_add {
    /*!
     * \brief Apply the operation on the expression lhs and the scalar rhs
     * \param lhs The expression
     * \param rhs The scalar
     */
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        auto impl = select_scalar_impl<T, true>(safe_is_gpu_up_to_date(lhs));

        if (impl == scalar_impl::BLAS) {
            return etl::impl::blas::scalar_add(lhs, rhs);
        } else if (impl == scalar_impl::CUBLAS) {
            return etl::impl::cublas::scalar_add(lhs, rhs);
        } else {
            return etl::impl::standard::scalar_add(lhs, rhs);
        }
    }
};

/*!
 * \brief Functor to remove a scalar from each element of an expression
 */
struct scalar_sub {
    /*!
     * \brief Apply the operation on the expression lhs and the scalar rhs
     * \param lhs The expression
     * \param rhs The scalar
     */
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        auto impl = select_scalar_impl<T, true>(safe_is_gpu_up_to_date(lhs));

        if (impl == scalar_impl::BLAS) {
            return etl::impl::blas::scalar_sub(lhs, rhs);
        } else if (impl == scalar_impl::CUBLAS) {
            return etl::impl::cublas::scalar_sub(lhs, rhs);
        } else {
            return etl::impl::standard::scalar_sub(lhs, rhs);
        }
    }
};

/*!
 * \brief Functor to multiply each element of an expression by a scalar
 */
struct scalar_mul {
    /*!
     * \brief Apply the operation on the expression lhs and the scalar rhs
     * \param lhs The expression
     * \param rhs The scalar
     */
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        auto impl = select_scalar_impl<T, false>(safe_is_gpu_up_to_date(lhs));

        if (impl == scalar_impl::BLAS) {
            return etl::impl::blas::scalar_mul(lhs, rhs);
        } else if (impl == scalar_impl::CUBLAS) {
            return etl::impl::cublas::scalar_mul(lhs, rhs);
        } else {
            return etl::impl::standard::scalar_mul(lhs, rhs);
        }
    }
};

/*!
 * \brief Functor to divide each element of an expression by a scalar
 */
struct scalar_div {
    /*!
     * \brief Apply the operation on the expression lhs and the scalar rhs
     * \param lhs The expression
     * \param rhs The scalar
     */
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        auto impl = select_scalar_impl<T, false>(safe_is_gpu_up_to_date(lhs));

        if (impl == scalar_impl::BLAS) {
            return etl::impl::blas::scalar_div(lhs, rhs);
        } else if (impl == scalar_impl::CUBLAS) {
            return etl::impl::cublas::scalar_div(lhs, rhs);
        } else {
            return etl::impl::standard::scalar_div(lhs, rhs);
        }
    }
};

/*!
 * \brief Functor to modulo each element of an expression by a scalar
 */
struct scalar_mod {
    /*!
     * \brief Apply the operation on the expression lhs and the scalar rhs
     * \param lhs The expression
     * \param rhs The scalar
     */
    template <typename T>
    static void apply(T&& lhs, value_t<T> rhs) {
        etl::impl::standard::scalar_mod(lhs, rhs);
    }
};

} //end of namespace detail

} //end of namespace etl
