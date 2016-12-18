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
#include "etl/impl/cublas/transpose.hpp"

#if __INTEL_MKL__ == 11 && __INTEL_MKL_MINOR__ == 2
#define SLOW_MKL
#endif

namespace etl {

namespace detail {

//TODO We should take into account parallel blas when selecting MKL transpose

/*!
 * \brief Select the transpose implementation for an expression of type A and C
 * \tparam A The type of rhs expression
 * \tparam C The type of lhs expression
 * \return The implementation to use
 */
template <typename A, typename C>
transpose_impl select_transpose_impl_smart() {
    if (local_context().transpose_selector.forced) {
        auto forced = local_context().transpose_selector.impl;

        switch (forced) {
            //MKL cannot always be used
            case transpose_impl::MKL:
                if (!mkl_enabled || !all_dma<A, C>::value || !all_floating<A, C>::value) {
                    std::cerr << "Forced selection to MKL transpose implementation, but not possible for this expression" << std::endl;
                    return transpose_impl::SELECT;
                }

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return transpose_impl::SELECT;
}

/*!
 * \brief Functor for inplace square matrix transposition
 */
struct inplace_square_transpose {
    /*!
     * \brief Tranpose c inplace
     * \param c The target matrix
     */
    template <typename C>
    static void apply(C&& c) {
        // Condition to use MKL
        static constexpr bool mkl_possible = mkl_enabled && all_dma<C>::value && all_floating<C>::value;

        const auto impl = select_transpose_impl_smart<C, C>();

        if(cpp_likely(impl == transpose_impl::SELECT)){
            // MKL is always faster than STD on inplace square transpositions
            if(mkl_possible){
                etl::impl::blas::inplace_square_transpose(c);
            } else {
                etl::impl::standard::inplace_square_transpose(c);
            }
        } else if (impl == transpose_impl::MKL) {
            etl::impl::blas::inplace_square_transpose(c);
        } else if (impl == transpose_impl::CUBLAS) {
            etl::impl::cublas::inplace_square_transpose(c);
        } else if(impl == transpose_impl::STD){
            etl::impl::standard::inplace_square_transpose(c);
        } else {
            cpp_unreachable("Invalid transpose_impl selection");
        }
    }
};

/*!
 * \brief Functor for inplace rectangular matrix transposition
 */
struct inplace_rectangular_transpose {
    /*!
     * \brief Tranpose c inplace
     * \param c The target matrix
     */
    template <typename C>
    static void apply(C&& c) {
        const auto impl = select_transpose_impl_smart<C, C>();

        if(cpp_likely(impl == transpose_impl::SELECT)){
#ifdef SLOW_MKL
            // STD is always faster than MKL for inplace rectangular transpositions
            etl::impl::standard::inplace_rectangular_transpose(c);
#else
            // Condition to use MKL
            static constexpr bool mkl_possible = mkl_enabled && all_dma<C>::value && all_floating<C>::value;

            if(mkl_possible){
                etl::impl::blas::inplace_rectangular_transpose(c);
            } else {
                etl::impl::standard::inplace_rectangular_transpose(c);
            }
#endif
        } else if (impl == transpose_impl::MKL) {
            etl::impl::blas::inplace_rectangular_transpose(c);
        } else if (impl == transpose_impl::CUBLAS) {
            etl::impl::cublas::inplace_rectangular_transpose(c);
        } else if(impl == transpose_impl::STD){
            etl::impl::standard::inplace_rectangular_transpose(c);
        } else {
            cpp_unreachable("Invalid transpose_impl selection");
        }
    }
};

/*!
 * \brief Functor for general matrix transposition
 */
struct transpose {
    /*!
     * \brief Tranpose a and store the results in c
     * \param a The source matrix
     * \param c The target matrix
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        const auto impl = select_transpose_impl_smart<A, C>();

        if(cpp_likely(impl == transpose_impl::SELECT)){
#ifdef SLOW_MKL
            // STD is always faster than MKL for out-of-place transpose
            etl::impl::standard::transpose(a, c);
#else
            // Condition to use MKL
            static constexpr bool mkl_possible = mkl_enabled && all_dma<C>::value && all_floating<C>::value;

            if(mkl_possible){
                etl::impl::blas::transpose(a, c);
            } else {
                etl::impl::standard::transpose(a, c);
            }
#endif
        } else if (impl == transpose_impl::MKL) {
            etl::impl::blas::transpose(a, c);
        } else if (impl == transpose_impl::CUBLAS) {
            etl::impl::cublas::transpose(a, c);
        } else if(impl == transpose_impl::STD){
            etl::impl::standard::transpose(a, c);
        } else {
            cpp_unreachable("Invalid transpose_impl selection");
        }
    }
};

} //end of namespace detail

} //end of namespace etl
