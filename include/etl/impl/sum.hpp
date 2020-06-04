//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the "sum" reduction implementations.
 *
 * The functions are responsible for selecting the most efficient
 * implementation for each case, based on what is available. The selection of
 * parallel versus serial is also done at this level. The implementation
 * functions should never be used directly, only functions of this header can
 * be used directly.
 *
 * Note: In a perfect world (full constexpr function and variable templates),
 * the selection should be done with a template parameter in a variable
 * template full sspecialization (alias for each real functions).
 */

#pragma once

//Include the implementations
#include "etl/impl/std/sum.hpp"
#include "etl/impl/vec/sum.hpp"
#include "etl/impl/blas/sum.hpp"
#include "etl/impl/cublas/sum.hpp"

namespace etl::detail {

/*!
 * \brief Select the sum implementation for an expression of type E
 *
 * This does not consider the local context
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename E>
constexpr etl::sum_impl select_default_sum_impl(bool no_gpu) {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    if (cublas_enabled && is_gpu_computable<E> && is_floating<E> && !no_gpu) {
        return etl::sum_impl::CUBLAS;
    }

    if (vec_enabled && all_vectorizable<vector_mode, E>) {
        return etl::sum_impl::VEC;
    }

    return etl::sum_impl::STD;
}

#ifdef ETL_MANUAL_SELECT

/*!
 * \brief Select the sum implementation for an expression of type E
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename E>
etl::sum_impl select_sum_impl() {
    if (local_context().sum_selector.forced) {
        auto forced = local_context().sum_selector.impl;

        switch (forced) {
            //VEC cannot always be used
            case sum_impl::VEC:
                if (!vec_enabled || !decay_traits<E>::template vectorizable<vector_mode>) {                                       //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to VEC sum implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_sum_impl<E>(local_context().cpu);                                                       //COVERAGE_EXCLUDE_LINE
                }                                                                                                                 //COVERAGE_EXCLUDE_LINE

                return forced;

            case sum_impl::CUBLAS:
                if (!cublas_enabled || !is_gpu_computable<E> || !is_floating<E> || local_context().cpu) {                                       //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUBLAS sum implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_sum_impl<E>(local_context().cpu);                                                          //COVERAGE_EXCLUDE_LINE
                }                                                                                                                    //COVERAGE_EXCLUDE_LINE

                return forced;

            case sum_impl::BLAS:
                if (!cblas_enabled || !is_dma<E> || !is_floating<E>) {                                                             //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to BLAS sum implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_sum_impl<E>(local_context().cpu);                                                        //COVERAGE_EXCLUDE_LINE
                }                                                                                                                  //COVERAGE_EXCLUDE_LINE

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_sum_impl<E>(local_context().cpu);
}

#else

/*!
 * \brief Select the sum implementation for an expression of type E
 *
 * This does not consider the local context
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename E>
constexpr etl::sum_impl select_sum_impl() {
    return select_default_sum_impl<E>(false);
}

#endif

/*!
 * \brief Sum operation implementation
 */
struct sum_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename E>
    static value_t<E> apply(const E& e) {
        constexpr_select const auto impl = select_sum_impl<E>();

        if
            constexpr_select(impl == etl::sum_impl::VEC) {
                inc_counter("impl:vec");
                return impl::vec::sum(e);
            }
        else if
            constexpr_select(impl == etl::sum_impl::BLAS) {
                inc_counter("impl:blas");
                return impl::blas::sum(e);
            }
        else if
            constexpr_select(impl == etl::sum_impl::CUBLAS) {
                inc_counter("impl:cublas");
                return impl::cublas::sum(e);
            }
        else {
            inc_counter("impl:std");
            return impl::standard::sum(e);
        }
    }
};

/*!
 * \brief Absolute Sum operation implementation
 */
struct asum_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename E>
    static value_t<E> apply(const E& e) {
        constexpr_select const auto impl = select_sum_impl<E>();

        if
            constexpr_select(impl == etl::sum_impl::VEC) {
                inc_counter("impl:vec");
                return impl::vec::asum(e);
            }
        else if
            constexpr_select(impl == etl::sum_impl::BLAS) {
                inc_counter("impl:blas");
                return impl::blas::asum(e);
            }
        else if
            constexpr_select(impl == etl::sum_impl::CUBLAS) {
                inc_counter("impl:cublas");
                return impl::cublas::asum(e);
            }
        else {
            inc_counter("impl:std");
            return impl::standard::asum(e);
        }
    }
};

} //end of namespace etl::detail
