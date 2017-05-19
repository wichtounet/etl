//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// Include all the modules

#include "etl/impl/max_pooling_derivative.hpp"
#include "etl/impl/max_pooling_upsample.hpp"
#include "etl/impl/avg_pooling.hpp"
#include "etl/impl/upsample.hpp"

// Include the implementations

#include "etl/impl/std/max_pooling.hpp"
#include "etl/impl/cudnn/max_pooling.hpp"

namespace etl {

namespace impl {

/*!
 * \brief Select the pool implementation for an expression of type X/Y
 *
 * This does not consider the local context
 *
 * \tparam X The type of expression to pool
 * \tparam Y The type of pooled expression
 *
 * \return The implementation to use
 */
template<typename X, typename Y>
cpp14_constexpr etl::pool_impl select_default_pool_impl() {
    static_assert(all_dma<X, Y>::value, "DMA should be ensured at this point");

    if (cublas_enabled && all_floating<X, Y>::value){
        return etl::pool_impl::CUDNN;
    }

    return etl::pool_impl::STD;
}

/*!
 * \brief Select the pool implementation for an expression of type X/Y
 * \tparam X The type of expression to pool
 * \tparam Y The type of pooled expression
 * \return The implementation to use
 */
template <typename X, typename Y>
etl::pool_impl select_pool_impl() {
    if (local_context().sum_selector.forced) {
        auto forced = local_context().pool_selector.impl;

        switch (forced) {
            // CUDNN cannot always be used
            case pool_impl::CUDNN:
                if (!cudnn_enabled || !all_floating<X, Y>::value) {                                                                  //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN pool implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_pool_impl<X, Y>();                                                                         //COVERAGE_EXCLUDE_LINE
                }                                                                                                                    //COVERAGE_EXCLUDE_LINE

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_pool_impl<X, Y>();
}

/*!
 * \brief Functor for 2D Max Pooling
 */
struct max_pool_2d {
    /*!
     * \brief Pool x into y
     *
     * \param x The expression to pol
     * \param y The expression in which to store the result
     *
     * \tparam C1 The first dimension pooling ratio
     * \tparam C2 The second dimension pooling ratio
     *
     * \tparam S1 The first dimension stride
     * \tparam S2 The second dimension stride
     *
     * \tparam P1 The first dimension padding
     * \tparam P2 The second dimension padding
     */
    template <size_t C1, size_t C2, size_t S1, size_t S2, size_t P1, size_t P2, typename X, typename Y>
    static void apply(const X& x, Y&& y) {
        const auto impl = select_pool_impl<X, Y>();

        if(impl == pool_impl::STD){
            etl::impl::standard::max_pool_2d::apply<C1, C2, S1, S2, P1, P2>(x, y);
        } else if(impl == pool_impl::CUDNN){
            etl::impl::cudnn::max_pool_2d::apply(x, y, C1, C2, S1, S2, P1, P2);
        } else {
            cpp_unreachable("Invalid selection for pooling");
        }
    }

    /*!
     * \brief Pool x into y
     *
     * \param x The expression to pol
     * \param y The expression in which to store the result
     *
     * tparam C1 The first dimension pooling ratio
     * tparam C2 The second dimension pooling ratio
     *
     * tparam S1 The first dimension stride
     * tparam S2 The second dimension stride
     *
     * tparam P1 The first dimension padding
     * tparam P2 The second dimension padding
     */
    template <typename X, typename Y>
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t s1, size_t s2, size_t p1, size_t p2) {
        const auto impl = select_pool_impl<X, Y>();

        if(impl == pool_impl::STD){
            etl::impl::standard::max_pool_2d::apply(x, y, c1, c2, s1, s2, p1, p2);
        } else if(impl == pool_impl::CUDNN){
            etl::impl::cudnn::max_pool_2d::apply(x, y, c1, c2, s1, s2, p1, p2);
        } else {
            cpp_unreachable("Invalid selection for pooling");
        }
    }
};

} // end of namespace impl

} // end of namespace etl
