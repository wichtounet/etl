//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

// Include all the modules

#include "etl/impl/max_pooling_derivative.hpp"

// Include the implementations

#include "etl/impl/std/max_pooling.hpp"
#include "etl/impl/std/avg_pooling.hpp"
#include "etl/impl/cudnn/max_pooling.hpp"

namespace etl::impl {

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
template <typename X, typename Y>
constexpr etl::pool_impl select_default_pool_impl(bool no_gpu) {
    static_assert(all_dma<X, Y>, "DMA should be ensured at this point");

    if (cudnn_enabled && all_floating<X, Y> && !no_gpu) {
        return etl::pool_impl::CUDNN;
    }

    return etl::pool_impl::STD;
}

#ifdef ETL_MANUAL_SELECT

/*!
 * \brief Select the pool implementation for an expression of type X/Y
 * \tparam X The type of expression to pool
 * \tparam Y The type of pooled expression
 * \return The implementation to use
 */
template <typename X, typename Y>
etl::pool_impl select_pool_impl() {
    if (local_context().pool_selector.forced) {
        auto forced = local_context().pool_selector.impl;

        switch (forced) {
            // CUDNN cannot always be used
            case pool_impl::CUDNN:
                if (!cudnn_enabled || !all_floating<X, Y> || local_context().cpu) {                                                  //COVERAGE_EXCLUDE_LINE
                    std::cerr << "Forced selection to CUDNN pool implementation, but not possible for this expression" << std::endl; //COVERAGE_EXCLUDE_LINE
                    return select_default_pool_impl<X, Y>(local_context().cpu);                                                      //COVERAGE_EXCLUDE_LINE
                }                                                                                                                    //COVERAGE_EXCLUDE_LINE

                return forced;

            //In other cases, simply use the forced impl
            default:
                return forced;
        }
    }

    return select_default_pool_impl<X, Y>(local_context().cpu);
}

#else

/*!
 * \brief Select the pool implementation for an expression of type X/Y
 *
 * \tparam X The type of expression to pool
 * \tparam Y The type of pooled expression
 *
 * \return The implementation to use
 */
template <typename X, typename Y>
constexpr etl::pool_impl select_pool_impl() {
    return select_default_pool_impl<X, Y>(false);
}

#endif

/*!
 * \brief Functor for 2D Max Pooling
 */
struct max_pool_2d {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cudnn_enabled;

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
        constexpr_select const auto impl = select_pool_impl<X, Y>();

        if
            constexpr_select(impl == pool_impl::STD) {
                etl::impl::standard::max_pool_2d::apply<C1, C2, S1, S2, P1, P2>(smart_forward(x), y);
            }
        else if
            constexpr_select(impl == pool_impl::CUDNN) {
                etl::impl::cudnn::max_pool_2d::apply(smart_forward_gpu(x), y, C1, C2, S1, S2, P1, P2);
            }
        else {
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
        constexpr_select const auto impl = select_pool_impl<X, Y>();

        if
            constexpr_select(impl == pool_impl::STD) {
                etl::impl::standard::max_pool_2d::apply(smart_forward(x), y, c1, c2, s1, s2, p1, p2);
            }
        else if
            constexpr_select(impl == pool_impl::CUDNN) {
                etl::impl::cudnn::max_pool_2d::apply(smart_forward_gpu(x), y, c1, c2, s1, s2, p1, p2);
            }
        else {
            cpp_unreachable("Invalid selection for pooling");
        }
    }
};

/*!
 * \brief Functor for 2D Average Pooling
 */
struct avg_pool_2d {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cudnn_enabled;

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
        constexpr_select const auto impl = select_pool_impl<X, Y>();

        if
            constexpr_select(impl == pool_impl::STD) {
                etl::impl::standard::avg_pool_2d::apply<C1, C2, S1, S2, P1, P2>(smart_forward(x), y);
            }
        else if
            constexpr_select(impl == pool_impl::CUDNN) {
                etl::impl::cudnn::avg_pool_2d::apply(smart_forward_gpu(x), y, C1, C2, S1, S2, P1, P2);
            }
        else {
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
        constexpr_select const auto impl = select_pool_impl<X, Y>();

        if
            constexpr_select(impl == pool_impl::STD) {
                etl::impl::standard::avg_pool_2d::apply(smart_forward(x), y, c1, c2, s1, s2, p1, p2);
            }
        else if
            constexpr_select(impl == pool_impl::CUDNN) {
                etl::impl::cudnn::avg_pool_2d::apply(smart_forward_gpu(x), y, c1, c2, s1, s2, p1, p2);
            }
        else {
            cpp_unreachable("Invalid selection for pooling");
        }
    }
};

/*!
 * \brief Functor for 3D Max Pooling
 */
struct max_pool_3d {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cudnn_enabled;

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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename X, typename Y>
    static void apply(const X& x, Y&& y) {
        constexpr_select const auto impl = select_pool_impl<X, Y>();

        if
            constexpr_select(impl == pool_impl::STD) {
                etl::impl::standard::max_pool_3d::apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(smart_forward(x), y);
            }
        else if
            constexpr_select(impl == pool_impl::CUDNN) {
                etl::impl::cudnn::max_pool_3d::apply(smart_forward_gpu(x), y, C1, C2, C3, S1, S2, S3, P1, P2, P3);
            }
        else {
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
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        constexpr_select const auto impl = select_pool_impl<X, Y>();

        if
            constexpr_select(impl == pool_impl::STD) {
                etl::impl::standard::max_pool_3d::apply(smart_forward(x), y, c1, c2, c3, s1, s2, s3, p1, p2, p3);
            }
        else if
            constexpr_select(impl == pool_impl::CUDNN) {
                etl::impl::cudnn::max_pool_3d::apply(smart_forward_gpu(x), y, c1, c2, c3, s1, s2, s3, p1, p2, p3);
            }
        else {
            cpp_unreachable("Invalid selection for pooling");
        }
    }
};

/*!
 * \brief Functor for 3D Average Pooling
 */
struct avg_pool_3d {
    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    template <typename A>
    static constexpr bool gpu_computable = cudnn_enabled;

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
    template <size_t C1, size_t C2, size_t C3, size_t S1, size_t S2, size_t S3, size_t P1, size_t P2, size_t P3, typename X, typename Y>
    static void apply(const X& x, Y&& y) {
        constexpr_select const auto impl = select_pool_impl<X, Y>();

        if
            constexpr_select(impl == pool_impl::STD) {
                etl::impl::standard::avg_pool_3d::apply<C1, C2, C3, S1, S2, S3, P1, P2, P3>(smart_forward(x), y);
            }
        else if
            constexpr_select(impl == pool_impl::CUDNN) {
                etl::impl::cudnn::avg_pool_3d::apply(smart_forward_gpu(x), y, C1, C2, C3, S1, S2, S3, P1, P2, P3);
            }
        else {
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
    static void apply(const X& x, Y&& y, size_t c1, size_t c2, size_t c3, size_t s1, size_t s2, size_t s3, size_t p1, size_t p2, size_t p3) {
        const auto impl = select_pool_impl<X, Y>();

        if
            constexpr_select(impl == pool_impl::STD) {
                etl::impl::standard::avg_pool_3d::apply(smart_forward(x), y, c1, c2, c3, s1, s2, s3, p1, p2, p3);
            }
        else if
            constexpr_select(impl == pool_impl::CUDNN) {
                etl::impl::cudnn::avg_pool_3d::apply(smart_forward_gpu(x), y, c1, c2, c3, s1, s2, s3, p1, p2, p3);
            }
        else {
            cpp_unreachable("Invalid selection for pooling");
        }
    }
};

} //end of namespace etl::impl
