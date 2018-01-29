//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the Categorical Cross Entropy reduction implementations.
 *
 * The functions are responsible for selecting the most efficient
 * implementation for each case, based on what is available.
 */

#pragma once

//Include the implementations
#include "etl/impl/std/cce.hpp"
#include "etl/impl/egblas/cce.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Select the CCE implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename O, typename L>
constexpr etl::cce_impl select_cce_loss_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    if(impl::egblas::has_cce_sloss && impl::egblas::has_cce_dloss){
        return etl::cce_impl::EGBLAS;
    }

    return etl::cce_impl::STD;
}

/*!
 * \brief Select the CCE implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename O, typename L>
constexpr etl::cce_impl select_cce_error_impl() {
    //Note: since the constexpr values will be known at compile time, the
    //conditions will be a lot simplified

    if(impl::egblas::has_cce_serror && impl::egblas::has_cce_derror){
        return etl::cce_impl::EGBLAS;
    }

    return etl::cce_impl::STD;
}

/*!
 * \brief Sum operation implementation
 */
struct cce_loss_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename O, typename L>
    static value_t<O> apply(const O& output, const L& labels, value_t<O> scale) {
        constexpr auto impl = select_cce_loss_impl<O, L>();

        if constexpr (impl == etl::cce_impl::STD) {
            etl::force(output);
            etl::force(labels);

            return impl::standard::cce_loss(output, labels, scale);
        } else if constexpr (impl == etl::cce_impl::EGBLAS) {
            decltype(auto) output_gpu = smart_forward_gpu(output);
            decltype(auto) labels_gpu = smart_forward_gpu(labels);

            output_gpu.ensure_gpu_up_to_date();
            labels_gpu.ensure_gpu_up_to_date();

            return impl::egblas::cce_loss(etl::size(output), scale, output_gpu.gpu_memory(), 1, labels_gpu.gpu_memory(), 1);
        } else {
            cpp_unreachable("Invalid selection for CCE");
        }
    }
};

/*!
 * \brief Sum operation implementation
 */
struct cce_error_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename O, typename L>
    static value_t<O> apply(const O& output, const L& labels, value_t<O> scale) {
        constexpr auto impl = select_cce_error_impl<O, L>();

        if constexpr (impl == etl::cce_impl::STD) {
            etl::force(output);
            etl::force(labels);

            return impl::standard::cce_error(output, labels, scale);
        } else if constexpr (impl == etl::cce_impl::EGBLAS) {
            decltype(auto) output_gpu = smart_forward_gpu(output);
            decltype(auto) labels_gpu = smart_forward_gpu(labels);

            output_gpu.ensure_gpu_up_to_date();
            labels_gpu.ensure_gpu_up_to_date();

            return impl::egblas::cce_error(etl::dim<0>(output), etl::dim<1>(output), scale, output_gpu.gpu_memory(), labels_gpu.gpu_memory());
        } else {
            cpp_unreachable("Invalid selection for CCE");
        }
    }
};

} //end of namespace detail

} //end of namespace etl
