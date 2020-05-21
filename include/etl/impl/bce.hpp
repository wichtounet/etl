//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the Binary Cross Entropy reduction implementations.
 *
 * The functions are responsible for selecting the most efficient
 * implementation for each case, based on what is available.
 */

#pragma once

//Include the implementations
#include "etl/impl/std/bce.hpp"
#include "etl/impl/egblas/bce.hpp"

namespace etl::detail {

/*!
 * \brief Select the BCE implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename O, typename L>
constexpr etl::bce_impl select_bce_impl() {
    if (impl::egblas::has_sbce && impl::egblas::has_dbce) {
        return etl::bce_impl::EGBLAS;
    }

    return etl::bce_impl::STD;
}

/*!
 * \brief Select the BCE implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename O, typename L>
constexpr etl::bce_impl select_bce_loss_impl() {
    if (impl::egblas::has_bce_sloss && impl::egblas::has_bce_dloss) {
        return etl::bce_impl::EGBLAS;
    }

    return etl::bce_impl::STD;
}

/*!
 * \brief Select the BCE implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename O, typename L>
constexpr etl::bce_impl select_bce_error_impl() {
    if (impl::egblas::has_bce_serror && impl::egblas::has_bce_derror) {
        return etl::bce_impl::EGBLAS;
    }

    return etl::bce_impl::STD;
}

/*!
 * \brief Sum operation implementation
 */
struct bce_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename O, typename L>
    static std::pair<value_t<O>, value_t<O>> apply(const O& output, const L& labels, value_t<O> alpha, value_t<O> beta) {
        constexpr auto impl = select_bce_impl<O, L>();

        if constexpr (impl == etl::bce_impl::STD) {
            etl::force(output);
            etl::force(labels);

            return impl::standard::bce(output, labels, alpha, beta);
        } else if constexpr (impl == etl::bce_impl::EGBLAS) {
            decltype(auto) output_gpu = smart_forward_gpu(output);
            decltype(auto) labels_gpu = smart_forward_gpu(labels);

            output_gpu.ensure_gpu_up_to_date();
            labels_gpu.ensure_gpu_up_to_date();

            return impl::egblas::bce(etl::size(output), alpha, beta, output_gpu.gpu_memory(), 1, labels_gpu.gpu_memory(), 1);
        } else {
            cpp_unreachable("Invalid selection for BCE");
        }
    }
};

/*!
 * \brief Sum operation implementation
 */
struct bce_loss_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename O, typename L>
    static value_t<O> apply(const O& output, const L& labels, value_t<O> scale) {
        constexpr auto impl = select_bce_loss_impl<O, L>();

        if constexpr (impl == etl::bce_impl::STD) {
            etl::force(output);
            etl::force(labels);

            return impl::standard::bce_loss(output, labels, scale);
        } else if constexpr (impl == etl::bce_impl::EGBLAS) {
            decltype(auto) output_gpu = smart_forward_gpu(output);
            decltype(auto) labels_gpu = smart_forward_gpu(labels);

            output_gpu.ensure_gpu_up_to_date();
            labels_gpu.ensure_gpu_up_to_date();

            return impl::egblas::bce_loss(etl::size(output), scale, output_gpu.gpu_memory(), 1, labels_gpu.gpu_memory(), 1);
        } else {
            cpp_unreachable("Invalid selection for BCE");
        }
    }
};

/*!
 * \brief Sum operation implementation
 */
struct bce_error_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename O, typename L>
    static value_t<O> apply(const O& output, const L& labels, value_t<O> scale) {
        constexpr auto impl = select_bce_error_impl<O, L>();

        if constexpr (impl == etl::bce_impl::STD) {
            etl::force(output);
            etl::force(labels);

            return impl::standard::bce_error(output, labels, scale);
        } else if constexpr (impl == etl::bce_impl::EGBLAS) {
            decltype(auto) output_gpu = smart_forward_gpu(output);
            decltype(auto) labels_gpu = smart_forward_gpu(labels);

            output_gpu.ensure_gpu_up_to_date();
            labels_gpu.ensure_gpu_up_to_date();

            return impl::egblas::bce_error(etl::size(output), scale, output_gpu.gpu_memory(), 1, labels_gpu.gpu_memory(), 1);
        } else {
            cpp_unreachable("Invalid selection for BCE");
        }
    }
};

} //end of namespace etl::detail
