//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Selector for the Mean Squared Error reduction implementations.
 *
 * The functions are responsible for selecting the most efficient
 * implementation for each case, based on what is available.
 */

#pragma once

//Include the implementations
#include "etl/impl/std/mse.hpp"
#include "etl/impl/egblas/mse.hpp"

namespace etl::detail {

/*!
 * \brief Select the MSE implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename O, typename L>
constexpr etl::mse_impl select_mse_impl() {
    if (impl::egblas::has_smse && impl::egblas::has_dmse) {
        return etl::mse_impl::EGBLAS;
    }

    return etl::mse_impl::STD;
}

/*!
 * \brief Select the MSE implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename O, typename L>
constexpr etl::mse_impl select_mse_loss_impl() {
    if (impl::egblas::has_mse_sloss && impl::egblas::has_mse_dloss) {
        return etl::mse_impl::EGBLAS;
    }

    return etl::mse_impl::STD;
}

/*!
 * \brief Select the MSE implementation for an expression of type E
 *
 * \tparam E The type of expression
 * \return The implementation to use
 */
template <typename O, typename L>
constexpr etl::mse_impl select_mse_error_impl() {
    if (impl::egblas::has_mse_serror && impl::egblas::has_mse_derror) {
        return etl::mse_impl::EGBLAS;
    }

    return etl::mse_impl::STD;
}

/*!
 * \brief Sum operation implementation
 */
struct mse_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename O, typename L>
    static std::pair<value_t<O>, value_t<O>> apply(const O& output, const L& labels, value_t<O> alpha, value_t<O> beta) {
        constexpr auto impl = select_mse_impl<O, L>();

        if constexpr (impl == etl::mse_impl::STD) {
            etl::force(output);
            etl::force(labels);

            return impl::standard::mse(output, labels, alpha, beta);
        } else if constexpr (impl == etl::mse_impl::EGBLAS) {
            decltype(auto) output_gpu = smart_forward_gpu(output);
            decltype(auto) labels_gpu = smart_forward_gpu(labels);

            output_gpu.ensure_gpu_up_to_date();
            labels_gpu.ensure_gpu_up_to_date();

            return impl::egblas::mse(etl::size(output), alpha, beta, output_gpu.gpu_memory(), 1, labels_gpu.gpu_memory(), 1);
        } else {
            cpp_unreachable("Invalid selection for MSE");
        }
    }
};

/*!
 * \brief Sum operation implementation
 */
struct mse_loss_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename O, typename L>
    static value_t<O> apply(const O& output, const L& labels, value_t<O> scale) {
        constexpr auto impl = select_mse_loss_impl<O, L>();

        if constexpr (impl == etl::mse_impl::STD) {
            etl::force(output);
            etl::force(labels);

            return impl::standard::mse_loss(etl::size(output), output, labels, scale);
        } else if constexpr (impl == etl::mse_impl::EGBLAS) {
            decltype(auto) output_gpu = smart_forward_gpu(output);
            decltype(auto) labels_gpu = smart_forward_gpu(labels);

            output_gpu.ensure_gpu_up_to_date();
            labels_gpu.ensure_gpu_up_to_date();

            return impl::egblas::mse_loss(etl::size(output), scale, output_gpu.gpu_memory(), 1, labels_gpu.gpu_memory(), 1);
        } else {
            cpp_unreachable("Invalid selection for MSE");
        }
    }
};

/*!
 * \brief Sum operation implementation
 */
struct mse_error_impl {
    /*!
     * \brief Apply the functor to e
     */
    template <typename O, typename L>
    static value_t<O> apply(const O& output, const L& labels, value_t<O> scale) {
        constexpr auto impl = select_mse_error_impl<O, L>();

        if constexpr (impl == etl::mse_impl::STD) {
            etl::force(output);
            etl::force(labels);

            return impl::standard::mse_error(etl::size(output), output, labels, scale);
        } else if constexpr (impl == etl::mse_impl::EGBLAS) {
            decltype(auto) output_gpu = smart_forward_gpu(output);
            decltype(auto) labels_gpu = smart_forward_gpu(labels);

            output_gpu.ensure_gpu_up_to_date();
            labels_gpu.ensure_gpu_up_to_date();

            return impl::egblas::mse_error(etl::size(output), scale, output_gpu.gpu_memory(), 1, labels_gpu.gpu_memory(), 1);
        } else {
            cpp_unreachable("Invalid selection for MSE");
        }
    }
};

} //end of namespace etl::detail
