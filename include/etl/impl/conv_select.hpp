//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains selectors for convolution implementations.
 */

#pragma once

namespace etl {

/*!
 * \brief Enumeration describing the different types of convolution
 */
enum class conv_type {
    VALID,       ///< Valid convolution
    VALID_MULTI, ///< Valid convolution, with multiple kernels
    SAME,        ///< Same convolution
    SAME_MULTI,  ///< Same convolution, with multiple kernels
    FULL,        ///< Full convolution
    FULL_MULTI   ///< Full convolution, with multiple kernels
};

namespace detail {

/*!
 * \brief Test if ETL should run in parallel for the conv of I and K in C
 * \tparam I The input type
 * \tparam K The kernel type
 * \tparam C The conv type
 * \return true to run in paralle, false otherwise
 */
template <typename I, typename K, typename C>
inline bool select_parallel(const I& /*input*/, const K& kernel, C&& conv) {
    if ((is_parallel && !local_context().serial) || (parallel_support && local_context().parallel)) {
        return size(conv) >= conv1_parallel_threshold_conv && size(kernel) >= conv1_parallel_threshold_kernel;
    } else {
        return false;
    }
}

} //end of namespace detail

} //end of namespace etl

#include "conv_normal_select.hpp"
#include "conv_multi_select.hpp"
#include "conv_4d_select.hpp"
