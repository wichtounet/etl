//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the "sum" reduction
 */

#pragma once

namespace etl::impl::standard {

/*!
 * \brief Compute the sum of the input in the given expression
 * \param input The input expression
 * \return the sum
 */
template <typename E>
value_t<E> sum(const E& input) {
    using T = value_t<E>;

    T acc(0);

    auto acc_functor = [&acc](T value) { acc += value; };

    auto batch_fun = [](auto& sub) {
        T acc(0);

        for (size_t i = 0; i < etl::size(sub); ++i) {
            acc += sub[i];
        }

        return acc;
    };

    engine_dispatch_1d_acc_slice(input, batch_fun, acc_functor, sum_parallel_threshold);

    return acc;
}

/*!
 * \brief Compute the sum of the absolute values in the given expression
 * \param input The input expression
 * \return the absolute sum
 */
template <typename E>
value_t<E> asum(const E& input) {
    using T = value_t<E>;

    T acc(0);

    auto acc_functor = [&acc](T value) { acc += value; };

    auto batch_fun = [](auto& sub) {
        T acc(0);

        for (size_t i = 0; i < etl::size(sub); ++i) {
            using std::abs;
            acc += abs(sub[i]);
        }

        return acc;
    };

    engine_dispatch_1d_acc_slice(input, batch_fun, acc_functor, sum_parallel_threshold);

    return acc;
}

} //end of namespace etl::impl::standard
