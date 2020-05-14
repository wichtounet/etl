//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Unified vectorized implementation of the "sum" reduction
 */

#pragma once

namespace etl::impl::vec {

/*!
 * \brief Vectorized sum computation
 * \param lhs The expression to compute the sum from
 * \tparam V The vectorization type
 * \return The sum of the given range
 */
template <typename V, typename L>
value_t<L> sum_impl(const L& lhs) {
    //Note: Padding cannot be taken into account we don't start at zero

    using vec_type = V;
    using T        = value_t<L>;

    const size_t n = etl::size(lhs);

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    safe_ensure_cpu_up_to_date(lhs);

    size_t i = 0;

    auto r1 = vec_type::template zero<T>();
    auto r2 = vec_type::template zero<T>();
    auto r3 = vec_type::template zero<T>();
    auto r4 = vec_type::template zero<T>();

    for (; i + (vec_size * 4) - 1 < n; i += 4 * vec_size) {
        r1 = vec_type::add(lhs.template load<vec_type>(i + 0 * vec_size), r1);
        r2 = vec_type::add(lhs.template load<vec_type>(i + 1 * vec_size), r2);
        r3 = vec_type::add(lhs.template load<vec_type>(i + 2 * vec_size), r3);
        r4 = vec_type::add(lhs.template load<vec_type>(i + 3 * vec_size), r4);
    }

    for (; i + (vec_size * 2) - 1 < n; i += 2 * vec_size) {
        r1 = vec_type::add(lhs.template load<vec_type>(i + 0 * vec_size), r1);
        r2 = vec_type::add(lhs.template load<vec_type>(i + 1 * vec_size), r2);
    }

    for (; i + vec_size - 1 < n; i += vec_size) {
        r1 = vec_type::add(lhs.template load<vec_type>(i + 0 * vec_size), r1);
    }

    auto p1 = vec_type::hadd(r1) + vec_type::hadd(r2) + vec_type::hadd(r3) + vec_type::hadd(r4);
    auto p2 = T();

    for (; i + 1 < n; i += 2) {
        p1 += lhs[i];
        p2 += lhs[i + 1];
    }

    if (i < n) {
        p1 += lhs[i];
    }

    return p1 + p2;
}

/*!
 * \brief Vectorized absolute sum computation
 * \param lhs The expression to compute the sum from
 * \tparam V The vectorization type
 * \return The absolute sum of the given range
 */
template <typename V, typename L>
value_t<L> asum_impl(const L& lhs) {
    //Note: Padding cannot be taken into account we don't start at zero

    using vec_type = V;
    using T        = value_t<L>;
    using std::abs;

    const size_t n = etl::size(lhs);

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    safe_ensure_cpu_up_to_date(lhs);

    size_t i = 0;

    auto r1 = vec_type::template zero<T>();
    auto r2 = vec_type::template zero<T>();
    auto r3 = vec_type::template zero<T>();
    auto r4 = vec_type::template zero<T>();

    for (; i + (vec_size * 4) - 1 < n; i += 4 * vec_size) {
        auto v1 = lhs.template load<vec_type>(i + 0 * vec_size);
        auto v2 = lhs.template load<vec_type>(i + 1 * vec_size);
        auto v3 = lhs.template load<vec_type>(i + 2 * vec_size);
        auto v4 = lhs.template load<vec_type>(i + 3 * vec_size);

        auto x1 = vec_type::max(v1, vec_type::sub(vec_type::template zero<T>(), v1));
        auto x2 = vec_type::max(v2, vec_type::sub(vec_type::template zero<T>(), v2));
        auto x3 = vec_type::max(v3, vec_type::sub(vec_type::template zero<T>(), v3));
        auto x4 = vec_type::max(v4, vec_type::sub(vec_type::template zero<T>(), v4));

        r1 = vec_type::add(x1, r1);
        r2 = vec_type::add(x2, r2);
        r3 = vec_type::add(x3, r3);
        r4 = vec_type::add(x4, r4);
    }

    for (; i + (vec_size * 2) - 1 < n; i += 2 * vec_size) {
        auto v1 = lhs.template load<vec_type>(i + 0 * vec_size);
        auto v2 = lhs.template load<vec_type>(i + 1 * vec_size);

        auto x1 = vec_type::max(v1, vec_type::sub(vec_type::template zero<T>(), v1));
        auto x2 = vec_type::max(v2, vec_type::sub(vec_type::template zero<T>(), v2));

        r1 = vec_type::add(x1, r1);
        r2 = vec_type::add(x2, r2);
    }

    for (; i + vec_size - 1 < n; i += vec_size) {
        auto v1 = lhs.template load<vec_type>(i + 0 * vec_size);
        auto x1 = vec_type::max(v1, vec_type::sub(vec_type::template zero<T>(), v1));
        r1      = vec_type::add(x1, r1);
    }

    auto p1 = vec_type::hadd(r1) + vec_type::hadd(r2) + vec_type::hadd(r3) + vec_type::hadd(r4);
    auto p2 = T();

    for (; i + 1 < n; i += 2) {
        p1 += abs(lhs[i]);
        p2 += abs(lhs[i + 1]);
    }

    if (i < n) {
        p1 += abs(lhs[i]);
    }

    return p1 + p2;
}

/*!
 * \brief Compute the sum of lhs
 * \param lhs The lhs expression
 * \return the sum of the elements of lhs
 */
template <typename L>
value_t<L> sum([[maybe_unused]] const L& lhs) {
    if constexpr (vec_enabled && all_vectorizable<vector_mode, L>) {
        using T = value_t<L>;

        T acc(0);

        auto acc_functor = [&acc](T value) { acc += value; };

        auto batch_fun = [](auto& sub) {
            // The default vectorization scheme should be sufficient
            return sum_impl<default_vec>(sub);
        };

        if (etl::size(lhs) < sum_parallel_threshold) {
            return sum_impl<default_vec>(lhs);
        } else {
            engine_dispatch_1d_acc_slice(lhs, batch_fun, acc_functor, vec_sum_parallel_threshold);
        }

        return acc;
    } else {
        cpp_unreachable("vec::sum called with invalid parameters");
    }
}

/*!
 * \brief Compute the absolute sum of lhs
 * \param lhs The lhs expression
 * \return the absolute sum of the elements of lhs
 */
template <typename L>
value_t<L> asum([[maybe_unused]] const L& lhs) {
    if constexpr (vec_enabled && all_vectorizable<vector_mode, L>) {
        using T = value_t<L>;

        T acc(0);

        auto acc_functor = [&acc](T value) { acc += value; };

        auto batch_fun = [](auto& sub) {
            // The default vectorization scheme should be sufficient
            return asum_impl<default_vec>(sub);
        };

        engine_dispatch_1d_acc_slice(lhs, batch_fun, acc_functor, vec_sum_parallel_threshold);

        return acc;
    } else {
        cpp_unreachable("vec::sum called with invalid parameters");
    }
}

} //end of namespace etl::impl::vec
