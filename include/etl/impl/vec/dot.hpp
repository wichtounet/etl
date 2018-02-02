//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Unified vectorized implementation of the "dot" reduction
 */

#pragma once

namespace etl::impl::vec {

/*!
 * \brief Compute the dot product of a and b using vectorized code
 * \param lhs The lhs expression
 * \param rhs The rhs expression
 * \return the dot product
 */
template <typename V, typename L, typename R>
value_t<L> dot_impl(const L& lhs, const R& rhs) {
    using vec_type = V;
    using T        = value_t<L>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    auto n = etl::size(lhs);

    static constexpr bool remainder = !padding || !all_padded<L, R>;
    const size_t last               = remainder ? (n & size_t(-vec_size)) : n;

    size_t i = 0;

    auto r1 = vec_type::template zero<T>();
    auto r2 = vec_type::template zero<T>();
    auto r3 = vec_type::template zero<T>();
    auto r4 = vec_type::template zero<T>();
    auto r5 = vec_type::template zero<T>();
    auto r6 = vec_type::template zero<T>();
    auto r7 = vec_type::template zero<T>();
    auto r8 = vec_type::template zero<T>();

    if (n <= 4 * cache_size / sizeof(T)) {
        for (; i + (vec_size * 7) < last; i += 8 * vec_size) {
            auto a1 = lhs.template load<vec_type>(i + 0 * vec_size);
            auto a2 = lhs.template load<vec_type>(i + 1 * vec_size);
            auto a3 = lhs.template load<vec_type>(i + 2 * vec_size);
            auto a4 = lhs.template load<vec_type>(i + 3 * vec_size);
            auto a5 = lhs.template load<vec_type>(i + 4 * vec_size);
            auto a6 = lhs.template load<vec_type>(i + 5 * vec_size);
            auto a7 = lhs.template load<vec_type>(i + 6 * vec_size);
            auto a8 = lhs.template load<vec_type>(i + 7 * vec_size);

            auto b1 = rhs.template load<vec_type>(i + 0 * vec_size);
            auto b2 = rhs.template load<vec_type>(i + 1 * vec_size);
            auto b3 = rhs.template load<vec_type>(i + 2 * vec_size);
            auto b4 = rhs.template load<vec_type>(i + 3 * vec_size);
            auto b5 = rhs.template load<vec_type>(i + 4 * vec_size);
            auto b6 = rhs.template load<vec_type>(i + 5 * vec_size);
            auto b7 = rhs.template load<vec_type>(i + 6 * vec_size);
            auto b8 = rhs.template load<vec_type>(i + 7 * vec_size);

            r1 = vec_type::fmadd(a1, b1, r1);
            r2 = vec_type::fmadd(a2, b2, r2);
            r3 = vec_type::fmadd(a3, b3, r3);
            r4 = vec_type::fmadd(a4, b4, r4);
            r5 = vec_type::fmadd(a5, b5, r5);
            r6 = vec_type::fmadd(a6, b6, r6);
            r7 = vec_type::fmadd(a7, b7, r7);
            r8 = vec_type::fmadd(a8, b8, r8);
        }

        for (; i + (vec_size * 3) < last; i += 4 * vec_size) {
            auto a1 = lhs.template load<vec_type>(i + 0 * vec_size);
            auto a2 = lhs.template load<vec_type>(i + 1 * vec_size);
            auto a3 = lhs.template load<vec_type>(i + 2 * vec_size);
            auto a4 = lhs.template load<vec_type>(i + 3 * vec_size);

            auto b1 = rhs.template load<vec_type>(i + 0 * vec_size);
            auto b2 = rhs.template load<vec_type>(i + 1 * vec_size);
            auto b3 = rhs.template load<vec_type>(i + 2 * vec_size);
            auto b4 = rhs.template load<vec_type>(i + 3 * vec_size);

            r1 = vec_type::fmadd(a1, b1, r1);
            r2 = vec_type::fmadd(a2, b2, r2);
            r3 = vec_type::fmadd(a3, b3, r3);
            r4 = vec_type::fmadd(a4, b4, r4);
        }
    }

    for (; i + (vec_size * 1) < last; i += 2 * vec_size) {
        auto a1 = lhs.template load<vec_type>(i + 0 * vec_size);
        auto a2 = lhs.template load<vec_type>(i + 1 * vec_size);

        auto b1 = rhs.template load<vec_type>(i + 0 * vec_size);
        auto b2 = rhs.template load<vec_type>(i + 1 * vec_size);

        r1 = vec_type::fmadd(a1, b1, r1);
        r2 = vec_type::fmadd(a2, b2, r2);
    }

    for (; i < last; i += vec_size) {
        auto a1 = lhs.template load<vec_type>(i);
        auto b1 = rhs.template load<vec_type>(i);

        r1 = vec_type::fmadd(a1, b1, r1);
    }

    auto rsum = vec_type::add(vec_type::add(vec_type::add(r1, r2), vec_type::add(r3, r4)), vec_type::add(vec_type::add(r5, r6), vec_type::add(r7, r8)));

    auto p1 = vec_type::hadd(rsum);
    auto p2 = T();

    for (; remainder && i + 1 < n; i += 2) {
        p1 += lhs[i] * rhs[i];
        p2 += lhs[i + 1] * rhs[i + 1];
    }

    if (remainder && i < n) {
        p1 += lhs[i] * rhs[i];
    }

    return p1 + p2;
}

/*!
 * \brief Compute the dot product of a and b
 * \param lhs The lhs expression
 * \param rhs The rhs expression
 * \return the dot product
 */
template <typename L, typename R>
value_t<L> dot(const L& lhs, const R& rhs) {
    lhs.ensure_cpu_up_to_date();
    rhs.ensure_cpu_up_to_date();

    // The default vectorization scheme should be sufficient
    return dot_impl<default_vec>(lhs, rhs);
}

} //end of namespace etl::impl::vec
