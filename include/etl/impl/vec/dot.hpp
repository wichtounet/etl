//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Unified vectorized implementation of the "dot" reduction
 */

#pragma once

namespace etl {

namespace impl {

namespace vec {

template <typename V, typename L, typename R>
value_t<L> selected_dot(const L& lhs, const R& rhs) {
    using vec_type = V;
    using T        = value_t<L>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    auto n = etl::size(lhs);

    size_t i = 0;

    auto r1 = vec_type::template zero<T>();
    auto r2 = vec_type::template zero<T>();
    auto r3 = vec_type::template zero<T>();
    auto r4 = vec_type::template zero<T>();
    auto r5 = vec_type::template zero<T>();
    auto r6 = vec_type::template zero<T>();
    auto r7 = vec_type::template zero<T>();
    auto r8 = vec_type::template zero<T>();

    if (n < 1000000) {
        for (; i + (vec_size * 8) - 1 < n; i += 8 * vec_size) {
            auto a1 = lhs.load(i + 0 * vec_size);
            auto a2 = lhs.load(i + 1 * vec_size);
            auto a3 = lhs.load(i + 2 * vec_size);
            auto a4 = lhs.load(i + 3 * vec_size);
            auto a5 = lhs.load(i + 4 * vec_size);
            auto a6 = lhs.load(i + 5 * vec_size);
            auto a7 = lhs.load(i + 6 * vec_size);
            auto a8 = lhs.load(i + 7 * vec_size);

            auto b1 = rhs.load(i + 0 * vec_size);
            auto b2 = rhs.load(i + 1 * vec_size);
            auto b3 = rhs.load(i + 2 * vec_size);
            auto b4 = rhs.load(i + 3 * vec_size);
            auto b5 = rhs.load(i + 4 * vec_size);
            auto b6 = rhs.load(i + 5 * vec_size);
            auto b7 = rhs.load(i + 6 * vec_size);
            auto b8 = rhs.load(i + 7 * vec_size);

            r1 = vec_type::template fmadd<false>(a1, b1, r1);
            r2 = vec_type::template fmadd<false>(a2, b2, r2);
            r3 = vec_type::template fmadd<false>(a3, b3, r3);
            r4 = vec_type::template fmadd<false>(a4, b4, r4);
            r5 = vec_type::template fmadd<false>(a5, b5, r5);
            r6 = vec_type::template fmadd<false>(a6, b6, r6);
            r7 = vec_type::template fmadd<false>(a7, b7, r7);
            r8 = vec_type::template fmadd<false>(a8, b8, r8);
        }
    }

    for (; i + (vec_size * 4) - 1 < n; i += 4 * vec_size) {
        auto a1 = lhs.load(i + 0 * vec_size);
        auto a2 = lhs.load(i + 1 * vec_size);
        auto a3 = lhs.load(i + 2 * vec_size);
        auto a4 = lhs.load(i + 3 * vec_size);

        auto b1 = rhs.load(i + 0 * vec_size);
        auto b2 = rhs.load(i + 1 * vec_size);
        auto b3 = rhs.load(i + 2 * vec_size);
        auto b4 = rhs.load(i + 3 * vec_size);

        r1 = vec_type::template fmadd<false>(a1, b1, r1);
        r2 = vec_type::template fmadd<false>(a2, b2, r2);
        r3 = vec_type::template fmadd<false>(a3, b3, r3);
        r4 = vec_type::template fmadd<false>(a4, b4, r4);
    }

    for(; i + (vec_size * 2) - 1 < n; i += 2 * vec_size){
        auto a1 = lhs.load(i + 0 * vec_size);
        auto a2 = lhs.load(i + 1 * vec_size);

        auto b1 = rhs.load(i + 0 * vec_size);
        auto b2 = rhs.load(i + 1 * vec_size);

        r1 = vec_type::template fmadd<false>(a1, b1, r1);
        r2 = vec_type::template fmadd<false>(a2, b2, r2);
    }

    for(; i + vec_size - 1 < n; i += vec_size){
        auto a1 = lhs.load(i);
        auto b1 = rhs.load(i);

        r1 = vec_type::template fmadd<false>(a1, b1, r1);
    }

    auto p1 = vec_type::hadd(r1) + vec_type::hadd(r2) + vec_type::hadd(r3) + vec_type::hadd(r4);
    auto p2 = vec_type::hadd(r5) + vec_type::hadd(r6) + vec_type::hadd(r7) + vec_type::hadd(r8);

    for(; i + 1 < n; i += 2){
        p1 += lhs[i] * rhs[i];
        p2 += lhs[i + 1] * rhs[i + 1];
    }

    if(i < n){
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
    // The default vectorization scheme should be sufficient
    return selected_dot<default_vec>(lhs, rhs);
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
