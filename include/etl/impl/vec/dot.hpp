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

    if (n < 1000000) {
        for(; i + (vec_size * 4) - 1 < n; i += 4 * vec_size){
            auto a1 = lhs.load(i + 0 * vec_size);
            auto a2 = lhs.load(i + 1 * vec_size);
            auto a3 = lhs.load(i + 2 * vec_size);
            auto a4 = lhs.load(i + 3 * vec_size);

            auto b1 = rhs.load(i + 0 * vec_size);
            auto b2 = rhs.load(i + 1 * vec_size);
            auto b3 = rhs.load(i + 2 * vec_size);
            auto b4 = rhs.load(i + 3 * vec_size);

            auto t1 = vec_type::template mul<false>(a1, b1);
            auto t2 = vec_type::template mul<false>(a2, b2);
            auto t3 = vec_type::template mul<false>(a3, b3);
            auto t4 = vec_type::template mul<false>(a4, b4);

            r1 = vec_type::add(r1, t1);
            r2 = vec_type::add(r2, t2);
            r3 = vec_type::add(r3, t3);
            r4 = vec_type::add(r4, t4);
        }
    }

    for(; i + vec_size - 1 < n; i += vec_size){
        auto a1 = lhs.load(i);
        auto b1 = rhs.load(i);

        auto t1 = vec_type::template mul<false>(a1, b1);
        r1 = vec_type::add(r1, t1);
    }

    auto product = vec_type::hadd(r1) + vec_type::hadd(r2) + vec_type::hadd(r3) + vec_type::hadd(r4);

    for(; i < n; ++i){
        product += lhs[i] * rhs[i];
    }

    return product;
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
