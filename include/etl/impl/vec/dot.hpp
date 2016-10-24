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

    for(; i + vec_size - 1 < n; i += vec_size){
        auto a1 = lhs.load(i);
        auto b1 = rhs.load(i);

        auto t1 = vec_type::template mul<false>(a1, b1);
        r1 = vec_type::add(r1, t1);
    }

    auto product = vec_type::hadd(r1);

    for(; i < n; ++i){
        product += lhs[i] * rhs[i];
    }

    return product;
}

/*!
 * \brief Compute the dot product of a and b
 * \param a The lhs expression
 * \param b The rhs expression
 * \return the sum
 */
template <typename L, typename R>
value_t<L> dot(const L& lhs, const R& rhs) {
    // The default vectorization scheme should be sufficient
    return selected_dot<default_vec>(lhs, rhs);
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
