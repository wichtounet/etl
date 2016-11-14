//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Unified vectorized implementation of the "sum" reduction
 */

#pragma once

namespace etl {

namespace impl {

namespace vec {

template <typename V, typename L>
value_t<L> selected_sum(const L& lhs, size_t first, size_t last) {
    using vec_type = V;
    using T        = value_t<L>;

    static constexpr size_t vec_size = vec_type::template traits<T>::size;

    auto n = last - first;

    size_t i = first;

    auto r1 = vec_type::template zero<T>();
    auto r2 = vec_type::template zero<T>();
    auto r3 = vec_type::template zero<T>();
    auto r4 = vec_type::template zero<T>();
    auto r5 = vec_type::template zero<T>();
    auto r6 = vec_type::template zero<T>();
    auto r7 = vec_type::template zero<T>();
    auto r8 = vec_type::template zero<T>();

    if (n > 1000) {
        for (; i + (vec_size * 8) - 1 < last; i += 8 * vec_size) {
            r1 = vec_type::add(lhs.template load<vec_type>(i + 0 * vec_size), r1);
            r2 = vec_type::add(lhs.template load<vec_type>(i + 1 * vec_size), r2);
            r3 = vec_type::add(lhs.template load<vec_type>(i + 2 * vec_size), r3);
            r4 = vec_type::add(lhs.template load<vec_type>(i + 3 * vec_size), r4);
            r5 = vec_type::add(lhs.template load<vec_type>(i + 4 * vec_size), r5);
            r6 = vec_type::add(lhs.template load<vec_type>(i + 5 * vec_size), r6);
            r7 = vec_type::add(lhs.template load<vec_type>(i + 6 * vec_size), r7);
            r8 = vec_type::add(lhs.template load<vec_type>(i + 7 * vec_size), r8);
        }
    }

    for (; i + (vec_size * 4) - 1 < last; i += 4 * vec_size) {
        r1 = vec_type::add(lhs.template load<vec_type>(i + 0 * vec_size), r1);
        r2 = vec_type::add(lhs.template load<vec_type>(i + 1 * vec_size), r2);
        r3 = vec_type::add(lhs.template load<vec_type>(i + 2 * vec_size), r3);
        r4 = vec_type::add(lhs.template load<vec_type>(i + 3 * vec_size), r4);
    }

    for(; i + (vec_size * 2) - 1 < last; i += 2 * vec_size){
        r1 = vec_type::add(lhs.template load<vec_type>(i + 0 * vec_size), r1);
        r2 = vec_type::add(lhs.template load<vec_type>(i + 1 * vec_size), r2);
    }

    for(; i + vec_size - 1 < last; i += vec_size){
        r1 = vec_type::add(lhs.template load<vec_type>(i + 0 * vec_size), r1);
    }

    auto p1 = vec_type::hadd(r1) + vec_type::hadd(r2) + vec_type::hadd(r3) + vec_type::hadd(r4);
    auto p2 = vec_type::hadd(r5) + vec_type::hadd(r6) + vec_type::hadd(r7) + vec_type::hadd(r8);

    for(; i + 1 < last; i += 2){
        p1 += lhs[i];
        p2 += lhs[i + 1];
    }

    if(i < last){
        p1 += lhs[i];
    }

    return p1 + p2;
}

/*!
 * \brief Compute the sum of lhs
 * \param lhs The lhs expression
 * \return the sum of the elements of lhs
 */
template <typename L, cpp_enable_if((vec_enabled && all_vectorizable<vector_mode, L>::value))>
value_t<L> sum(const L& lhs, size_t first, size_t last) {
    cpp_assert(vec_enabled, "At least one vector mode must be enabled for impl::VEC");

    // The default vectorization scheme should be sufficient
    return selected_sum<default_vec>(lhs, first, last);
}

/*!
 * \brief Compute the sum of lhs
 * \param lhs The lhs expression
 * \return the sum of the elements of lhs
 */
template <typename L, cpp_disable_if((vec_enabled && all_vectorizable<vector_mode, L>::value))>
value_t<L> sum(const L& lhs, size_t first, size_t last) {
    cpp_unused(lhs);
    cpp_unused(first);
    cpp_unused(last);
    cpp_unreachable("vec::sum called with invalid parameters");
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
