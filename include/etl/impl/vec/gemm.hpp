//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace impl {

namespace vec {

template <typename A, typename B, typename C, cpp_enable_if((all_row_major<A, B, C>::value))>
void gemv(A&& a, B&& b, C&& c) {
    const auto m = rows(a);
    const auto n = columns(a);

    c = 0;

    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < n; k++) {
            c(i) += a(i, k) * b(k);
        }
    }
}

// Default, unoptimized should not be called unless in tests
template <typename A, typename B, typename C, cpp_disable_if((all_row_major<A, B, C>::value))>
void gemv(A&& a, B&& b, C&& c) {
    const auto m = rows(a);
    const auto n = columns(a);

    c = 0;

    for (size_t i = 0; i < m; i++) {
        for (size_t k = 0; k < n; k++) {
            c(i) += a(i, k) * b(k);
        }
    }
}

} //end of namespace vec

} //end of namespace impl

} //end of namespace etl
