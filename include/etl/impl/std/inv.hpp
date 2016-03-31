//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/allocator.hpp"

namespace etl {

template <typename AT, typename LT, typename UT, typename PT>
bool lu(const AT& A, LT& L, UT& U, PT& P);

namespace impl {

namespace standard {

/*!
 * \brief Compute inv(a) and store the result in c
 * \param a The input expression
 * \param c The output expression
 */
template <typename A, typename C>
void inv(A&& a, C&& c) {
    // The inverse of a permutation matrix is its transpose
    if(is_permutation_matrix(a)){
        c = etl::transpose(a);
        return;
    }

    const auto n = etl::dim<0>(a);

    // Use forward propagation for lower triangular matrix
    if(is_lower_triangular(a)){
        c = 0;

        // The column in c
        for (std::size_t s = 0; s < n; ++s) {
            // The row in a
            for (std::size_t row = 0; row < n; ++row) {
                auto b = row == s ? 1.0 : 0.0;

                if (row == 0) {
                    c(0, s) = b / a(0, 0);
                } else {
                    value_t<A> acc(0);

                    // The column in a
                    for (std::size_t col = 0; col < row; ++col) {
                        acc += a(row, col) * c(col, s);
                    }

                    c(row, s) = (b - acc) / a(row, row);
                }
            }
        }

        return;
    }

    // Use backward propagation for upper triangular matrix
    if(is_upper_triangular(a)){
        c = 0;

        // The column in c
        for (long s = n - 1; s >= 0; --s) {
            // The row in a
            for (long row = n - 1; row >= 0; --row) {
                auto b = row == s ? 1.0 : 0.0;

                if (row == long(n) - 1) {
                    c(row, s) = b / a(row, row);
                } else {
                    value_t<A> acc(0);

                    // The column in a
                    for (long col = n - 1; col > row; --col) {
                        acc += a(row, col) * c(col, s);
                    }

                    c(row, s) = (b - acc) / a(row, row);
                }
            }
        }

        return;
    }

    auto L = force_temporary_dyn(a);
    auto U = force_temporary_dyn(a);
    auto P = force_temporary_dyn(a);

    etl::lu(a, L, U, P);

    c = inv(U) * inv(L) * P;
}

} //end of namespace standard

} //end of namespace impl

} //end of namespace etl
