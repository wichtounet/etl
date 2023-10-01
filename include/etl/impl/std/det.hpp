//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the determinant
 */

#pragma once

namespace etl {

/*!
 * \copydoc etl::lu
 */
template <etl_expr AT, etl_expr LT, etl_expr UT, etl_expr PT>
bool lu(const AT& A, LT& L, UT& U, PT& P);

namespace impl {

namespace standard {

/*!
 * \brief Compute the determinant of the given matrix
 * \return The determinant of the given matrix
 */
template <etl_expr AT>
value_t<AT> det(const AT& A) {
    using T = value_t<AT>;

    const auto n = etl::dim<0>(A);

    if (is_permutation_matrix(A)) {
        size_t t = 0;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (A(i, j) != 0.0 && i != j) {
                    ++t;
                }
            }
        }

        return std::pow(T(-1.0), t - 1);
    }

    if (is_triangular(A)) {
        T det(1.0);

        for (size_t i = 0; i < n; ++i) {
            det *= A(i, i);
        }

        return det;
    }

    auto L = force_temporary_dim_only(A);
    auto U = force_temporary_dim_only(A);
    auto P = force_temporary_dim_only(A);

    etl::lu(A, L, U, P);

    return det(L) * det(U) * det(P);
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
