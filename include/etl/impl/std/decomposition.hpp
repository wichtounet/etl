//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the decompositions
 */

#pragma once

namespace etl {

namespace impl {

namespace standard {

/*!
 * \brief Performs the PA=LU decomposition of the matrix A
 * \param A The matrix to decompose
 * \param L The resulting L matrix
 * \param U The resulting U matrix
 * \param P The resulting P matrix
 */
template <typename AT, typename LT, typename UT, typename PT>
void lu(const AT& A, LT& L, UT& U, PT& P) {
    const auto n = etl::dim(A, 0);

    L = 0;
    U = 0;
    P = 0;

    // 1. Create the pivot matrix

    for(std::size_t i = 0; i < n; ++i){
        P(i, i) = 1;
    }

    for(std::size_t i = 0; i < n; ++i){
        auto max_j = i;

        for(std::size_t j = i; j < n; ++j){
            if(std::abs(A(j, i)) > A(max_j, i)){
                max_j = j;
            }
        }

        if(max_j != i){
            for(std::size_t k = 0; k < n; ++k){
                using std::swap;
                swap(P(i, k), P(max_j, k));
            }
        }
    }

    auto Ap = etl::force_temporary(P * A);

    for(std::size_t i = 0; i < n; ++i){
        L(i, i) = 1;
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j <= i) {
                value_t<AT> s = 0;
                for (std::size_t k = 0; k < j; ++k) {
                    s += L(j, k) * U(k, i);
                }

                U(j, i) = Ap(j, i) - s;
            }

            if (j >= i) {
                value_t<AT> s = 0;
                for (std::size_t k = 0; k < i; ++k) {
                    s += L(j, k) * U(k, i);
                }

                L(j, i) = (Ap(j, i) - s) / U(i, i);
            }
        }
    }
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
