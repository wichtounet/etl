//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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

template <typename AT, typename LT, typename UT, typename PT>
bool lu(const AT& A, LT& L, UT& U, PT& P);

namespace impl {

namespace standard {

template <typename AT>
value_t<AT> det(const AT& A) {
    const auto n = etl::dim<0>(A);

    if(is_permutation_matrix(A)){
        std::size_t t = 0;
        for(std::size_t i = 0; i < n; ++i){
            for(std::size_t j = 0; j < n; ++j){
                if(A(i, j) != 0.0 && i != j){
                    ++t;
                }
            }
        }

        return std::pow(value_t<AT>(-1.0), t - 1);
    }

    if(is_triangular(A)){
        value_t<AT> det(1.0);
        for(std::size_t i = 0; i < n; ++i){
            det *= A(i, i);
        }
        return det;
    }

    auto L = force_temporary_dyn(A);
    auto U = force_temporary_dyn(A);
    auto P = force_temporary_dyn(A);

    etl::lu(A, L, U, P);

    return det(L) * det(U) * det(P);
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
