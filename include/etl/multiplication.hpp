//=======================================================================
// Copyright (c) 2014 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef ETL_MULTIPLICATION_HPP
#define ETL_MULTIPLICATION_HPP

#include <algorithm>

#include "tmp.hpp"

namespace etl {

template<typename A, typename B, typename C>
static C& mmul(const A& a, const B& b, C& c){
    static_assert(is_etl_expr<A>::value && is_etl_expr<B>::value && is_etl_expr<C>::value, "Matrix multiplication only supported for ETL expressions");
    //static_assert(A::columns == B::rows, "The central dimensions of the multiplied matrices must be the same");
    //static_assert(C::rows == A::rows && C::columns == B::columns, "The output matrix is not of the good dimension");

    c = 0;

    for(std::size_t i = 0; i < rows(a); i++){
        for(std::size_t j = 0; j < columns(b); j++){
            for(std::size_t k = 0; k < columns(a); k++){
                c(i,j) += a(i,k) * b(k,j);
            }
        }
    }
    
    return c;
}

} //end of namespace etl

#endif
