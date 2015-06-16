//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "../../traits_lite.hpp"

namespace etl {

namespace impl {

namespace standard {

template<typename A, typename B, typename C>
static void mm_mul(A&& a, B&& b, C&& c){
    c = 0;

    for(std::size_t i = 0; i < rows(a); i++){
        for(std::size_t k = 0; k < columns(a); k++){
            for(std::size_t j = 0; j < columns(b); j++){
                c(i,j) += a(i,k) * b(k,j);
            }
        }
    }
}

template<typename A, typename B, typename C>
static void vm_mul(A&& a, B&& b, C&& c){
    c = 0;

    for(std::size_t k = 0; k < etl::dim<0>(a); k++){
        for(std::size_t j = 0; j < columns(b); j++){
            c(j) += a(k) * b(k,j);
        }
    }
}

template<typename A, typename B, typename C>
static void mv_mul(A&& a, B&& b, C&& c){
    c = 0;

    for(std::size_t i = 0; i < rows(a); i++){
        for(std::size_t k = 0; k < columns(a); k++){
            c(i) += a(i,k) * b(k);
        }
    }
}

} //end of namespace standard

} //end of namespace impl

} //end of namespace etl
