//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/traits_lite.hpp"

namespace etl {

namespace impl {

namespace standard {

template<typename A, typename B, typename C>
static void mm_mul(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    c = 0;

    if(row_major){
        for(std::size_t i = 0; i < rows(a); i++){
            for(std::size_t k = 0; k < columns(a); k++){
                for(std::size_t j = 0; j < columns(b); j++){
                    c(i,j) += a(i,k) * b(k,j);
                }
            }
        }
    } else {
        for(std::size_t j = 0; j < columns(b); j++){
            for(std::size_t k = 0; k < columns(a); k++){
                for(std::size_t i = 0; i < rows(a); i++){
                    c(i,j) += a(i,k) * b(k,j);
                }
            }
        }
    }
}

inline void add_mul(float& c, float a, float b){
    c += a * b;
}

inline void add_mul(double& c, double a, double b){
    c += a * b;
}

//Note: For some reason, compilers have a real hard time
//inlining/vectorizing std::complex operations
//This helper improves performance by more than 50% on some cases

template<typename T>
inline void add_mul(std::complex<T>& c, std::complex<T> a, std::complex<T> b){
    auto ac = a.real() * b.real();
    auto bd = a.imag() * b.imag();

    auto abcd = (a.real() + a.imag()) * (b.real() + b.imag());

    c.real(c.real() + ac - bd);
    c.imag(c.imag() + abcd - ac - bd);
}

template<typename A, typename B, typename C>
static void vm_mul(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<B>::storage_order == order::RowMajor;

    c = 0;

    if(row_major){
        for(std::size_t k = 0; k < etl::dim<0>(a); k++){
            for(std::size_t j = 0; j < columns(b); j++){
                //c(j) += a(k) * b(k,j);
                add_mul(c(j), a(k), b(k,j));
            }
        }
    } else {
        for(std::size_t j = 0; j < columns(b); j++){
            for(std::size_t k = 0; k < etl::dim<0>(a); k++){
                //c(j) += a(k) * b(k,j);
                add_mul(c(j), a(k), b(k,j));
            }
        }
    }
}

template<typename A, typename B, typename C>
static void mv_mul(A&& a, B&& b, C&& c){
    bool row_major = decay_traits<A>::storage_order == order::RowMajor;

    c = 0;

    if(row_major){
        for(std::size_t i = 0; i < rows(a); i++){
            for(std::size_t k = 0; k < columns(a); k++){
                //c(i) += a(i,k) * b(k);
                add_mul(c(i), a(i,k), b(k));
            }
        }
    } else {
        for(std::size_t k = 0; k < columns(a); k++){
            for(std::size_t i = 0; i < rows(a); i++){
                //c(i) += a(i,k) * b(k);
                add_mul(c(i), a(i,k), b(k));
            }
        }
    }
}

} //end of namespace standard

} //end of namespace impl

} //end of namespace etl
