//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementations of inplace matrix transposition
 */

#pragma once

//Include the implementations
#include "etl/impl/std/transpose.hpp"
#include "etl/impl/blas/transpose.hpp"

namespace etl {

namespace detail {

/*!
 * \brief Enumeration describing the different implementations of transpose
 */
enum class transpose_imple {
    STD, ///< Standard implementation
    MKL, ///< MKL implementation
};

/*!
 * \brief Select the trnaspose implementation for an expression of type A and C
 * \tparam A The type of rhs expression
 * \tparam C The type of lhs expression
 * \return The implementation to use
 */
template <typename A, typename C>
cpp14_constexpr transpose_imple select_transpose_impl() {
    if(all_dma<A, C>::value && all_floating<A, C>::value){
        if (is_mkl_enabled) {
            return transpose_imple::MKL;
        } else {
            return transpose_imple::STD;
        }
    }

    return transpose_imple::STD;
}

struct inplace_square_transpose {
    template <typename C>
    static void apply(C&& c) {
        cpp14_constexpr const auto impl = select_transpose_impl<C, C>();

        if(impl == transpose_imple::MKL){
            etl::impl::blas::inplace_square_transpose(c);
        } else {
            etl::impl::standard::inplace_square_transpose(c);
        }
    }
};

struct inplace_rectangular_transpose {
    template <typename C>
    static void apply(C&& c) {
        cpp14_constexpr const auto impl = select_transpose_impl<C, C>();

        if(impl == transpose_imple::MKL){
            etl::impl::blas::inplace_rectangular_transpose(c);
        } else {
            etl::impl::standard::inplace_rectangular_transpose(c);
        }
    }
};

struct transpose {
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        cpp14_constexpr const auto impl = select_transpose_impl<A, C>();

        if(impl == transpose_imple::MKL){
            etl::impl::blas::transpose(a, c);
        } else {
            etl::impl::standard::transpose(a, c);
        }
    }
};

} //end of namespace detail

} //end of namespace etl
