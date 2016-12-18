//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief MKL implementation of the transpose algorithm
 */

#pragma once

#ifdef ETL_MKL_MODE
#include "mkl.h"
#include "mkl_trans.h"
#endif

namespace etl {

namespace impl {

namespace blas {

#ifdef ETL_MKL_MODE

/*!
 * \brief MKL out-of-place transposition wrapper
 * \param a The right hand side
 * \param c The left hand side
 */
template <typename A, typename C, cpp_enable_if(all_single_precision<A, C>::value)>
void mkl_otrans(A&& a, C&& c) {
    auto mem_c = c.memory_start();
    auto mem_a = a.memory_start();

    if (decay_traits<A>::storage_order == order::RowMajor) {
        mkl_somatcopy('R', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0f, mem_a, etl::dim<1>(a), mem_c, etl::dim<0>(a));
    } else {
        mkl_somatcopy('C', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0f, mem_a, etl::dim<0>(a), mem_c, etl::dim<1>(a));
    }
}

/*!
 * \brief MKL out-of-place transposition wrapper
 * \param a The right hand side
 * \param c The left hand side
 */
template <typename A, typename C, cpp_enable_if(all_double_precision<A, C>::value)>
void mkl_otrans(A&& a, C&& c) {
    auto mem_c = c.memory_start();
    auto mem_a = a.memory_start();

    if (decay_traits<A>::storage_order == order::RowMajor) {
        mkl_domatcopy('R', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0, mem_a, etl::dim<1>(a), mem_c, etl::dim<0>(a));
    } else {
        mkl_domatcopy('C', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0, mem_a, etl::dim<0>(a), mem_c, etl::dim<1>(a));
    }
}

/*!
 * \brief MKL inplace transposition wrapper
 * \param c The left hand side
 */
template <typename C, cpp_enable_if(all_single_precision<C>::value)>
void mkl_itrans(C&& c) {
    if (decay_traits<C>::storage_order == order::RowMajor) {
        mkl_simatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<1>(c), etl::dim<0>(c));
    } else {
        mkl_simatcopy('C', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<0>(c), etl::dim<1>(c));
    }
}

/*!
 * \brief MKL inplace transposition wrapper
 * \param c The left hand side
 */
template <typename C, cpp_enable_if(all_double_precision<C>::value)>
void mkl_itrans(C&& c) {
    if (decay_traits<C>::storage_order == order::RowMajor) {
        mkl_dimatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, c.memory_start(), etl::dim<1>(c), etl::dim<0>(c));
    } else {
        mkl_dimatcopy('C', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, c.memory_start(), etl::dim<0>(c), etl::dim<1>(c));
    }
}

/*!
 * \brief Inplace transposition of the square matrix c
 * \param c The matrix to transpose
 */
template <typename C, cpp_enable_if(all_dma<C>::value&& all_floating<C>::value)>
void inplace_square_transpose(C&& c) {
    mkl_itrans(c);
}

/*!
 * \brief Inplace transposition of the rectangular matrix c
 * \param c The matrix to transpose
 */
template <typename C, cpp_enable_if(all_dma<C>::value&& all_floating<C>::value)>
void inplace_rectangular_transpose(C&& c) {
    mkl_otrans(force_temporary(c), c);
}

/*!
 * \brief Transpose the matrix a and the store the result in c
 * \param a The matrix to transpose
 * \param c The target matrix
 */
template <typename A, typename C, cpp_enable_if(all_dma<A, C>::value&& all_floating<A, C>::value)>
void transpose(A&& a, C&& c) {
    auto mem_c = c.memory_start();
    auto mem_a = a.memory_start();

    // Delegate aliasing transpose to inplace algorithm
    if (mem_c == mem_a) {
        if (etl::dim<0>(a) == etl::dim<1>(a)) {
            mkl_itrans(c);
        } else {
            mkl_otrans(force_temporary(c), c);
        }
    } else {
        mkl_otrans(a, c);
    }
}

/*!
 * \brief Inplace transposition of the square matrix c
 * \param c The matrix to transpose
 */
template <typename C, cpp_disable_if(all_dma<C>::value&& all_floating<C>::value)>
void inplace_square_transpose(C&& /*c*/) {}

/*!
 * \brief Inplace transposition of the rectangular matrix c
 * \param c The matrix to transpose
 */ template <typename C, cpp_disable_if(all_dma<C>::value&& all_floating<C>::value)>
void inplace_rectangular_transpose(C&& /*c*/) {}

/*!
 * \brief Transpose the matrix a and the store the result in c
 * \param a The matrix to transpose
 * \param c The target matrix
 */
template <typename A, typename C, cpp_disable_if(all_dma<A, C>::value&& all_floating<A, C>::value)>
void transpose(A&& /*a*/, C&& /*c*/) {}

#else

//COVERAGE_EXCLUDE_BEGIN

/*!
 * \brief Inplace transposition of the square matrix c
 * \param c The matrix to transpose
 */
template <typename C>
void inplace_square_transpose(C&& c) {
    cpp_unused(c);
    cpp_unreachable("MKL not enabled/available");
}

/*!
 * \brief Inplace transposition of the rectangular matrix c
 * \param c The matrix to transpose
 */
template <typename C>
void inplace_rectangular_transpose(C&& c) {
    cpp_unused(c);
    cpp_unreachable("MKL not enabled/available");
}

/*!
 * \brief Transpose the matrix a and the store the result in c
 * \param a The matrix to transpose
 * \param c The target matrix
 */
template <typename A, typename C>
void transpose(A&& a, C&& c) {
    cpp_unused(a);
    cpp_unused(c);
    cpp_unreachable("MKL not enabled/available");
}

//COVERAGE_EXCLUDE_END

#endif

} //end of namespace blas
} //end of namespace impl
} //end of namespace etl
