//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

namespace etl::impl::blas {

#ifdef ETL_MKL_MODE

/*!
 * \brief MKL out-of-place transposition wrapper
 * \param a The right hand side
 * \param c The left hand side
 */
template <typename A, typename C>
void mkl_otrans(A&& a, C&& c) {
    a.ensure_cpu_up_to_date();

    auto mem_c = c.memory_start();
    auto mem_a = a.memory_start();

    if constexpr (all_single_precision<A, C>) {
        if (is_row_major<A>) {
            mkl_somatcopy('R', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0f, mem_a, etl::dim<1>(a), mem_c, etl::dim<0>(a));
        } else {
            mkl_somatcopy('C', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0f, mem_a, etl::dim<0>(a), mem_c, etl::dim<1>(a));
        }
    } else if constexpr (all_double_precision<A, C>) {
        if (is_row_major<A>) {
            mkl_domatcopy('R', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0, mem_a, etl::dim<1>(a), mem_c, etl::dim<0>(a));
        } else {
            mkl_domatcopy('C', 'T', etl::dim<0>(a), etl::dim<1>(a), 1.0, mem_a, etl::dim<0>(a), mem_c, etl::dim<1>(a));
        }
    }

    c.invalidate_gpu();
}

/*!
 * \brief MKL inplace transposition wrapper
 * \param c The left hand side
 */
template <typename C>
void mkl_itrans(C&& c) {
    c.ensure_cpu_up_to_date();

    if constexpr (is_single_precision<C>) {
        if (is_row_major<C>) {
            mkl_simatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<1>(c), etl::dim<0>(c));
        } else {
            mkl_simatcopy('C', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0f, c.memory_start(), etl::dim<0>(c), etl::dim<1>(c));
        }
    } else if constexpr (is_double_precision<C>) {
        if (is_row_major<C>) {
            mkl_dimatcopy('R', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, c.memory_start(), etl::dim<1>(c), etl::dim<0>(c));
        } else {
            mkl_dimatcopy('C', 'T', etl::dim<0>(c), etl::dim<1>(c), 1.0, c.memory_start(), etl::dim<0>(c), etl::dim<1>(c));
        }
    }

    c.invalidate_gpu();
}

/*!
 * \brief Inplace transposition of the square matrix c
 * \param c The matrix to transpose
 */
template <typename C>
void inplace_square_transpose(C&& c) {
    if constexpr (is_dma<C> && is_floating<C>) {
        mkl_itrans(c);
    } else {
        cpp_unreachable("Invalid call to blas:inplace_square_transpose");
    }
}

/*!
 * \brief Inplace transposition of the rectangular matrix c
 * \param c The matrix to transpose
 */
template <typename C>
void inplace_rectangular_transpose(C&& c) {
    if constexpr (is_dma<C> && is_floating<C>) {
        mkl_otrans(force_temporary(c), c);
    } else {
        cpp_unreachable("Invalid call to blas:inplace_rectangular_transpose");
    }
}

/*!
 * \brief Transpose the matrix a and the store the result in c
 * \param a The matrix to transpose
 * \param c The target matrix
 */
template <typename A, typename C>
void transpose(A&& a, C&& c) {
    if constexpr (all_dma<A, C> && all_floating<A, C>) {
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
    } else {
        cpp_unreachable("Invalid call to blas:tranpose");
    }
}

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

} //end of namespace etl::impl::blas
