//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Standard implementation of the "transpose" algorithm
 */

#pragma once

#include "etl/temporary.hpp"

namespace etl {

namespace impl {

namespace standard {

/*!
 * \brief Inplace transposition of the square matrix c
 * \param c The matrix to transpose
 */
template <typename C>
void inplace_square_transpose(C&& c) {
    using std::swap;

    const std::size_t N = etl::dim<0>(c);

    for (std::size_t i = 0; i < N - 1; ++i) {
        for (std::size_t j = i + 1; j < N; ++j) {
            swap(c(i, j), c(j, i));
        }
    }
}

/*!
 * \brief Inplace transposition of the rectangular matrix c
 * \param mat The matrix to transpose
 */
template <typename C>
void inplace_rectangular_transpose(C&& mat) {
    auto copy = force_temporary(mat);

    auto data = mat.memory_start();

    //Dimensions prior to transposition
    const std::size_t N = etl::dim<0>(mat);
    const std::size_t M = etl::dim<1>(mat);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            data[j * N + i] = copy(i, j);
        }
    }
}

/*!
 * \brief Perform an inplace matrix transposition in O(1).
 *
 * This implementation is quite slow and should only be used when space is of
 * the essence.
 *
 * \param mat the matrix to transpose inplace
 */
template <typename C>
void real_inplace(C&& mat) {
    using std::swap;

    const std::size_t N = etl::dim<0>(mat);
    const std::size_t M = etl::dim<1>(mat);

    auto data = mat.memory_start();

    for (std::size_t k = 0; k < N * M; k++) {
        auto idx = k;
        do {
            idx = (idx % N) * M + (idx / N);
        } while (idx < k);
        std::swap(data[k], data[idx]);
    }
}

/*!
 * \brief Transpose the matrix a and the store the result in c
 * \param a The matrix to transpose
 * \param c The target matrix
 */
template <typename A, typename C>
void transpose(A&& a, C&& c) {
    auto mem_c = c.memory_start();
    auto mem_a = a.memory_start();

    // Delegate aliasing transpose to inplace algorithm
    if (mem_c == mem_a) {
        if (etl::dim<0>(a) == etl::dim<1>(a)) {
            inplace_square_transpose(c);
        } else {
            inplace_rectangular_transpose(c);
        }
    } else {
        if (decay_traits<A>::storage_order == order::RowMajor) {
            for (std::size_t i = 0; i < etl::dim<0>(a); ++i) {
                for (std::size_t j = 0; j < etl::dim<1>(a); ++j) {
                    mem_c[j * etl::dim<1>(c) + i] = mem_a[i * etl::dim<1>(a) + j];
                }
            }
        } else {
            for (std::size_t j = 0; j < etl::dim<1>(a); ++j) {
                for (std::size_t i = 0; i < etl::dim<0>(a); ++i) {
                    mem_c[i * etl::dim<0>(c) + j] = mem_a[j * etl::dim<0>(a) + i];
                }
            }
        }
    }
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
