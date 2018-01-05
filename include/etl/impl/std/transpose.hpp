//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
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

    const size_t N = etl::dim<0>(c);

    for (size_t i = 0; i < N - 1; ++i) {
        size_t j = i + 1;

        for (; j + 1 < N; j += 2) {
            swap(c(i, j + 0), c(j + 0, i));
            swap(c(i, j + 1), c(j + 1, i));
        }

        if (j < N) {
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
    static constexpr bool row_major = decay_traits<C>::storage_order == order::RowMajor;

    auto copy = force_temporary(mat);

    //Dimensions prior to transposition
    const size_t N = etl::dim<0>(mat);
    const size_t M = etl::dim<1>(mat);

    // Note: cannot use operator(i,j) for lhs because it is indexed by its
    // previous scheme (N instead of M)

    if /*constexpr*/ (row_major) {
        for (size_t i = 0; i < N; ++i) {
            size_t j = 0;

            for (; j + 1 < M; j += 2) {
                mat[(j + 0) * N + i] = copy(i, (j + 0));
                mat[(j + 1) * N + i] = copy(i, (j + 1));
            }

            if (j < M) {
                mat[j * N + i] = copy(i, j);
            }
        }
    } else {
        for (size_t i = 0; i < N; ++i) {
            size_t j = 0;

            for (; j + 1 < M; j += 2) {
                mat[(j + 0) + M * i] = copy(i, (j + 0));
                mat[(j + 1) + M * i] = copy(i, (j + 1));
            }

            if (j < M) {
                mat[j + M * i] = copy(i, j);
            }
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

    const size_t N = etl::dim<0>(mat);
    const size_t M = etl::dim<1>(mat);

    mat.ensure_cpu_up_to_date();

    auto data = mat.memory_start();

    for (size_t k = 0; k < N * M; k++) {
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
    // Delegate aliasing transpose to inplace algorithm
    if (a.alias(c)) {
        if (etl::dim<0>(a) == etl::dim<1>(a)) {
            inplace_square_transpose(c);
        } else {
            inplace_rectangular_transpose(c);
        }
    } else {
        const size_t m = etl::dim<0>(a);
        const size_t n = etl::dim<1>(a);

        // Note: cannot use operator(i,j) for rhs because it is indexed by its
        // previous scheme (M instead of N)

        if /*constexpr*/ (decay_traits<A>::storage_order == order::RowMajor) {
            for (size_t i = 0; i < m; ++i) {
                size_t j = 0;

                for (; j + 3 < n; j += 4) {
                    c[(j + 0) * m + i] = a[i * n + j + 0];
                    c[(j + 1) * m + i] = a[i * n + j + 1];
                    c[(j + 2) * m + i] = a[i * n + j + 2];
                    c[(j + 3) * m + i] = a[i * n + j + 3];
                }

                for (; j + 1 < n; j += 2) {
                    c[(j + 0) * m + i] = a[i * n + j + 0];
                    c[(j + 1) * m + i] = a[i * n + j + 1];
                }

                if (j < n) {
                    c[(j + 0) * m + i] = a[i * n + j];
                }
            }
        } else {
            for (size_t j = 0; j < n; ++j) {
                for (size_t i = 0; i < m; ++i) {
                    c[i * n + j] = a[j * m + i];
                }
            }
        }
    }
}

} //end of namespace standard
} //end of namespace impl
} //end of namespace etl
