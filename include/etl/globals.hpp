//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains some global functions.
*/

#pragma once

#include "etl/temporary.hpp"

#include "etl/impl/decomposition.hpp"
#include "etl/impl/det.hpp"

namespace etl {

/*!
 * \brief Indicates if the given expression is a square matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a square matrix, false otherwise.
 */
template <typename E>
bool is_square(E&& expr) {
    return decay_traits<E>::dimensions() == 2 && etl::dim<0>(expr) == etl::dim<1>(expr);
}

/*!
 * \brief Indicates if the given expression is a real matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a real matrix, false otherwise.
 */
template <typename E>
bool is_real_matrix(E&& expr) {
    cpp_unused(expr);
    return !is_complex<E>::value;
}

/*!
 * \brief Indicates if the given expression is a complex matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a complex matrix, false otherwise.
 */
template <typename E>
bool is_complex_matrix(E&& expr) {
    cpp_unused(expr);
    return is_complex<E>::value;
}

/*!
 * \brief Indicates if the given expression is a rectangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a rectangular matrix, false otherwise.
 */
template <typename E>
bool is_rectangular(E&& expr) {
    return decay_traits<E>::dimensions() == 2 && etl::dim<0>(expr) != etl::dim<1>(expr);
}

/*!
 * \brief Indicates if the given expression contains sub matrices that are square.
 * \param expr The expression to test
 * \return true if the given expression contains sub matrices that are square, false otherwise.
 */
template <typename E>
bool is_sub_square(E&& expr) {
    return decay_traits<E>::dimensions() == 3 && etl::dim<1>(expr) == etl::dim<2>(expr);
}

/*!
 * \brief Indicates if the given expression contains sub matrices that are rectangular.
 * \param expr The expression to test
 * \return true if the given expression contains sub matrices that are rectangular, false otherwise.
 */
template <typename E>
bool is_sub_rectangular(E&& expr) {
    return decay_traits<E>::dimensions() == 3 && etl::dim<1>(expr) != etl::dim<2>(expr);
}

/*!
 * \brief Indicates if the given expression is a symmetric matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a symmetric matrix, false otherwise.
 */
template <typename E>
bool is_symmetric(E&& expr) {
    // sym_matrix<E> is already enforced to be symmetric
    if (is_symmetric_matrix<E>::value) {
        return true;
    }

    if (is_square(expr)) {
        for (std::size_t i = 0; i < etl::dim<0>(expr) - 1; ++i) {
            for (std::size_t j = i + 1; j < etl::dim<0>(expr); ++j) {
                if (expr(i, j) != expr(j, i)) {
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}

/*!
 * \brief Indicates if the given expression is a lower triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a lower triangular matrix, false otherwise.
 */
template <typename E>
bool is_lower_triangular(E&& expr) {
    if (is_square(expr)) {
        for (std::size_t i = 0; i < etl::dim<0>(expr) - 1; ++i) {
            for (std::size_t j = i + 1; j < etl::dim<0>(expr); ++j) {
                if (expr(i, j) != 0.0) {
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}

/*!
 * \brief Indicates if the given expression is a strictly lower triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a strictly lower triangular matrix, false otherwise.
 */
template <typename E>
bool is_strictly_lower_triangular(E&& expr) {
    if (is_square(expr)) {
        for (std::size_t i = 0; i < etl::dim<0>(expr); ++i) {
            for (std::size_t j = i; j < etl::dim<0>(expr); ++j) {
                if (expr(i, j) != 0.0) {
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}

/*!
 * \brief Indicates if the given expression is a upper triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a upper triangular matrix, false otherwise.
 */
template <typename E>
bool is_upper_triangular(E&& expr) {
    if (is_square(expr)) {
        for (std::size_t i = 1; i < etl::dim<0>(expr); ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                if (expr(i, j) != 0.0) {
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}

/*!
 * \brief Indicates if the given expression is a strictly upper triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a strictly upper triangular matrix, false otherwise.
 */
template <typename E>
bool is_strictly_upper_triangular(E&& expr) {
    if (is_square(expr)) {
        for (std::size_t i = 0; i < etl::dim<0>(expr); ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                if (expr(i, j) != 0.0) {
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}

/*!
 * \brief Indicates if the given expression is a triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a triangular matrix, false otherwise.
 */
template <typename E>
bool is_triangular(E&& expr) {
    return is_upper_triangular(expr) || is_lower_triangular(expr);
}

/*!
 * \brief Indicates if the given expression is a diagonal matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a diagonal matrix, false otherwise.
 */
template <typename E>
bool is_diagonal(E&& expr) {
    if (is_square(expr)) {
        for (std::size_t i = 0; i < etl::dim<0>(expr); ++i) {
            for (std::size_t j = 0; j < etl::dim<0>(expr); ++j) {
                if (i != j && expr(i, j) != 0.0) {
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}

/*!
 * \brief Indicates if the given expression is uniform (all elements of the same value)
 * \param expr The expression to test
 * \return true if the given expression is uniform, false otherwise.
 */
template <typename E>
bool is_uniform(E&& expr) {
    if (!etl::size(expr)) {
        return false;
    }

    auto first = *expr.begin();

    for (auto v : expr) {
        if (v != first) {
            return false;
        }
    }

    return true;
}

/*!
 * \brief Indicates if the given expression represents a permutation matrix
 * \param expr The expression to test
 * \return true if the given expression is an hermitian matrix, false otherwise.
 */
template <typename E>
bool is_permutation_matrix(E&& expr){
    if(!is_square(expr)){
        return false;
    }

    //Conditions:
    //a) Must be a square matrix
    //b) Every row must have one 1
    //c) Every column must have one 1

    for (std::size_t i = 0; i < etl::dim<0>(expr); ++i) {
        auto sum = value_t<E>(0);
        for (std::size_t j = 0; j < etl::dim<0>(expr); ++j) {
            if(expr(i, j) != value_t<E>(0) && expr(i, j) != value_t<E>(1)){
                return false;
            }

            sum += expr(i, j);
        }

        if(sum != value_t<E>(1)){
            return false;
        }
    }

    for (std::size_t j = 0; j < etl::dim<0>(expr); ++j) {
        auto sum = value_t<E>(0);
        for (std::size_t i = 0; i < etl::dim<0>(expr); ++i) {
            sum += expr(i, j);
        }

        if(sum != value_t<E>(1)){
            return false;
        }
    }

    return true;
}

/*!
 * \brief Indicates if the given expression represents an hermitian matrix
 * \param expr The expression to test
 * \return true if the given expression is an hermitian matrix, false otherwise.
 */
template <typename E, cpp_enable_if(is_complex<E>::value)>
bool is_hermitian(E&& expr){
    // herm_matrix<E> is already enforced to be hermitian
    if (is_hermitian_matrix<E>::value) {
        return true;
    }

    if(!is_square(expr)){
        return false;
    }

    for (std::size_t i = 0; i < etl::dim<0>(expr); ++i) {
        for (std::size_t j = 0; j < etl::dim<0>(expr); ++j) {
            if(i != j && expr(i, j) != get_conj(expr(j, i))){
                return false;
            }
        }
    }

    return true;
}

/*!
 * \brief Indicates if the given expression represents an hermitian matrix
 * \param expr The expression to test
 * \return true if the given expression is an hermitian matrix, false otherwise.
 */
template <typename E, cpp_disable_if(is_complex<E>::value)>
bool is_hermitian(E&& expr){
    cpp_unused(expr);
    return false;
}

/*!
 * \brief Returns the trace of the given square matrix.
 *
 * If the given expression does not represent a square matrix, this function will fail
 *
 * \param expr The expression to get the trace from.
 * \return The trace of the given expression
 */
template <typename E>
value_t<E> trace(E&& expr) {
    assert_square(expr);

    auto value = value_t<E>();

    for (std::size_t i = 0; i < etl::dim<0>(expr); ++i) {
        value += expr(i, i);
    }

    return value;
}

/*!
 * \brief Returns the determinant of the given square matrix.
 *
 * If the given expression does not represent a square matrix, this function will fail
 *
 * \param expr The expression to get the determinant from.
 * \return The determinant of the given expression
 */
template <typename E>
value_t<E> determinant(E&& expr) {
    assert_square(expr);

    return detail::det_impl::apply(expr);
}

/*!
 * \brief Decomposition the matrix so that P * A = L * U
 * \param A The A matrix
 * \param L The L matrix (Lower Diagonal)
 * \param U The U matrix (Upper Diagonal)
 * \param P The P matrix (Pivot Permutation Matrix)
 * \return true if the decomposition suceeded, false otherwise
 */
template <typename AT, typename LT, typename UT, typename PT>
bool lu(const AT& A, LT& L, UT& U, PT& P) {
    // All matrices must be square
    if (!is_square(A) || !is_square(L) || !is_square(U) || !is_square(P)) {
        return false;
    }

    // All matrices must be of the same dimension
    if (etl::dim(A, 0) != etl::dim(L, 0) || etl::dim(A, 0) != etl::dim(U, 0) || etl::dim(A, 0) != etl::dim(P, 0)) {
        return true;
    }

    detail::lu_impl::apply(A, L, U, P);

    return true;
}

} //end of namespace etl
