//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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

#include "etl/impl/egblas/shuffle.hpp"

namespace etl {

/*!
 * \brief Indicates if the given expression is a square matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a square matrix, false otherwise.
 */
template <typename E>
bool is_square(E&& expr) {
    return is_2d<E> && etl::dim<0>(expr) == etl::dim<1>(expr);
}

/*!
 * \brief Indicates if the given expression is a real matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a real matrix, false otherwise.
 */
template <typename E>
bool is_real_matrix([[maybe_unused]] E&& expr) {
    return !is_complex<E>;
}

/*!
 * \brief Indicates if the given expression is a complex matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a complex matrix, false otherwise.
 */
template <typename E>
bool is_complex_matrix([[maybe_unused]] E&& expr) {
    return is_complex<E>;
}

/*!
 * \brief Indicates if the given expression is a rectangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a rectangular matrix, false otherwise.
 */
template <typename E>
bool is_rectangular(E&& expr) {
    return is_2d<E> && etl::dim<0>(expr) != etl::dim<1>(expr);
}

/*!
 * \brief Indicates if the given expression contains sub matrices that are square.
 * \param expr The expression to test
 * \return true if the given expression contains sub matrices that are square, false otherwise.
 */
template <typename E>
bool is_sub_square(E&& expr) {
    return is_3d<E> && etl::dim<1>(expr) == etl::dim<2>(expr);
}

/*!
 * \brief Indicates if the given expression contains sub matrices that are rectangular.
 * \param expr The expression to test
 * \return true if the given expression contains sub matrices that are rectangular, false otherwise.
 */
template <typename E>
bool is_sub_rectangular(E&& expr) {
    return is_3d<E> && etl::dim<1>(expr) != etl::dim<2>(expr);
}

/*!
 * \brief Indicates if the given expression is a symmetric matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a symmetric matrix, false otherwise.
 */
template <typename E>
bool is_symmetric(E&& expr) {
    // symmetric_matrix<E> is already enforced to be symmetric
    if constexpr (is_symmetric_matrix<E>) {
        return true;
    } else {
        if (is_square(expr)) {
            for (size_t i = 0; i < etl::dim<0>(expr) - 1; ++i) {
                for (size_t j = i + 1; j < etl::dim<0>(expr); ++j) {
                    if (expr(i, j) != expr(j, i)) {
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }
}

/*!
 * \brief Indicates if the given expression is a lower triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a lower triangular matrix, false otherwise.
 */
template <typename E>
bool is_lower_triangular(E&& expr) {
    // lower_matrix<E> is already enforced to be lower triangular
    if constexpr (is_lower_matrix<E>) {
        return true;
    }
    // strictly_lower_matrix<E> is already enforced to be lower triangular
    else if constexpr (is_strictly_lower_matrix<E>) {
        return true;
    }
    // uni_lower_matrix<E> is already enforced to be lower triangular
    else if constexpr (is_uni_lower_matrix<E>) {
        return true;
    }
    // diagonal_matrix<E> is already enforced to be lower triangular
    else if constexpr (is_diagonal_matrix<E>) {
        return true;
    } else {
        if (is_square(expr)) {
            for (size_t i = 0; i < etl::dim<0>(expr) - 1; ++i) {
                for (size_t j = i + 1; j < etl::dim<0>(expr); ++j) {
                    if (expr(i, j) != 0.0) {
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }
}

/*!
 * \brief Indicates if the given expression is a uni lower triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a lower triangular matrix, false otherwise.
 */
template <typename E>
bool is_uni_lower_triangular(E&& expr) {
    // uni_lower_matrix<E> is already enforced to be uni lower triangular
    if constexpr (is_uni_lower_matrix<E>) {
        return true;
    } else {
        if (is_square(expr)) {
            for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
                if (expr(i, i) != 1.0) {
                    return false;
                }

                for (size_t j = i + 1; j < etl::dim<0>(expr); ++j) {
                    if (expr(i, j) != 0.0) {
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }
}

/*!
 * \brief Indicates if the given expression is a strictly lower triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a strictly lower triangular matrix, false otherwise.
 */
template <typename E>
bool is_strictly_lower_triangular(E&& expr) {
    // strictly_lower_matrix<E> is already enforced to be strictly lower triangular
    if constexpr (is_strictly_lower_matrix<E>) {
        return true;
    } else {
        if (is_square(expr)) {
            for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
                for (size_t j = i; j < etl::dim<0>(expr); ++j) {
                    if (expr(i, j) != 0.0) {
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }
}

/*!
 * \brief Indicates if the given expression is a upper triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a upper triangular matrix, false otherwise.
 */
template <typename E>
bool is_upper_triangular(E&& expr) {
    // upper_matrix<E> is already enforced to be upper triangular
    if constexpr (is_upper_matrix<E>) {
        return true;
    }
    // strictly_upper_matrix<E> is already enforced to be upper triangular
    else if constexpr (is_strictly_upper_matrix<E>) {
        return true;
    }
    // uni_upper_matrix<E> is already enforced to be upper triangular
    else if constexpr (is_uni_upper_matrix<E>) {
        return true;
    }
    // diagonal_matrix<E> is already enforced to be upper triangular
    else if constexpr (is_diagonal_matrix<E>) {
        return true;
    } else {
        if (is_square(expr)) {
            for (size_t i = 1; i < etl::dim<0>(expr); ++i) {
                for (size_t j = 0; j < i; ++j) {
                    if (expr(i, j) != 0.0) {
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }
}

/*!
 * \brief Indicates if the given expression is a strictly upper triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a strictly upper triangular matrix, false otherwise.
 */
template <typename E>
bool is_uni_upper_triangular(E&& expr) {
    // uni_upper_matrix<E> is already enforced to be uni upper triangular
    if constexpr (is_uni_upper_matrix<E>) {
        return true;
    } else {
        if (is_square(expr)) {
            for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
                if (expr(i, i) != 1.0) {
                    return false;
                }

                for (size_t j = 0; j < i; ++j) {
                    if (expr(i, j) != 0.0) {
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }
}

/*!
 * \brief Indicates if the given expression is a strictly upper triangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a strictly upper triangular matrix, false otherwise.
 */
template <typename E>
bool is_strictly_upper_triangular(E&& expr) {
    // strictly_upper_matrix<E> is already enforced to be strictly upper triangular
    if constexpr (is_strictly_upper_matrix<E>) {
        return true;
    } else {
        if (is_square(expr)) {
            for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    if (expr(i, j) != 0.0) {
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }
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
    // diagonal_matrix<E> is already enforced to be diagonal
    if constexpr (is_diagonal_matrix<E>) {
        return true;
    } else {
        if (is_square(expr)) {
            for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
                for (size_t j = 0; j < etl::dim<0>(expr); ++j) {
                    if (i != j && expr(i, j) != 0.0) {
                        return false;
                    }
                }
            }

            return true;
        }

        return false;
    }
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
bool is_permutation_matrix(E&& expr) {
    if (!is_square(expr)) {
        return false;
    }

    //Conditions:
    //a) Must be a square matrix
    //b) Every row must have one 1
    //c) Every column must have one 1

    for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
        auto sum = value_t<E>(0);
        for (size_t j = 0; j < etl::dim<0>(expr); ++j) {
            if (expr(i, j) != value_t<E>(0) && expr(i, j) != value_t<E>(1)) {
                return false;
            }

            sum += expr(i, j);
        }

        if (sum != value_t<E>(1)) {
            return false;
        }
    }

    for (size_t j = 0; j < etl::dim<0>(expr); ++j) {
        auto sum = value_t<E>(0);
        for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
            sum += expr(i, j);
        }

        if (sum != value_t<E>(1)) {
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
template <typename E>
bool is_hermitian(E&& expr) {
    if constexpr (is_complex<E>) {
        // hermitian_matrix<E> is already enforced to be hermitian
        if constexpr (is_hermitian_matrix<E>) {
            return true;
        } else {
            if (!is_square(expr)) {
                return false;
            }

            for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
                for (size_t j = 0; j < etl::dim<0>(expr); ++j) {
                    if (i != j && expr(i, j) != get_conj(expr(j, i))) {
                        return false;
                    }
                }
            }

            return true;
        }
    } else {
        return false;
    }
}

/*!
 * \brief Compare two ETL expressions for equality.
 *
 * \return true if the expressions contains the same sequence of values, false othwerise.
 */
template <typename L, typename R, cpp_enable_iff(all_etl_expr<L, R>)>
bool operator==(L&& lhs, R&& rhs) {
    // Both expressions must have the same number of dimensions
    if (etl::dimensions(lhs) != etl::dimensions(rhs)) {
        return false;
    }

    // The dimensions must be the same
    for (size_t i = 0; i < etl::dimensions(rhs); ++i) {
        if (etl::dim(lhs, i) != etl::dim(rhs, i)) {
            return false;
        }
    }

    // At this point, the values are necessary for the comparison
    force(lhs);
    force(rhs);

    // Note: Ideally, we should use std::equal, but this is significantly
    // faster to compile

    for (size_t i = 0; i < etl::size(lhs); ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }

    return true;
}

/*!
 * \brief Compare the expression with another expression for inequality.
 *
 * \return false if the expressions contains the same sequence of values, true othwerise.
 */
template <typename L, typename R, cpp_enable_iff(all_etl_expr<L, R>)>
bool operator!=(L&& lhs, R&& rhs) {
    return !(lhs == rhs);
}

/*!
 * \brief Test if two floating point numbers are approximately equals
 * \param a The first number of test
 * \param b The second number of test
 * \param epsilon The epsilon for comparison (0.0000001f is good default)
 * \return true if the two numbers are approximately equals, false otherwise
 *
 * The logic is taken from http://floating-point-gui.de/errors/comparison/ (Michael Borgwardt)
 */
template <typename T, typename TE = T>
inline bool approx_equals_float(T a, T b, TE epsilon) {
    using std::fabs;

    const auto abs_a    = fabs(a);
    const auto abs_b    = fabs(b);
    const auto abs_diff = fabs(a - b);

    // Note: min for floating points is the min normalized value
    static constexpr T min_normal = std::numeric_limits<T>::min();
    static constexpr T max        = std::numeric_limits<T>::max();

    if (a == b) {
        // This should handle infinities properly
        return true;
    } else if (a == 0 || b == 0 || abs_diff < min_normal) {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return abs_diff < epsilon;
    } else { // use relative error
        return (abs_diff / std::min(abs_a + abs_b, max)) < epsilon;
    }
}

/*!
 * \brief Test if two ETL expression are approximately equals
 * \param lhs The left hand-side
 * \param rhs The right hand-side
 * \param eps The epsilon for comparison
 * \return true if the two expression are aproximately equals, false othwerise
 */
template <typename L, typename E>
bool approx_equals(L&& lhs, E&& rhs, value_t<L> eps) {
    // Both expressions must have the same number of dimensions
    if constexpr (etl::dimensions<L>() != etl::dimensions<E>()) {
        return false;
    } else {
        // The dimensions must be the same
        for (size_t i = 0; i < etl::dimensions(rhs); ++i) {
            if (etl::dim(lhs, i) != etl::dim(rhs, i)) {
                return false;
            }
        }

        // At this point, the values are necessary for the comparison
        force(lhs);
        force(rhs);

        for (size_t i = 0; i < etl::size(lhs); ++i) {
            if (!approx_equals_float(lhs[i], rhs[i], eps)) {
                return false;
            }
        }

        return true;
    }
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

    for (size_t i = 0; i < etl::dim<0>(expr); ++i) {
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

/*!
 * \brief Decomposition the matrix so that A = Q * R
 * \param A The A matrix (mxn)
 * \param Q The Q matrix (Orthogonal mxm)
 * \param R The R matrix (Upper Triangular mxn)
 * \return true if the decomposition suceeded, false otherwise
 */
template <typename AT, typename QT, typename RT>
bool qr(AT& A, QT& Q, RT& R) {
    // A and R have the same dimensions
    if (etl::dim(A, 0) != etl::dim(R, 0) || etl::dim(A, 1) != etl::dim(R, 1)) {
        return false;
    }

    // A and Q have corresponding first dimensions and Q is square
    if (etl::dim(A, 0) != etl::dim(Q, 0) || etl::dim(A, 0) != etl::dim(Q, 1)) {
        return false;
    }

    detail::qr_impl::apply(A, Q, R);

    return true;
}

/*!
 * \brief Shuffle all the elements of an ETL vector or matrix (considered as
 * array).
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the shuffle operation is available.
 *
 * \param vector The vector to shuffle
 * \param g The generator to use for random number generation
 */
template <typename T, typename G>
void shuffle_flat(T& vector, G&& g) {
    const auto n = etl::size(vector);

    if (n < 2) {
        return;
    }

    if constexpr (impl::egblas::has_shuffle_seed) {
        std::uniform_int_distribution<size_t> seed_dist;

        vector.ensure_gpu_up_to_date();

        impl::egblas::shuffle_seed(n, vector.gpu_memory(), sizeof(etl::value_t<T>), seed_dist(g));

        vector.invalidate_cpu();
    } else {
        using distribution_t = typename std::uniform_int_distribution<size_t>;
        using param_t        = typename distribution_t::param_type;

        distribution_t dist;

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            using std::swap;
            swap(vector[i], vector[new_i]);
        }
    }
}

/*!
 * \brief Shuffle all the elements of an ETL vector or matrix (considered as
 * array)
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the shuffle operation is available.
 *
 * \param vector The vector to shuffle
 */
template <typename T>
void shuffle_flat(T& vector) {
    static std::random_device rd;
    static etl::random_engine g(rd());

    shuffle_flat(vector, g);
}

/*!
 * \brief Shuffle all the elements of a matrix.
 *
 * The elements will be shuffled according to the first dimension of
 * the matrix.
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the shuffle operation is available.
 *
 * \param matrix The matrix to shuffle
 * \param g The generator to use for random number generation
 */
template <typename T, typename G>
void shuffle_first(T& matrix, G&& g) {
    const auto n = etl::dim<0>(matrix);

    if (n < 2) {
        return;
    }

    if constexpr (impl::egblas::has_shuffle_seed) {
        std::uniform_int_distribution<size_t> seed_dist;

        matrix.ensure_gpu_up_to_date();

        impl::egblas::shuffle_seed(n, matrix.gpu_memory(), sizeof(etl::value_t<T>) * (etl::size(matrix) / n), seed_dist(g));

        matrix.invalidate_cpu();
    } else {
        using distribution_t = typename std::uniform_int_distribution<size_t>;
        using param_t        = typename distribution_t::param_type;

        distribution_t dist;

        auto temp = etl::force_temporary(matrix(0));

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            temp          = matrix(i);
            matrix(i)     = matrix(new_i);
            matrix(new_i) = temp;
        }
    }
}

/*!
 * \brief Shuffle all the elements of a matrix.
 *
 * The elements will be shuffled according to the first dimension of
 * the matrix.
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the shuffle operation is available.
 *
 * \param matrix The matrix to shuffle
 */
template <typename T>
void shuffle_first(T& matrix) {
    static std::random_device rd;
    static etl::random_engine g(rd());

    shuffle_first(matrix, g);
}

/*!
 * \brief Shuffle all the elements of an ETL vector
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the shuffle operation is available.
 *
 * \param vector The vector to shuffle
 */
template <typename T>
void shuffle(T& vector) {
    if constexpr (is_1d<T>) {
        shuffle_flat(vector);
    } else {
        shuffle_first(vector);
    }
}

/*!
 * \brief Shuffle all the elements of an ETL vector
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the shuffle operation is available.
 *
 * \param vector The vector to shuffle
 * \param g The generator to use for random number generation
 */
template <typename T, typename G>
void shuffle(T& vector, G&& g) {
    if constexpr (is_1d<T>) {
        shuffle_flat(vector, g);
    } else {
        shuffle_first(vector, g);
    }
}

/*!
 * \brief Shuffle all the elements of two vectors, using the same permutation
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the par_shuffle operation is available.
 *
 * \param v1 The first vector to shuffle
 * \param v2 The second vector to shuffle
 */
template <typename T1, typename T2, typename G>
void parallel_shuffle_flat(T1& v1, T2& v2, G&& g) {
    static_assert(decay_traits<T1>::dimensions() == decay_traits<T2>::dimensions(), "Impossible to shuffle vector of different dimensions");

    cpp_assert(etl::size(v1) == etl::size(v2), "Impossible to shuffle vector of different dimensions");

    const auto n = etl::size(v1);

    if (n < 2) {
        return;
    }

    if constexpr (impl::egblas::has_par_shuffle_seed) {
        std::uniform_int_distribution<size_t> seed_dist;

        v1.ensure_gpu_up_to_date();
        v2.ensure_gpu_up_to_date();

        impl::egblas::par_shuffle_seed(n, v1.gpu_memory(), sizeof(etl::value_t<T1>), v2.gpu_memory(), sizeof(etl::value_t<T2>), seed_dist(g));

        v1.invalidate_cpu();
        v2.invalidate_cpu();
    } else {
        using distribution_t = typename std::uniform_int_distribution<size_t>;
        using param_t        = typename distribution_t::param_type;

        distribution_t dist;

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            using std::swap;
            swap(v1[i], v1[new_i]);
            swap(v2[i], v2[new_i]);
        }
    }
}

/*!
 * \brief Shuffle all the elements of two vectors, using the same permutation
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the par_shuffle operation is available.
 *
 * \param v1 The first vector to shuffle
 * \param v2 The second vector to shuffle
 */
template <typename T1, typename T2>
void parallel_shuffle_flat(T1& v1, T2& v2) {
    static std::random_device rd;
    static etl::random_engine g(rd());

    parallel_shuffle_flat(v1, v2, g);
}

/*!
 * \brief Shuffle all the elements of two matrices, using the same permutation.
 *
 * The elements will be shuffled according to the first dimension of
 * the matrix.
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the par_shuffle operation is available.
 *
 * \param m1 The first matrix to shuffle
 * \param m2 The first matrix to shuffle
 */
template <typename T1, typename T2, typename G>
void parallel_shuffle_first(T1& m1, T2& m2, G&& g) {
    cpp_assert(etl::dim<0>(m1) == etl::dim<0>(m2), "Impossible to shuffle together matrices of different first dimension");

    const auto n = etl::dim<0>(m1);

    if (n < 2) {
        return;
    }

    if constexpr (impl::egblas::has_par_shuffle_seed) {
        std::uniform_int_distribution<size_t> seed_dist;

        m1.ensure_gpu_up_to_date();
        m2.ensure_gpu_up_to_date();

        impl::egblas::par_shuffle_seed(n, m1.gpu_memory(), sizeof(etl::value_t<T1>) * (etl::size(m1) / n), m2.gpu_memory(),
                                       sizeof(etl::value_t<T2>) * (etl::size(m2) / n), seed_dist(g));

        m1.invalidate_cpu();
        m2.invalidate_cpu();
    } else {
        using distribution_t = typename std::uniform_int_distribution<size_t>;
        using param_t        = typename distribution_t::param_type;

        distribution_t dist;

        auto t1 = etl::force_temporary(m1(0));
        auto t2 = etl::force_temporary(m2(0));

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            t1        = m1(i);
            m1(i)     = m1(new_i);
            m1(new_i) = t1;

            t2        = m2(i);
            m2(i)     = m2(new_i);
            m2(new_i) = t2;
        }
    }
}

/*!
 * \brief Shuffle all the elements of two matrices, using the same permutation.
 *
 * The elements will be shuffled according to the first dimension of
 * the matrix.
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the par_shuffle operation is available.
 *
 * \param m1 The first matrix to shuffle
 * \param m2 The first matrix to shuffle
 */
template <typename T1, typename T2>
void parallel_shuffle_first(T1& m1, T2& m2) {
    static std::random_device rd;
    static etl::random_engine g(rd());

    parallel_shuffle_first(m1, m2, g);
}

/*!
 * \brief Shuffle all the elements of two vectors or matrices, using the same permutation
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the par_shuffle_seed operation is available.
 *
 * \param v1 The first vector or matrix to shuffle
 * \param v2 The second vector or matrix to shuffle
 */
template <typename T1, typename T2>
void parallel_shuffle(T1& v1, T2& v2) {
    static std::random_device rd;
    static etl::random_engine g(rd());

    parallel_shuffle(v1, v2, g);
}

template <typename T>
void shuffle_swap(T& v1, size_t i, size_t new_i) {
    if constexpr (is_1d<T>) {
        auto t    = v1(i);
        v1(i)     = v1(new_i);
        v1(new_i) = t;
    } else {
        auto s1 = v1(i);
        auto s2 = v1(new_i);

        for (size_t index = 0 ; index < etl::subsize(v1); ++index) {
            auto t    = s1[index];
            s1[index] = s2[index];
            s2[index] = t;
        }
    }
}

/*!
 * \brief Shuffle all the elements of two vectors or matrices, using the same permutation
 *
 * Note: This function will be executed on GPU if EGBLAS is
 * configured and the par_shuffle_seed operation is available.
 *
 * \param v1 The first vector or matrix to shuffle
 * \param v2 The second vector or matrix to shuffle
 */
template <typename T1, typename T2, typename G>
void parallel_shuffle(T1& v1, T2& v2, G&& g) {
    cpp_assert(etl::dim<0>(v1) == etl::dim<0>(v2), "Impossible to shuffle together matrices of different first dimension");

    const auto n = etl::dim<0>(v1);

    if (n < 2) {
        return;
    }

    if constexpr (impl::egblas::has_par_shuffle_seed) {
        std::uniform_int_distribution<size_t> seed_dist;

        v1.ensure_gpu_up_to_date();
        v2.ensure_gpu_up_to_date();

        impl::egblas::par_shuffle_seed(n, v1.gpu_memory(), sizeof(etl::value_t<T1>) * (etl::size(v1) / n), v2.gpu_memory(),
                                       sizeof(etl::value_t<T2>) * (etl::size(v2) / n), seed_dist(g));

        v1.invalidate_cpu();
        v2.invalidate_cpu();
    } else {
        using distribution_t = typename std::uniform_int_distribution<size_t>;
        using param_t        = typename distribution_t::param_type;

        distribution_t dist;

        // Note: We must handle non-homogeneous matrices here
        // For instance, a matrix and a vector hence, the code in shuffle_swap

        for (auto i = n - 1; i > 0; --i) {
            auto new_i = dist(g, param_t(0, i));

            shuffle_swap(v1, i, new_i);
            shuffle_swap(v2, i, new_i);
        }
    }
}

/*!
 * \brief Merge sub inside merged at the given position
 *
 * \param merged The merged result
 * \param sub The data to merge inside merged
 * \param index The position at which to merge
 *
 * \return merged
 */
template <typename M, typename N>
M& merge(M& merged, const N& sub, size_t index) {
    const size_t s = etl::size(sub);
    const size_t n = index * s;

    memory_slice(merged, n, n + s) = sub;

    return merged;
}

/*!
 * \brief Merge sub inside merged at the given position, for each batch.
 *
 * \param merged The merged result
 * \param sub The data to merge inside merged
 * \param index The position at which to merge
 *
 * \return merged
 */
template <typename M, typename N>
M& batch_merge(M& merged, const N& sub, size_t index) {
    const size_t s = etl::size(sub(0));
    const size_t n = index * s;

    for (size_t b = 0; b < etl::dim<0>(merged); ++b) {
        memory_slice(merged(b), n, n + s) = sub(b);
    }

    return merged;
}

/*!
 * \brief Dispatch a part of merged to dispatched from the given position
 *
 * \param dispatched The dispatched result
 * \param merged The data to dispatch inside dispatched
 * \param index The position from which to dispatch
 *
 * \return dispatched
 */
template <typename M, typename N>
M& dispatch(M& dispatched, const N& merged, size_t index) {
    const size_t s = etl::size(dispatched);
    const size_t n = index * s;

    dispatched = memory_slice(merged, n, n + s);

    return dispatched;
}

/*!
 * \brief Dispatch a part of merged to dispatched from the given position, for each batch
 *
 * \param dispatched The dispatched result
 * \param merged The data to dispatch inside dispatched
 * \param index The position from which to dispatch
 *
 * \return dispatched
 */
template <typename M, typename N>
M& batch_dispatch(M& dispatched, const N& merged, size_t index) {
    const size_t s = etl::size(dispatched(0));
    const size_t n = index * s;

    for (size_t b = 0; b < etl::dim<0>(merged); ++b) {
        dispatched(b) = memory_slice(merged(b), n, n + s);
    }

    return dispatched;
}

/*!
 * \brief Binarize the given ETL contrainer.
 * \param matrix The container binarize
 * \param b The binarization threshold.
 */
template <typename M, typename T>
void binarize(M& matrix, T b) {
    using VT = value_t<M>;

    for (auto& value : matrix) {
        value = value > b ? VT(1) : VT(0);
    }
}

/*!
 * \brief Normalize the given ETL contrainer to zero-mean and unit-variance.
 * \param matrix The container to normalize
 */
template <typename M>
void normalize_flat(M& matrix) {
    using VT = value_t<M>;

    auto m = mean(matrix);

    matrix = matrix - m;

    auto s = stddev(matrix, 0.0);

    if (s != VT(0)) {
        matrix = matrix / s;
    }
}

/*!
 * \brief Normalize each sub container of the given ETL contrainer
 * to zero-mean and unit-variance.
 *
 * \param matrix The container to normalize
 */
template <typename M>
void normalize_sub(M& matrix) {
    using VT = value_t<M>;

#ifdef ETL_CUDA
    // The case of CUDA must be handled differently to avoid going
    // back and forth between both memories
    // While this could simply use egblas_normalize_sub, this would
    // be much slower

    auto mm          = matrix.memory_start();
    const auto n     = etl::dim<0>(matrix);
    const auto sub_n = etl::size(matrix) / n;

    for (size_t i = 0; i < n; ++i) {
        VT m(0);

        for (size_t j = 0; j < sub_n; ++j) {
            m += mm[i * sub_n + j];
        }

        m /= VT(sub_n);

        VT s(0);

        for (size_t j = 0; j < sub_n; ++j) {
            s += (mm[i * sub_n + j] - m) * (mm[i * sub_n + j] - m);
        }

        s = std::sqrt(s / VT(sub_n));

        for (size_t j = 0; j < sub_n; ++j) {
            mm[i * sub_n + j] = (mm[i * sub_n + j] - m) / s;
        }
    }
#else
    for (size_t i = 0; i < etl::dim<0>(matrix); ++i) {
        auto m = mean(matrix(i));

        matrix(i) = matrix(i) - m;

        auto s = stddev(matrix(i), 0.0);

        if (s != VT(0)) {
            matrix(i) = matrix(i) / s;
        }
    }
#endif
}

} //end of namespace etl
