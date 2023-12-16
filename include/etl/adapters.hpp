//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains some global functions for adapters.
 */

#pragma once

namespace etl {

/*!
 * \brief Traits indicating if the given ETL type is a symmetric matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_symmetric_matrix = cpp::specialization_of<etl::symmetric_matrix, T>;

/*!
 * \brief Traits indicating if the given ETL type is a hermitian matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_hermitian_matrix = cpp::specialization_of<etl::hermitian_matrix, T>;

/*!
 * \brief Traits indicating if the given ETL type is a diagonal matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_diagonal_matrix = cpp::specialization_of<etl::diagonal_matrix, T>;

/*!
 * \brief Traits indicating if the given ETL type is an upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_upper_matrix = cpp::specialization_of<etl::upper_matrix, T>;

/*!
 * \brief Traits indicating if the given ETL type is a lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_lower_matrix = cpp::specialization_of<etl::lower_matrix, T>;

/*!
 * \brief Traits indicating if the given ETL type is a strictly lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_strictly_lower_matrix = cpp::specialization_of<etl::strictly_lower_matrix, T>;

/*!
 * \brief Traits indicating if the given ETL type is a strictly upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_strictly_upper_matrix = cpp::specialization_of<etl::strictly_upper_matrix, T>;

/*!
 * \brief Traits indicating if the given ETL type is a uni lower triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_uni_lower_matrix = cpp::specialization_of<etl::uni_lower_matrix, T>;

/*!
 * \brief Traits indicating if the given ETL type is a uni upper triangular matrix
 * \tparam T The type to test
 */
template <typename T>
constexpr bool is_uni_upper_matrix = cpp::specialization_of<etl::uni_upper_matrix, T>;


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

} //end of namespace etl
