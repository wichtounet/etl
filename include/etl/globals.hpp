//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains some global functions.
*/

#pragma once

namespace etl {

/*!
 * \brief Indicates if the given expression is a square matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a square matrix, false otherwise.
 */
template<typename E>
bool is_square(E&& expr) {
    return decay_traits<E>::dimensions() == 2 && etl::dim<0>(expr) == etl::dim<1>(expr);
}

/*!
 * \brief Indicates if the given expression is a rectangular matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a rectangular matrix, false otherwise.
 */
template<typename E>
bool is_rectangular(E&& expr) {
    return decay_traits<E>::dimensions() == 2 && etl::dim<0>(expr) != etl::dim<1>(expr);
}

/*!
 * \brief Indicates if the given expression contains sub matrices that are square.
 * \param expr The expression to test
 * \return true if the given expression contains sub matrices that are square, false otherwise.
 */
template<typename E>
bool is_sub_square(E&& expr) {
    return decay_traits<E>::dimensions() == 3 && etl::dim<1>(expr) == etl::dim<2>(expr);
}

/*!
 * \brief Indicates if the given expression contains sub matrices that are rectangular.
 * \param expr The expression to test
 * \return true if the given expression contains sub matrices that are rectangular, false otherwise.
 */
template<typename E>
bool is_sub_rectangular(E&& expr) {
    return decay_traits<E>::dimensions() == 3 && etl::dim<1>(expr) != etl::dim<2>(expr);
}

/*!
 * \brief Indicates if the given expression is a symmetric matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a symmetric matrix, false otherwise.
 */
template<typename E>
bool is_symmetric(E&& expr) {
    if(is_square(expr)){
        for(std::size_t i = 0; i < etl::dim<0>(expr) - 1; ++i){
            for(std::size_t j = i + 1; j < etl::dim<0>(expr); ++j){
                if(expr(i, j) != expr(j, i)){
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
template<typename E>
bool is_lower_triangular(E&& expr) {
    if(is_square(expr)){
        for(std::size_t i = 0; i < etl::dim<0>(expr) - 1; ++i){
            for(std::size_t j = i + 1; j < etl::dim<0>(expr); ++j){
                if(expr(i, j) != 0.0){
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
template<typename E>
bool is_strictly_lower_triangular(E&& expr) {
    if(is_square(expr)){
        for(std::size_t i = 0; i < etl::dim<0>(expr); ++i){
            for(std::size_t j = i; j < etl::dim<0>(expr); ++j){
                if(expr(i, j) != 0.0){
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
template<typename E>
bool is_upper_triangular(E&& expr) {
    if(is_square(expr)){
        for(std::size_t i = 1; i < etl::dim<0>(expr); ++i){
            for(std::size_t j = 0; j < i; ++j){
                if(expr(i, j) != 0.0){
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
template<typename E>
bool is_strictly_upper_triangular(E&& expr) {
    if(is_square(expr)){
        for(std::size_t i = 0; i < etl::dim<0>(expr); ++i){
            for(std::size_t j = 0; j <= i; ++j){
                if(expr(i, j) != 0.0){
                    return false;
                }
            }
        }

        return true;
    }

    return false;
}

/*!
 * \brief Indicates if the given expression is a diagonal matrix or not.
 * \param expr The expression to test
 * \return true if the given expression is a diagonal matrix, false otherwise.
 */
template<typename E>
bool is_diagonal(E&& expr) {
    if(is_square(expr)){
        for(std::size_t i = 0; i < etl::dim<0>(expr); ++i){
            for(std::size_t j = 0; j < etl::dim<0>(expr); ++j){
                if(i != j && expr(i, j) != 0.0){
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
template<typename E>
bool is_uniform(E&& expr){
    if(!etl::size(expr)){
        return false;
    }

    auto first = *expr.begin();

    for(auto v : expr){
        if(v != first){
            return false;
        }
    }

    return true;
}

/*!
 * \brief Returns the trace of the given square matrix.
 *
 * If the given expression does not represent a square matrix, this function will fail
 *
 * \param expr The expression to get the trace from.
 * \return The trace of the given expression
 */
template<typename E>
value_t<E> trace(E&& expr){
    assert_square(expr);

    auto value = value_t<E>();

    for(std::size_t i = 0; i < etl::dim<0>(expr); ++i){
        value += expr(i, i);
    }

    return value;
}

} //end of namespace etl
