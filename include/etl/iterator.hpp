//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Iterator implementation for ETL expressions.
 */

#pragma once

#include <iterator>

namespace etl {

/*!
 * \brief Configurable iterator for ETL expressions
 * \tparam Expr The type of expr for which the iterator is working
 */
template <typename Expr>
struct iterator : public std::iterator<std::random_access_iterator_tag, value_t<Expr>> {
private:
    Expr* expr; ///< Pointer to the expression
    size_t i;   ///< Current index

public:
    using base_iterator_t = std::iterator<std::random_access_iterator_tag, value_t<Expr>>; ///< The base iterator type
    using value_type      = value_t<Expr>;                                                 ///< The value type
    using reference_t     = decltype(std::declval<Expr>()[i]);                             ///< The type of reference
    using pointer_t       = std::add_pointer_t<decltype(std::declval<Expr>()[i])>;         ///< The type of pointer
    using difference_t    = typename base_iterator_t::difference_type;                     ///< The type used for subtracting two iterators

    /*!
     * \brief Construct a new iterator
     * \param expr The expr to iterate over.
     * \param i The starting position
     */
    iterator(Expr& expr, size_t i) : expr(&expr), i(i) {}

    /*!
     * \brief Dereference the iterator to get the current value
     * \return a reference to the current element
     */
    reference_t operator*() {
        return (*expr)[i];
    }

    /*!
     * \brief Dereferences the iterator at n forward position
     * \param n The number of forward position to advance
     * \return a reference to the element at the current position plus n
     */
    reference_t operator[](difference_t n) {
        return (*expr)[i + n];
    }

    /*!
     * \brief Dereference the iterator to get the current value
     * \return a pointer to the current element
     */
    auto operator-> () {
        return &(*expr)[i];
    }

    /*!
     * \brief Predecrement the iterator
     * \return a reference to the iterator
     */
    iterator& operator--() {
        --i;
        return *this;
    }

    /*!
     * \brief Postdecrement the iterator
     * \return an iterator the position prior to the decrement.
     */
    iterator operator--(int) {
        iterator prev(*this);
        --i;
        return prev;
    }

    /*!
     * \brief Preincrement the iterator
     * \return a reference to the iterator
     */
    iterator& operator++() {
        ++i;
        return *this;
    }

    /*!
     * \brief Postincrement the iterator
     * \return an iterator the position prior to the increment.
     */
    iterator operator++(int) {
        iterator prev(*this);
        ++i;
        return prev;
    }

    /*!
     * \brief Advances the iterator n positions
     * \param n The number of position to advance
     * \return a reference to the iterator
     */
    iterator& operator+=(difference_t n) {
        i += n;
        return *this;
    }

    /*!
     * \brief Back away the iterator n positions
     * \param n The number of position to back
     * \return a reference to the iterator
     */
    iterator& operator-=(difference_t n) {
        i -= n;
        return *this;
    }

    /*!
     * \brief Creates a new iterator poiting to the current position plus n
     * \param n The number of position to advance
     * \return the new interator
     */
    iterator operator+(difference_t n) {
        iterator it(*this);
        it += n;
        return it;
    }

    /*!
     * \brief Creates a new iterator poiting to the current position minus n
     * \param n The number of position to back away
     * \return the new interator
     */
    iterator operator-(difference_t n) {
        iterator it(*this);
        it -= n;
        return it;
    }

    /*!
     * \brief Computes the difference between two iterators
     * \param it the other iterator
     * \return the number of positions between the two iterators
     */
    difference_t operator-(const iterator& it) {
        return i - it.i;
    }

    /*!
     * \brief Compare two iterators
     * \param other The other iterator
     * \return true if this operator is equal to the other iterator
     */
    bool operator==(const iterator& other) const {
        return expr == other.expr && i == other.i;
    }

    /*!
     * \brief Compare two iterators
     * \param other The other iterator
     * \return true if this operator is not equal to the other iterator
     */
    bool operator!=(const iterator& other) const {
        return !(*this == other);
    }

    /*!
     * \brief Compare two iterators
     * \param other The other iterator
     * \return true if this operator is greater than the other iterator
     */
    bool operator>(const iterator& other) const {
        return i > other.i;
    }

    /*!
     * \brief Compare two iterators
     * \param other The other iterator
     * \return true if this operator is greater than or equal to the other iterator
     */
    bool operator>=(const iterator& other) const {
        return i >= other.i;
    }

    /*!
     * \brief Compare two iterators
     * \param other The other iterator
     * \return true if this operator is less than the other iterator
     */
    bool operator<(const iterator& other) const {
        return i < other.i;
    }

    /*!
     * \brief Compare two iterators
     * \param other The other iterator
     * \return true if this operator is less than or equal to the other iterator
     */
    bool operator<=(const iterator& other) const {
        return i <= other.i;
    }
};

} //end of namespace etl
