//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains runtime matrix implementation
 */

#pragma once

#include "etl/dyn_base.hpp"    //The base class and utilities

namespace etl {

namespace sym_detail {

template <typename Matrix, typename Enable = void>
struct static_check_square {};

template <typename Matrix>
struct static_check_square<Matrix, std::enable_if_t<all_fast<Matrix>::value && is_2d<Matrix>::value>> {
    static_assert(etl_traits<Matrix>::template dim<0>() == etl_traits<Matrix>::template dim<1>(), "Static matrix must be square");
};

/*!
 * \brief A proxy representing a reference to a mutable element of a symmetric matrix
 * \tparam M The matrix type
 */
template <typename M>
struct symmetric_reference {
    using matrix_type              = M;                                ///< The matrix type
    using value_type               = typename matrix_type::value_type; ///< The value type
    using raw_pointer_type         = value_type*;                      ///< A raw pointer type
    using raw_reference_type       = value_type&;                      ///< A raw reference type
    using const_raw_reference_type = std::add_const_t<value_type>&;    ///< A raw reference type

    matrix_type& matrix;   ///< Reference to the matrix
    std::size_t i;         ///< The first index
    std::size_t j;         ///< The second index
    value_type& value;     ///< Reference to the value
    value_type& sym_value; ///< Reference to the symmetric value

    /*!
     * \brief Constructs a new symmetric_reference
     * \param matrix The source matrix
     * \param i The index i of the first dimension
     * \param j The index j of the second dimension
     */
    symmetric_reference(matrix_type& matrix, std::size_t i, std::size_t j)
            : matrix(matrix), i(i), j(j), value(matrix(i, j)), sym_value(matrix(j, i)) {
        //Nothing else to init
    }

    /*!
     * \brief Sets a new value to the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator=(const value_type& rhs) {
        value = rhs;
        if(i != j){
            sym_value = rhs;
        }
        return *this;
    }

    /*!
     * \brief Adds a new value to the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator+=(value_type rhs) {
        value += rhs;
        if(i != j){
            sym_value += rhs;
        }
        return *this;
    }

    /*!
     * \brief Subtract a new value from the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator-=(value_type rhs) {
        value -= rhs;
        if(i != j){
            sym_value -= rhs;
        }
        return *this;
    }

    /*!
     * \brief Multiply by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator*=(value_type rhs) {
        value *= rhs;
        if(i != j){
            sym_value *= rhs;
        }
        return *this;
    }

    /*!
     * \brief Divide by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator/=(value_type rhs) {
        value /= rhs;
        if(i != j){
            sym_value /= rhs;
        }
        return *this;
    }

    /*!
     * \brief Modulo by a new value the proxy reference
     * \param rhs The new value
     * \return a reference to the proxy reference
     */
    symmetric_reference& operator%=(value_type rhs) {
        value %= rhs;
        if(i != j){
            sym_value %= rhs;
        }
        return *this;
    }

    /*!
     * \brief Casts the proxy reference to the raw reference type
     * \return a raw reference to the element
     */
    operator const_raw_reference_type&() const {
        return value;
    }
};

} //end of namespace sym_detail

/*!
 * \brief A symmetric matrix adapter.
 *
 * This is only a prototype.
 */
template <typename Matrix>
struct sym_matrix final {
    using matrix_t = Matrix;   ///< The adapted matrix type
    using expr_t   = matrix_t; ///< The wrapped expression type

    static_assert(etl_traits<matrix_t>::is_value, "Symmetric matrix only works with value classes");
    static_assert(etl_traits<matrix_t>::dimensions() == 2, "Symmetric matrix must be two-dimensional");
    using scs = sym_detail::static_check_square<matrix_t>; ///< static_check trick

    static constexpr const std::size_t n_dimensions = etl_traits<matrix_t>::dimensions();    ///< The number of dimensions
    static constexpr const order storage_order      = etl_traits<matrix_t>::storage_order;   ///< The storage order
    static constexpr const std::size_t alignment    = intrinsic_traits<matrix_t>::alignment; ///< The memory alignment

    using value_type        = value_t<matrix_t>; ///< The value type
    using memory_type       = value_type*;       ///< The memory type
    using const_memory_type = const value_type*; ///< The const memory type
    using iterator          = memory_type;       ///< The type of iterator
    using const_iterator    = const_memory_type; ///< The type of const iterator

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_t>;

private:
    matrix_t matrix; ///< The adapted matrix

public:
    sym_matrix() noexcept : matrix(value_type()) {
        //Nothing else to init
    }

    sym_matrix(value_type value) noexcept : matrix(value) {
        //Nothing else to init
    }

    sym_matrix(std::size_t dim) noexcept : matrix(dim, dim, value_type()) {
        //Nothing else to init
    }

    sym_matrix(std::size_t dim, value_type value) noexcept : matrix(dim, dim, value) {
        //Nothing else to init
    }

    sym_matrix(const sym_matrix& rhs) = default;
    sym_matrix& operator=(const sym_matrix& rhs) = default;

    sym_matrix(sym_matrix&& rhs) = default;
    sym_matrix& operator=(sym_matrix&& rhs) = default;

    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr std::size_t dimensions() noexcept {
        return 2;
    }

    /*!
     * \brief Access the (i, j) element of the 2D matrix
     * \param i The index of the first dimension
     * \param j The index of the second dimension
     * \return a reference to the (i,j) element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    sym_detail::symmetric_reference<matrix_t> operator()(std::size_t i, std::size_t j) noexcept {
        return {matrix, i, j};
    }

    /*!
     * \brief Access the (i, j) element of the 2D matrix
     * \param i The index of the first dimension
     * \param j The index of the second dimension
     * \return a reference to the (i,j) element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    const value_type& operator()(std::size_t i, std::size_t j) const noexcept {
        return matrix(i, j);
    }

    /*!
     * \returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const noexcept {
        return matrix.read_flat(i);
    }

    const expr_t& value() const noexcept {
        return matrix;
    }
};

template <typename Matrix>
struct etl_traits<sym_matrix<Matrix>> : wrapper_traits<sym_matrix<Matrix>> {};

} //end of namespace etl
