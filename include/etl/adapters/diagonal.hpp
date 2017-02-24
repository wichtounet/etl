//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains diagonal matrix implementation
 */

#pragma once

#include "etl/adapters/adapter.hpp"            // The adapter base class
#include "etl/adapters/diagonal_exception.hpp" // The diagonal exception
#include "etl/adapters/diagonal_reference.hpp" // The reference proxy

namespace etl {

/*!
 * \brief A diagonal matrix adapter.
 *
 * This is only a prototype.
 */
template <typename Matrix>
struct diagonal_matrix final : adapter<Matrix>, iterable<const diagonal_matrix<Matrix>> {
    using matrix_t = Matrix;   ///< The adapted matrix type
    using expr_t   = matrix_t; ///< The wrapped expression type

    static_assert(etl_traits<matrix_t>::is_value, "Diagonal matrix only works with value classes");
    static_assert(etl_traits<matrix_t>::dimensions() == 2, "Diagonal matrix must be two-dimensional");
    static_assert(is_square_matrix<matrix_t>::value, "Diagonal matrix must be square");

    static constexpr std::size_t n_dimensions = etl_traits<matrix_t>::dimensions();  ///< The number of dimensions
    static constexpr order storage_order      = etl_traits<matrix_t>::storage_order; ///< The storage order
    static constexpr std::size_t alignment    = matrix_t::alignment;                 ///< The memory alignment

    using value_type        = value_t<matrix_t>;                 ///< The value type
    using memory_type       = value_type*;                       ///< The memory type
    using const_memory_type = const value_type*;                 ///< The const memory type

    using iterator       = typename matrix_t::const_iterator; ///< The type of const iterator
    using const_iterator = typename matrix_t::const_iterator; ///< The type of const iterator

    using base_type = adapter<Matrix>; ///< The base type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<value_type>;

    using base_type::matrix;

public:
    /*!
     * \brief Construct a new diagonal matrix and fill it with zeros
     *
     * This constructor can only be used when the matrix is fast
     */
    diagonal_matrix() noexcept : base_type() {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new diagonal matrix and fill it witht the given value
     *
     * \param value The value to fill the matrix with
     *
     * This constructor can only be used when the matrix is fast
     */
    diagonal_matrix(value_type value) noexcept : base_type(value) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new diagonal matrix and fill it with zeros
     * \param dim The dimension of the matrix
     */
    diagonal_matrix(std::size_t dim) noexcept : base_type(dim) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a diagonal_matrix by copy
     * \param rhs The right-hand-side matrix
     */
    diagonal_matrix(const diagonal_matrix& rhs) = default;

    /*!
     * \brief Assign to the matrix by copy
     * \param rhs The right-hand-side matrix
     * \return a reference to the assigned matrix
     */
    diagonal_matrix& operator=(const diagonal_matrix& rhs) = default;

    /*!
     * \brief Construct a diagonal_matrix by move
     * \param rhs The right-hand-side matrix
     */
    diagonal_matrix(diagonal_matrix&& rhs) = default;

    /*!
     * \brief Assign to the matrix by move
     * \param rhs The right-hand-side matrix
     * \return a reference to the assigned matrix
     */
    diagonal_matrix& operator=(diagonal_matrix&& rhs) = default;

    /*!
     * \brief Assign the values of the ETL expression to the diagonal matrix
     * \param e The ETL expression to get the values from
     * \return a reference to the fast matrix
     */
    template <typename E, cpp_enable_if(std::is_convertible<value_t<E>, value_type>::value, is_etl_expr<E>::value)>
    diagonal_matrix& operator=(E&& e) noexcept(false) {
        // Make sure the other matrix is diagonal
        if(!is_diagonal(e)){
            throw diagonal_exception();
        }

        // Perform the real assign

        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);

        return *this;
    }

    /*!
     * \brief Multiply each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    diagonal_matrix& operator+=(const value_type& rhs) noexcept {
        detail::scalar_add::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    diagonal_matrix& operator+=(const R& rhs){
        // Make sure the other matrix is diagonal
        if(!is_diagonal(rhs)){
            throw diagonal_exception();
        }

        validate_expression(*this, rhs);
        add_evaluate(rhs, *this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    diagonal_matrix& operator-=(const value_type& rhs) noexcept {
        detail::scalar_sub::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    diagonal_matrix& operator-=(const R& rhs){
        // Make sure the other matrix is diagonal
        if(!is_diagonal(rhs)){
            throw diagonal_exception();
        }

        validate_expression(*this, rhs);
        sub_evaluate(rhs, *this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    diagonal_matrix& operator*=(const value_type& rhs) noexcept {
        detail::scalar_mul::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    diagonal_matrix& operator*=(const R& rhs) {
        // Make sure the other matrix is diagonal
        if(!is_diagonal(rhs)){
            throw diagonal_exception();
        }

        validate_expression(*this, rhs);
        mul_evaluate(rhs, *this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    diagonal_matrix& operator>>=(const value_type& rhs) noexcept {
        detail::scalar_mul::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    diagonal_matrix& operator>>=(const R& rhs) {
        // Make sure the other matrix is diagonal
        if(!is_diagonal(rhs)){
            throw diagonal_exception();
        }

        validate_expression(*this, rhs);
        mul_evaluate(rhs, *this);
        return *this;
    }

    /*!
     * \brief Divide each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    diagonal_matrix& operator/=(const value_type& rhs) noexcept {
        detail::scalar_div::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Modulo each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    diagonal_matrix& operator/=(const R& rhs) {
        // Make sure the other matrix is diagonal
        if(!is_diagonal(rhs)){
            throw diagonal_exception();
        }

        validate_expression(*this, rhs);
        div_evaluate(rhs, *this);
        return *this;
    }

    /*!
     * \brief Modulo each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    diagonal_matrix& operator%=(const value_type& rhs) noexcept {
        detail::scalar_mod::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Modulo each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    diagonal_matrix& operator%=(const R& rhs){
        // Make sure the other matrix is diagonal
        if(!is_diagonal(rhs)){
            throw diagonal_exception();
        }

        validate_expression(*this, rhs);
        mod_evaluate(rhs, *this);
        return *this;
    }

    /*!
     * \brief Access the (i, j) element of the 2D matrix
     * \param i The index of the first dimension
     * \param j The index of the second dimension
     * \return a reference to the (i,j) element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    diagonal_detail::diagonal_reference<matrix_t> operator()(std::size_t i, std::size_t j) noexcept {
        return {matrix, i, j};
    }

    using base_type::operator();
};

/*!
 * \brief Traits specialization for diagonal_matrix
 */
template <typename Matrix>
struct etl_traits<diagonal_matrix<Matrix>> : wrapper_traits<diagonal_matrix<Matrix>> {};

} //end of namespace etl
