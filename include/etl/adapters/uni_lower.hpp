//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains uni lower triangular matrix implementation
 */

#pragma once

#include "etl/adapters/adapter.hpp"             // The adapter base class
#include "etl/adapters/uni_lower_exception.hpp" // The exception
#include "etl/adapters/uni_lower_reference.hpp" // The reference proxy

namespace etl {

/*!
 * \brief A uni lower triangular matrix adapter.
 *
 * This is only a prototype.
 */
template <adaptable Matrix>
struct uni_lower_matrix final : adapter<Matrix>, iterable<const uni_lower_matrix<Matrix>> {
    using matrix_t  = Matrix;                   ///< The adapted matrix type
    using expr_t    = matrix_t;                 ///< The wrapped expression type
    using this_type = uni_lower_matrix<Matrix>; ///< The type of this matrix

    static constexpr size_t n_dimensions = etl_traits<matrix_t>::dimensions();  ///< The number of dimensions
    static constexpr order storage_order = etl_traits<matrix_t>::storage_order; ///< The storage order
    static constexpr size_t alignment    = matrix_t::alignment;                 ///< The memory alignment

    using value_type        = value_t<matrix_t>; ///< The value type
    using memory_type       = value_type*;       ///< The memory type
    using const_memory_type = const value_type*; ///< The const memory type

    using iterator       = typename matrix_t::const_iterator; ///< The type of const iterator
    using const_iterator = typename matrix_t::const_iterator; ///< The type of const iterator

    using base_type = adapter<Matrix>; ///< The base type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<value_type>;

    using base_type::value;

public:
    /*!
     * \brief Construct a new uni lower triangular matrix and fill it with zeros
     *
     * This constructor can only be used when the matrix is fast
     */
    uni_lower_matrix() noexcept : base_type() {
        // Fill the diagonal
        for (size_t i = 0; i < etl::dim<0>(value); ++i) {
            value(i, i) = value_type(1);
        }
    }

    /*!
     * \brief Construct a new uni lower triangular matrix and fill it with zeros
     * \param dim The dimension of the matrix
     */
    explicit uni_lower_matrix(size_t dim) noexcept : base_type(dim) {
        // Fill the diagonal
        for (size_t i = 0; i < etl::dim<0>(value); ++i) {
            value(i, i) = value_type(1);
        }
    }

    /*!
     * \brief Construct a uni_lower_matrix by copy
     * \param rhs The right-hand-side matrix
     */
    uni_lower_matrix(const uni_lower_matrix& rhs) = default;

    /*!
     * \brief Assign to the matrix by copy
     * \param rhs The right-hand-side matrix
     * \return a reference to the assigned matrix
     */
    uni_lower_matrix& operator=(const uni_lower_matrix& rhs) = default;

    /*!
     * \brief Construct a uni_lower_matrix by move
     * \param rhs The right-hand-side matrix
     */
    uni_lower_matrix(uni_lower_matrix&& rhs) noexcept = default;

    /*!
     * \brief Assign to the matrix by move
     * \param rhs The right-hand-side matrix
     * \return a reference to the assigned matrix
     */
    uni_lower_matrix& operator=(uni_lower_matrix&& rhs) noexcept = default;

    /*!
     * \brief Assign the values of the ETL expression to the uni lower triangular matrix
     * \param e The ETL expression to get the values from
     * \return a reference to the fast matrix
     */
    template <convertible_expr<value_type> E>
    uni_lower_matrix& operator=(E&& e) noexcept(false) {
        // Make sure the other matrix is uni lower triangular
        if (!is_uni_lower_triangular(e)) {
            throw uni_lower_exception();
        }

        // Perform the real assign

        validate_assign(*this, e);

        // Avoid aliasing issues
        if constexpr (!decay_traits<E>::is_linear) {
            if (e.alias(*this)) {
                // Create a temporary to hold the result
                this_type tmp(*this);

                // Assign the expression to the temporary
                tmp = e;

                // Assign the temporary to this matrix
                *this = tmp;
            } else {
                e.assign_to(*this);
            }
        } else {
            // Direct assignment of the expression into this matrix
            e.assign_to(*this);
        }

        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template <etl_expr R>
    uni_lower_matrix& operator+=(const R & rhs) {
        // Make sure the other matrix is uni lower triangular
        if (!is_uni_lower_triangular(rhs)) {
            throw uni_lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_add_to(*this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template <etl_expr R>
    uni_lower_matrix& operator-=(const R & rhs) {
        // Make sure the other matrix is uni lower triangular
        if (!is_uni_lower_triangular(rhs)) {
            throw uni_lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_sub_to(*this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template <etl_expr R>
    uni_lower_matrix& operator*=(const R & rhs) {
        // Make sure the other matrix is uni lower triangular
        if (!is_uni_lower_triangular(rhs)) {
            throw uni_lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_mul_to(*this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template <etl_expr R>
    uni_lower_matrix& operator>>=(const R & rhs) {
        // Make sure the other matrix is uni lower triangular
        if (!is_uni_lower_triangular(rhs)) {
            throw uni_lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_mul_to(*this);
        return *this;
    }

    /*!
     * \brief Modulo each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template <etl_expr R>
    uni_lower_matrix& operator/=(const R & rhs) {
        // Make sure the other matrix is uni lower triangular
        if (!is_uni_lower_triangular(rhs)) {
            throw uni_lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_div_to(*this);
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
    uni_lower_detail::uni_lower_reference<matrix_t> operator()(size_t i, size_t j) noexcept {
        return {value, i, j};
    }

    using base_type::operator();
};

/*!
 * \brief Traits specialization for uni_lower_matrix
 */
template <typename Matrix>
struct etl_traits<uni_lower_matrix<Matrix>> : wrapper_traits<uni_lower_matrix<Matrix>> {};

} //end of namespace etl
