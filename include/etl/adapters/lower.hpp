//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains lower triangular matrix implementation
 */

#pragma once

#include "etl/adapters/adapter.hpp"         // The adapter base class
#include "etl/adapters/lower_exception.hpp" // The lower exception
#include "etl/adapters/lower_reference.hpp" // The reference proxy

namespace etl {

/*!
 * \brief A lower triangular matrix adapter.
 *
 * This is only a prototype.
 */
template <typename Matrix>
struct lower_matrix final : adapter<Matrix>, iterable<const lower_matrix<Matrix>> {
    using matrix_t  = Matrix;               ///< The adapted matrix type
    using expr_t    = matrix_t;             ///< The wrapped expression type
    using this_type = lower_matrix<Matrix>; ///< The type of this matrix

    static_assert(etl_traits<matrix_t>::is_value, "Lower triangular matrix only works with value classes");
    static_assert(is_2d<matrix_t>, "Lower triangular matrix must be two-dimensional");
    static_assert(is_square_matrix<matrix_t>, "Lower triangular matrix must be square");

    static constexpr size_t n_dimensions = etl_traits<matrix_t>::dimensions();  ///< The number of dimensions
    static constexpr order storage_order      = etl_traits<matrix_t>::storage_order; ///< The storage order
    static constexpr size_t alignment    = matrix_t::alignment;                 ///< The memory alignment

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

    using base_type::value;

public:
    /*!
     * \brief Construct a new lower triangular matrix and fill it with zeros
     *
     * This constructor can only be used when the matrix is fast
     */
    lower_matrix() noexcept : base_type() {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new lower triangular matrix and fill it witht the given value
     *
     * \param value The value to fill the matrix with
     *
     * This constructor can only be used when the matrix is fast
     */
    explicit lower_matrix(value_type value) noexcept : base_type(value) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new lower triangular matrix and fill it with zeros
     * \param dim The dimension of the matrix
     */
    explicit lower_matrix(size_t dim) noexcept : base_type(dim) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a lower_matrix by copy
     * \param rhs The right-hand-side matrix
     */
    lower_matrix(const lower_matrix& rhs) = default;

    /*!
     * \brief Assign to the matrix by copy
     * \param rhs The right-hand-side matrix
     * \return a reference to the assigned matrix
     */
    lower_matrix& operator=(const lower_matrix& rhs) = default;

    /*!
     * \brief Construct a lower_matrix by move
     * \param rhs The right-hand-side matrix
     */
    lower_matrix(lower_matrix&& rhs) noexcept = default;

    /*!
     * \brief Assign to the matrix by move
     * \param rhs The right-hand-side matrix
     * \return a reference to the assigned matrix
     */
    lower_matrix& operator=(lower_matrix&& rhs) noexcept = default;

    /*!
     * \brief Assign the values of the ETL expression to the lower triangular matrix
     * \param e The ETL expression to get the values from
     * \return a reference to the fast matrix
     */
    template <typename E, cpp_enable_iff(std::is_convertible<value_t<E>, value_type>::value && is_etl_expr<E>)>
    lower_matrix& operator=(E&& e) noexcept(false) {
        // Make sure the other matrix is lower triangular
        if(!is_lower_triangular(e)){
            throw lower_exception();
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
    template<typename R>
    lower_matrix& operator+=(R&& rhs){
        static_assert(is_etl_expr<R>, "Can only add ETL expression to lower triangular matrix");

        // Make sure the other matrix is lower triangular
        if(!is_lower_triangular(rhs)){
            throw lower_exception();
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
    template<typename R>
    lower_matrix& operator-=(R&& rhs){
        static_assert(is_etl_expr<R>, "Can only subtract ETL expression from lower triangular matrix");

        // Make sure the other matrix is lower triangular
        if(!is_lower_triangular(rhs)){
            throw lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_sub_to(*this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    lower_matrix& operator*=(const value_type& rhs) noexcept {
        etl::scalar<value_type>(rhs).assign_mul_to(*this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_iff(is_etl_expr<R>)>
    lower_matrix& operator*=(R&& rhs) {
        // Make sure the other matrix is lower triangular
        if(!is_lower_triangular(rhs)){
            throw lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_mul_to(*this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    lower_matrix& operator>>=(const value_type& rhs) noexcept {
        etl::scalar<value_type>(rhs).assign_mul_to(*this);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_iff(is_etl_expr<R>)>
    lower_matrix& operator>>=(R&& rhs) {
        // Make sure the other matrix is lower triangular
        if(!is_lower_triangular(rhs)){
            throw lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_mul_to(*this);
        return *this;
    }

    /*!
     * \brief Divide each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    lower_matrix& operator/=(const value_type& rhs) noexcept {
        etl::scalar<value_type>(rhs).assign_div_to(*this);
        return *this;
    }

    /*!
     * \brief Modulo each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_iff(is_etl_expr<R>)>
    lower_matrix& operator/=(R&& rhs) {
        // Make sure the other matrix is lower triangular
        if(!is_lower_triangular(rhs)){
            throw lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_div_to(*this);
        return *this;
    }

    /*!
     * \brief Modulo each element by the right hand side scalar
     * \param rhs The right hand side scalar
     * \return a reference to the matrix
     */
    lower_matrix& operator%=(const value_type& rhs) noexcept {
        etl::scalar<value_type>(rhs).assign_mod_to(*this);
        return *this;
    }

    /*!
     * \brief Modulo each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_iff(is_etl_expr<R>)>
    lower_matrix& operator%=(R&& rhs){
        // Make sure the other matrix is lower triangular
        if(!is_lower_triangular(rhs)){
            throw lower_exception();
        }

        validate_expression(*this, rhs);
        rhs.assign_mod_to(*this);
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
    lower_detail::lower_reference<matrix_t> operator()(size_t i, size_t j) noexcept {
        return {value, i, j};
    }

    using base_type::operator();
};

/*!
 * \brief Traits specialization for lower_matrix
 */
template <typename Matrix>
struct etl_traits<lower_matrix<Matrix>> : wrapper_traits<lower_matrix<Matrix>> {};

} //end of namespace etl
