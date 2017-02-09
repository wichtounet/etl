//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains runtime matrix implementation
 */

#pragma once

#include <exception>

#include "etl/dyn_base.hpp"    //The base class and utilities

#include "etl/adapters/sym_reference.hpp"    // The reference proxy

namespace etl {

/*!
 * \brief Exception that is thrown when an operation is made to
 * a symmetric matrix that would render it non-symmetric.
 */
struct symmetric_exception : std::exception {
    /*!
     * \brief Returns a description of the exception
     */
    virtual const char* what() const noexcept {
        return "Invalid assignment to a symmetric matrix";
    }
};

/*!
 * \brief A symmetric matrix adapter.
 *
 * This is only a prototype.
 */
template <typename Matrix>
struct sym_matrix final : comparable<sym_matrix<Matrix>>, iterable<const sym_matrix<Matrix>> {
    using matrix_t = Matrix;   ///< The adapted matrix type
    using expr_t   = matrix_t; ///< The wrapped expression type

    static_assert(etl_traits<matrix_t>::is_value, "Symmetric matrix only works with value classes");
    static_assert(etl_traits<matrix_t>::dimensions() == 2, "Symmetric matrix must be two-dimensional");
    static_assert(is_square_matrix<matrix_t>::value, "Symmetric matrix must be square");

    static constexpr std::size_t n_dimensions = etl_traits<matrix_t>::dimensions();  ///< The number of dimensions
    static constexpr order storage_order      = etl_traits<matrix_t>::storage_order; ///< The storage order
    static constexpr std::size_t alignment    = matrix_t::alignment;                 ///< The memory alignment

    using value_type        = value_t<matrix_t>;                 ///< The value type
    using memory_type       = value_type*;                       ///< The memory type
    using const_memory_type = const value_type*;                 ///< The const memory type

    using iterator       = typename matrix_t::const_iterator; ///< The type of const iterator
    using const_iterator = typename matrix_t::const_iterator; ///< The type of const iterator

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<value_type>;

private:
    matrix_t matrix; ///< The adapted matrix

public:
    /*!
     * \brief Construct a new sym matrix and fill it with zeros
     *
     * This constructor can only be used when the matrix is fast
     */
    sym_matrix() noexcept : matrix(value_type()) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new sym matrix and fill it witht the given value
     *
     * \param value The value to fill the matrix with
     *
     * This constructor can only be used when the matrix is fast
     */
    sym_matrix(value_type value) noexcept : matrix(value) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new sym matrix and fill it with zeros
     * \param dim The dimension of the matrix
     */
    sym_matrix(std::size_t dim) noexcept : matrix(dim, dim, value_type()) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new sym matrix and fill it witht the given value
     *
     * \param value The value to fill the matrix with
     * \param dim The dimension of the matrix
     */
    sym_matrix(std::size_t dim, value_type value) noexcept : matrix(dim, dim, value) {
        //Nothing else to init
    }

    sym_matrix(const sym_matrix& rhs) = default;
    sym_matrix& operator=(const sym_matrix& rhs) = default;

    sym_matrix(sym_matrix&& rhs) = default;
    sym_matrix& operator=(sym_matrix&& rhs) = default;

    /*!
     * \brief Assign the values of the ETL expression to the symmetric matrix
     * \param e The ETL expression to get the values from
     * \return a reference to the fast matrix
     */
    template <typename E, cpp_enable_if(std::is_convertible<value_t<E>, value_type>::value, is_etl_expr<E>::value)>
    sym_matrix& operator=(E&& e) noexcept(false) {
        // Make sure the other matrix is symmetric
        if(!is_symmetric(e)){
            throw symmetric_exception();
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
    sym_matrix& operator+=(const value_type& rhs) noexcept {
        detail::scalar_add::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    sym_matrix& operator+=(const R& rhs){
        // Make sure the other matrix is symmetric
        if(!is_symmetric(rhs)){
            throw symmetric_exception();
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
    sym_matrix& operator-=(const value_type& rhs) noexcept {
        detail::scalar_sub::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    sym_matrix& operator-=(const R& rhs){
        // Make sure the other matrix is symmetric
        if(!is_symmetric(rhs)){
            throw symmetric_exception();
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
    sym_matrix& operator*=(const value_type& rhs) noexcept {
        detail::scalar_mul::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    sym_matrix& operator*=(const R& rhs) {
        // Make sure the other matrix is symmetric
        if(!is_symmetric(rhs)){
            throw symmetric_exception();
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
    sym_matrix& operator>>=(const value_type& rhs) noexcept {
        detail::scalar_mul::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Multiply each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    sym_matrix& operator>>=(const R& rhs) {
        // Make sure the other matrix is symmetric
        if(!is_symmetric(rhs)){
            throw symmetric_exception();
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
    sym_matrix& operator/=(const value_type& rhs) noexcept {
        detail::scalar_div::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Modulo each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    sym_matrix& operator/=(const R& rhs) {
        // Make sure the other matrix is symmetric
        if(!is_symmetric(rhs)){
            throw symmetric_exception();
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
    sym_matrix& operator%=(const value_type& rhs) noexcept {
        detail::scalar_mod::apply(*this, rhs);
        return *this;
    }

    /*!
     * \brief Modulo each element by the value of the elements in the right hand side expression
     * \param rhs The right hand side
     * \return a reference to the matrix
     */
    template<typename R, cpp_enable_if(is_etl_expr<R>::value)>
    sym_matrix& operator%=(const R& rhs){
        // Make sure the other matrix is symmetric
        if(!is_symmetric(rhs)){
            throw symmetric_exception();
        }

        validate_expression(*this, rhs);
        mod_evaluate(rhs, *this);
        return *this;
    }

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
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    const value_type& operator[](std::size_t i) const noexcept {
        return matrix[i];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type& operator[](std::size_t i) noexcept {
        return matrix[i];
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

    /*!
     * \brief Returns a reference to the underlying matrix
     *
     * This should only be used by ETL itself.
     */
    const expr_t& value() const noexcept {
        return matrix;
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     *
     * This should only be used by ETL itself in order not to void
     * the symmetric guarantee.
     */
    memory_type memory_start() noexcept {
        return matrix.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     *
     * This should only be used by ETL itself in order not to void
     * the symmetric guarantee.
     */
    const_memory_type memory_start() const noexcept {
        return matrix.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     *
     * This should only be used by ETL itself in order not to void
     * the symmetric guarantee.
     */
    memory_type memory_end() noexcept {
        return matrix.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     *
     * This should only be used by ETL itself in order not to void
     * the symmetric guarantee.
     */
    const_memory_type memory_end() const noexcept {
        return matrix.memory_end();
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template<typename V = default_vec>
    vec_type<V> load(std::size_t i) const noexcept {
        return matrix.template load<V>(i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template<typename V = default_vec>
    vec_type<V> loadu(std::size_t i) const noexcept {
        return matrix.template loadu<V>(i);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal stores
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t i) noexcept {
        matrix.template stream<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t i) noexcept {
        matrix.template store<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t i) noexcept {
        matrix.template storeu<V>(in, i);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return matrix.alias(rhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor) const {
        cpp_unused(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor) const {
        cpp_unused(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::evaluator_visitor& visitor) const {
        cpp_unused(visitor);
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return matrix.gpu_memory();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        matrix.gpu_evict();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        matrix.invalidate_cpu();
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        matrix.invalidate_gpu();
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        matrix.validate_cpu();
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        matrix.validate_gpu();
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        matrix.ensure_gpu_allocated();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        matrix.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        matrix.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const value_type* gpu_memory) const {
        matrix.gpu_copy_from(gpu_memory);
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return matrix.is_cpu_up_to_date();
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return matrix.is_gpu_up_to_date();
    }
};

template <typename Matrix>
struct etl_traits<sym_matrix<Matrix>> : wrapper_traits<sym_matrix<Matrix>> {};

} //end of namespace etl
