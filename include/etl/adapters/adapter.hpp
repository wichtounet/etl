//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains base class for adapters
 */

#pragma once

namespace etl {

/*!
 * \brief A base class for adapters.
 */
template <typename Matrix>
struct adapter {
    using matrix_t = Matrix;   ///< The adapted matrix type
    using expr_t   = matrix_t; ///< The wrapped expression type

    using value_type        = value_t<matrix_t>; ///< The value type
    using memory_type       = value_type*;       ///< The memory type
    using const_memory_type = const value_type*; ///< The const memory type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<value_type>;

protected:
    matrix_t value; ///< The adapted matrix

public:
    /*!
     * \brief Construct a new matrix and fill it with zeros
     *
     * This constructor can only be used when the matrix is fast
     */
    adapter() noexcept : value(value_type()) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new adapter matrix and fill it witht the given value
     *
     * \param value The value to fill the matrix with
     *
     * This constructor can only be used when the matrix is fast
     */
    explicit adapter(value_type value) noexcept : value(value) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new adapter matrix and fill it with zeros
     * \param dim The dimension of the matrix
     */
    explicit adapter(size_t dim) noexcept : value(dim, dim, value_type()) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new adapter matrix and fill it witht the given value
     *
     * \param value The value to fill the matrix with
     * \param dim The dimension of the matrix
     */
    adapter(size_t dim, value_type value) noexcept : value(dim, dim, value) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a adapter by copy
     * \param rhs The right-hand-side matrix
     */
    adapter(const adapter& rhs) = default;

    /*!
     * \brief Assign to the matrix by copy
     * \param rhs The right-hand-side matrix
     * \return a reference to the assigned matrix
     */
    adapter& operator=(const adapter& rhs) = default;

    /*!
     * \brief Construct a adapter by move
     * \param rhs The right-hand-side matrix
     */
    adapter(adapter&& rhs) noexcept = default;

    /*!
     * \brief Assign to the matrix by move
     * \param rhs The right-hand-side matrix
     * \return a reference to the assigned matrix
     */
    adapter& operator=(adapter&& rhs) noexcept = default;

    /*!
     * \brief Access the (i, j) element of the 2D matrix
     * \param i The index of the first dimension
     * \param j The index of the second dimension
     * \return a reference to the (i,j) element
     *
     * Accessing an element outside the matrix results in Undefined Behaviour.
     */
    const value_type& operator()(size_t i, size_t j) const noexcept {
        return value(i, j);
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    const value_type& operator[](size_t i) const noexcept {
        return value[i];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type& operator[](size_t i) noexcept {
        return value[i];
    }

    /*!
     * \returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const noexcept {
        return value.read_flat(i);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     *
     * This should only be used by ETL itself in order not to void
     * the adapter guarantee.
     */
    memory_type memory_start() noexcept {
        return value.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     *
     * This should only be used by ETL itself in order not to void
     * the adapter guarantee.
     */
    const_memory_type memory_start() const noexcept {
        return value.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     *
     * This should only be used by ETL itself in order not to void
     * the adapter guarantee.
     */
    memory_type memory_end() noexcept {
        return value.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     *
     * This should only be used by ETL itself in order not to void
     * the adapter guarantee.
     */
    const_memory_type memory_end() const noexcept {
        return value.memory_end();
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> load(size_t i) const noexcept {
        return value.template load<V>(i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> loadu(size_t i) const noexcept {
        return value.template loadu<V>(i);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal stores
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, size_t i) noexcept {
        value.template stream<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, size_t i) noexcept {
        value.template store<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, size_t i) noexcept {
        value.template storeu<V>(in, i);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return value.alias(rhs);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    auto& gpu_compute_hint([[maybe_unused]] Y& y) {
        value.ensure_gpu_up_to_date();
        return value;
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    const auto& gpu_compute_hint([[maybe_unused]] Y& y) const {
        value.ensure_gpu_up_to_date();
        return value;
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(value, std::forward<L>(lhs));
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(value, std::forward<L>(lhs));
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(value, std::forward<L>(lhs));
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(value, std::forward<L>(lhs));
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(value, std::forward<L>(lhs));
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(value, std::forward<L>(lhs));
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit([[maybe_unused]] const detail::evaluator_visitor& visitor) const {}

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return value.gpu_memory();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        value.gpu_evict();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        value.invalidate_cpu();
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        value.invalidate_gpu();
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        value.validate_cpu();
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        value.validate_gpu();
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        value.ensure_gpu_allocated();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        value.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        value.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const value_type* gpu_memory) const {
        value.gpu_copy_from(gpu_memory);
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return value.is_cpu_up_to_date();
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return value.is_gpu_up_to_date();
    }

    /*!
     * \brief Returns the number of dimensions of the matrix
     * \return the number of dimensions of the matrix
     */
    static constexpr size_t dimensions() noexcept {
        return 2;
    }
};

} //end of namespace etl
