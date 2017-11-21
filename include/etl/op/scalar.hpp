//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains scalar expression implementation
 */

#pragma once

namespace etl {

/*!
 * \brief Represents a scalar value
 */
template <typename T>
struct scalar {
    using value_type = T; ///< The value type

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    const T value; ///< The scalar value

    /*!
     * \brief Builds a new scalar
     * \þaram v The scalar value
     */
    explicit constexpr scalar(T v)
            : value(v) {}

    /*!
     * \brief Returns the element at the given index
     * \param d The index
     * \return a reference to the element at the given index.
     */
    constexpr T operator[](size_t d) const noexcept {
        return (void)d, value;
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param d The index
     * \return the value at the given index.
     */
    constexpr T read_flat(size_t d) const noexcept {
        return (void)d, value;
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param d The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> load(size_t d) const noexcept {
        return (void)d, V::set(value);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param d The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> loadu(size_t d) const noexcept {
        return (void)d, V::set(value);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    constexpr T operator()(__attribute__((unused)) S... args) const noexcept {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return value;
    }

    /*!
     * \brief Indicate if the expression aliases with the given expression.
     * \return true if the expressions alias, false otherwise
     */
    template <typename E>
    constexpr bool alias(const E& /*rhs*/) const noexcept {
        return false;
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_add_to(L&& lhs)  const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_sub_to(L&& lhs)  const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mul_to(L&& lhs)  const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_div_to(L&& lhs)  const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mod_to(L&& lhs)  const {
        std_mod_evaluate(*this, lhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template<typename V>
    void visit(V&& visitor) const {
        cpp_unused(visitor);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template<typename Y>
    decltype(auto) gpu_compute_hint(Y& y) const {
        // TODO Maybe make a full vector with the hint
        cpp_unused(y);

        gpu_dyn_matrix_impl<T, order::RowMajor, 1> t1;

        t1.resize_scalar();
        t1.ensure_gpu_allocated();

#ifdef ETL_CUDA
        cuda_check(cudaMemcpy(t1.gpu_memory(), &value, 1 * sizeof(T), cudaMemcpyHostToDevice));
#endif

        return t1;
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template<typename Y>
    decltype(auto) gpu_compute(Y& y) const {
        y.ensure_gpu_allocated();

#ifdef ETL_CUDA
        cuda_check(cudaMemcpy(y.gpu_memory(), &value, etl::size(y) * sizeof(T), cudaMemcpyHostToDevice));
#endif

        return y;
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // Nothing to ensure
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // Nothing to ensure
    }

    /*!
     * \brief Prints a scalar value to the given stream
     * \param os The output stream
     * \param s The scalar to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const scalar& s) {
        return os << s.value;
    }
};

/*!
 * \brief Specialization scalar
 */
template <typename T>
struct etl_traits<etl::scalar<T>, void> {
    using value_type = T; ///< The value type of the expression

    static constexpr bool is_etl         = true;                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = true;                                 ///< Indicates if the expression is fast
    static constexpr bool is_value       = false;                                ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = false;                                ///< Indicates if the expression has direct memory access
    static constexpr bool is_linear      = true;                                 ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;                                 ///< Indicates if the expression is thread safe
    static constexpr bool is_generator   = true;                                 ///< Indicates if the expression is a generator expression
    static constexpr bool is_temporary   = false;                                ///< Indicaes if the expression needs an evaluator visitor
    static constexpr bool is_padded      = true;                                 ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = true;                                 ///< Indicates if the expression is padded
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled; ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = order::RowMajor;                      ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam VV The vector mode
     */
    template <vector_mode_t VV>
    static constexpr bool vectorizable = true;

    /*!
     * \brief Return the size of the expression
     */
    static constexpr size_t size() {
        return 0;
    }

    /*!
     * \brief Return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 0;
    }
};

} //end of namespace etl
