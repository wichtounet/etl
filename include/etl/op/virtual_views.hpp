//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief Compute the (i,j) element of a magic matrix of size n
 * \param n The size of the magix matrix
 * \param i The first index
 * \param j The second index
 * \return The (i,j) element of the magix matrix
 */
template <typename V>
V compute(size_t n, size_t i, size_t j) {
    if (n == 1) {
        return 1;
    } else if (n == 2) {
        return i == 0 && j == 0 ? 1 : i == 0 && j == 1 ? 3 : i == 1 && j == 0 ? 4 : 2;
    } else {
        //Siamese method
        return n * (((i + 1) + (j + 1) - 1 + n / 2) % n) + (((i + 1) + 2 * (j + 1) - 2) % n) + 1;
    }
}

} //end of namespace detail

//Note: Matrix of even order > 2 are only pseudo-magic
//TODO Add algorithm for even order

/*!
 * \brief A view (virtual) of a dynamic magic matrix
 */
template <typename V>
struct magic_view {
    using value_type = V; ///< The value type in the matrix

    const size_t n; ///< The dimensions of the magic matrix

    static constexpr bool gpu_computable = false;

    /*!
     * \brief Construct a new magic_view with the given dimension
     */
    explicit magic_view(size_t n) : n(n) {}

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](size_t i) {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](size_t i) const {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(size_t i, size_t j) {
        return detail::compute<value_type>(n, i, j);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(size_t i, size_t j) const {
        return detail::compute<value_type>(n, i, j);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    constexpr bool alias(const E& rhs) const noexcept {
        return (void)rhs, false;
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template <typename Visitor>
    void visit([[maybe_unused]] Visitor&& visitor) const {}

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
};

/*!
 * \brief A view (virtual) of a static magic matrix
 */
template <typename V, size_t N>
struct fast_magic_view {
    using value_type = V; ///< The value type

    static constexpr bool gpu_computable = false;

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](size_t i) {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](size_t i) const {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(size_t i, size_t j) {
        return detail::compute<value_type>(N, i, j);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(size_t i, size_t j) const {
        return detail::compute<value_type>(N, i, j);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    constexpr bool alias(const E& rhs) const noexcept {
        return (void)rhs, false;
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template <typename Visitor>
    void visit([[maybe_unused]] Visitor&& visitor) const {}

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
};

/*!
 * \brief traits speciflization for magic_view
 */
template <typename V>
struct etl_traits<etl::magic_view<V>> {
    using expr_t     = etl::magic_view<V>; ///< The inspected expression type
    using value_type = V;                  ///< The value type of the expression

    static constexpr bool is_etl         = true;            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;           ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;           ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = true;            ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = false;           ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;           ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;            ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;           ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = false;           ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;           ///< Indicates if the expression is a generator
    static constexpr bool is_temporary   = false;           ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr bool is_padded      = false;           ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = false;           ///< Indicates if the expression is padded
    static constexpr bool gpu_computable = false;           ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = order::RowMajor; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam VV The vector mode
     */
    template <vector_mode_t VV>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) {
        return v.n * v.n;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, [[maybe_unused]] size_t d) {
        return v.n;
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief traits speciflization for fast_magic_view
 */
template <size_t N, typename V>
struct etl_traits<etl::fast_magic_view<V, N>> {
    using expr_t     = etl::fast_magic_view<V, N>; ///< The inspected expression type
    using value_type = V;                          ///< The value type of the expression

    static constexpr bool is_etl         = true;            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;           ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;           ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = true;            ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = true;            ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;           ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;            ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;           ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = false;           ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;           ///< Indicates if the expression is a generator
    static constexpr bool is_temporary   = false;           ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr bool is_padded      = false;           ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = false;           ///< Indicates if the expression is padded
    static constexpr bool gpu_computable = false;           ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = order::RowMajor; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam VV The vector mode
     */
    template <vector_mode_t VV>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() {
        return N * N;
    }

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static constexpr size_t size(const expr_t& v) {
        return (void)v, N * N;
    }

    /*!
     * \brief Returns the D2th dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the D2th dimension of an expression of this type
     */
    template <size_t D>
    static constexpr size_t dim() {
        return N;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param e The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static constexpr size_t dim(const expr_t& e, size_t d) {
        return (void)e, (void)d, N;
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return 2;
    }
};

} //end of namespace etl
