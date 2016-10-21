//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

namespace detail {

template <typename V>
V compute(std::size_t n, std::size_t i, std::size_t j) {
    if (n == 1) {
        return 1;
    } else if (n == 2) {
        return i == 0 && j == 0 ? 1 : i == 0 && j == 1 ? 3 : i == 1 && j == 0 ? 4
                                                                              : 2;
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

    const std::size_t n; ///< The dimensions of the magic matrix

    /*!
     * \brief Construct a new magic_view with the given dimension
     */
    explicit magic_view(std::size_t n)
            : n(n) {}

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) const {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        return detail::compute<value_type>(n, i / n, i % n);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) {
        return detail::compute<value_type>(n, i, j);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
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
};

/*!
 * \brief A view (virtual) of a static magic matrix
 */
template <typename V, std::size_t N>
struct fast_magic_view {
    using value_type = V; ///< The value type

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) const {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        return detail::compute<value_type>(N, i / N, i % N);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) {
        return detail::compute<value_type>(N, i, j);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
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
};

/*!
 * \brief traits speciflization for magic_view
 */
template <typename V>
struct etl_traits<etl::magic_view<V>> {
    using expr_t = etl::magic_view<V>; ///< The inspected expression type

    static constexpr const bool is_etl                  = true;            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;           ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = true;            ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = false;           ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = false;           ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe          = true;            ///< Indicates if the expression is thread safe
    static constexpr const bool is_value                = false;           ///< Indicates if the expression is of value type
    static constexpr const bool is_direct               = false;           ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = false;           ///< Indicates if the expression is a generator
    static constexpr const bool needs_temporary_visitor = false;           ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = false;           ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr const order storage_order          = order::RowMajor; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam VV The vector mode
     */
    template <vector_mode_t VV>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return v.n * v.n;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        cpp_unused(d);
        return v.n;
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

/*!
 * \brief traits speciflization for fast_magic_view
 */
template <std::size_t N, typename V>
struct etl_traits<etl::fast_magic_view<V, N>> {
    using expr_t = etl::fast_magic_view<V, N>; ///< The inspected expression type

    static constexpr const bool is_etl                  = true;            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;           ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = true;            ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = true;            ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = false;           ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe          = true;            ///< Indicates if the expression is thread safe
    static constexpr const bool is_value                = false;           ///< Indicates if the expression is of value type
    static constexpr const bool is_direct               = false;           ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = false;           ///< Indicates if the expression is a generator
    static constexpr const bool needs_temporary_visitor = false;           ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = false;           ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr const order storage_order          = order::RowMajor; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam VV The vector mode
     */
    template <vector_mode_t VV>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return N * N;
    }

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static constexpr std::size_t size(const expr_t& v) {
        return (void)v, N * N;
    }

    /*!
     * \brief Returns the D2th dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the D2th dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return N;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param e The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static constexpr std::size_t dim(const expr_t& e, std::size_t d) {
        return (void)e, (void)d, N;
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

} //end of namespace etl
