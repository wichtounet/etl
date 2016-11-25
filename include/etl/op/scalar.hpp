//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
    constexpr T operator[](std::size_t d) const noexcept {
        return (void)d, value;
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param d The index
     * \return the value at the given index.
     */
    constexpr T read_flat(std::size_t d) const noexcept {
        return (void)d, value;
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param d The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    constexpr vec_type<V> load(std::size_t d) const noexcept {
        return (void)d, V::set(value);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param d The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    constexpr vec_type<V> loadu(std::size_t d) const noexcept {
        return (void)d, V::set(value);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    constexpr T operator()(__attribute__((unused)) S... args) const noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

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
};

/*!
 * \brief Specialization scalar
 */
template <typename T>
struct etl_traits<etl::scalar<T>, void> {
    static constexpr bool is_etl                  = true;            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;           ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;           ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = true;            ///< Indicates if the expression is fast
    static constexpr bool is_value                = false;           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = false;           ///< Indicates if the expression has direct memory access
    static constexpr bool is_linear               = true;            ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;            ///< Indicates if the expression is thread safe
    static constexpr bool is_generator            = true;            ///< Indicates if the expression is a generator expression
    static constexpr bool needs_temporary_visitor = false;           ///< Indicates if the expression needs a temporary visitor
    static constexpr bool needs_evaluator_visitor = false;           ///< Indicaes if the expression needs an evaluator visitor
    static constexpr bool is_padded               = true;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = true;                          ///< Indicates if the expression is padded
    static constexpr order storage_order          = order::RowMajor; ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam VV The vector mode
     */
    template <vector_mode_t VV>
    using vectorizable = std::true_type;

    /*!
     * \brief Return the size of the expression
     */
    static constexpr std::size_t size() {
        return 0;
    }
};

/*!
 * \brief Prints a scalar value to the given stream
 * \param os The output stream
 * \param s The scalar to print
 * \return the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const scalar<T>& s) {
    return os << s.value;
}

} //end of namespace etl
