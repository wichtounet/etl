//=======================================================================
// Copyright (c) 2014-2015 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "cpp_utils/tmp.hpp"

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
    template<typename V = default_vec>
    using vec_type   = typename V::template vec_type<T>;

    const T value; ///< The scalar value

    /*!
     * \brief Builds a new scalar
     * \þaram v The scalar value
     */
    explicit constexpr scalar(T v)
            : value(v) {}

    constexpr T operator[](std::size_t /*d*/) const noexcept {
        return value;
    }

    constexpr T read_flat(std::size_t /*d*/) const noexcept {
        return value;
    }

    template<typename V = default_vec>
    constexpr const vec_type<V> load(std::size_t /*d*/) const noexcept {
        return V::set(value);
    }

    template <typename... S>
    constexpr T operator()(S... /*args*/) const noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value;
    }

    template<typename E>
    constexpr bool alias(const E& /*rhs*/) const noexcept {
        return false;
    }
};

/*!
 * \brief Specialization scalar
 */
template <typename T>
struct etl_traits<etl::scalar<T>, void> {
    static constexpr const bool is_etl                  = true;            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;           ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;           ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = true;            ///< Indicates if the expression is fast
    static constexpr const bool is_value                = false;           ///< Indicates if the expression is of value type
    static constexpr const bool is_linear               = true;            ///< Indicates if the expression is linear
    static constexpr const bool is_generator            = true;            ///< Indicates if the expression is a generator expression
    static constexpr const bool vectorizable            = true;            ///< Indicates if the expression is vectorizable
    static constexpr const bool needs_temporary_visitor = false;           ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = false;           ///< Indicaes if the expression needs an evaluator visitor
    static constexpr const order storage_order          = order::RowMajor; ///< The expression storage order
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
