//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief Transform (dynamic) that flips a matrix horizontally
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct hflip_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub; ///< The subexpression

    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit hflip_transformer(sub_type expr)
            : sub(expr) {}

    static constexpr const bool matrix = etl_traits<std::decay_t<sub_type>>::dimensions() == 2; ///< INdicates if the sub type is a matrix or not

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    template <bool C = matrix, cpp_disable_if(C)>
    value_type operator[](std::size_t i) const {
        return sub[size(sub) - i - 1];
    }

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    template <bool C = matrix, cpp_enable_if(C)>
    value_type operator[](std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return sub[i_i * dim<1>(sub) + (dim<1>(sub) - 1 - i_j)];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    template <bool C = matrix, cpp_disable_if(C)>
    value_type read_flat(std::size_t i) const {
        return sub.read_flat(size(sub) - i - 1);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    template <bool C = matrix, cpp_enable_if(C)>
    value_type read_flat(std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return sub.read_flat(i_i * dim<1>(sub) + (dim<1>(sub) - 1 - i_j));
    }

    /*!
     * \brief Access to the value at the given (i) position
     * \param i The index
     * \return The value at the position (i)
     */
    value_type operator()(std::size_t i) const {
        return sub(size(sub) - 1 - i);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(i, columns(sub) - 1 - j);
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }
};

/*!
 * \brief Transform (dynamic) that flips a matrix vertically
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct vflip_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub; ///< The subexpression

    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit vflip_transformer(sub_type expr)
            : sub(expr) {}

    static constexpr const bool matrix = etl_traits<std::decay_t<sub_type>>::dimensions() == 2; ///< Indicates if the sub type is a 2D matrix or not

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    template <bool C = matrix, cpp_disable_if(C)>
    value_type operator[](std::size_t i) const {
        return sub[i];
    }

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    template <bool C = matrix, cpp_enable_if(C)>
    value_type operator[](std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return sub[(dim<0>(sub) - 1 - i_i) * dim<1>(sub) + i_j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    template <bool C = matrix, cpp_disable_if(C)>
    value_type read_flat(std::size_t i) const {
        return sub.read_flat(i);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    template <bool C = matrix, cpp_enable_if(C)>
    value_type read_flat(std::size_t i) const {
        std::size_t i_i = i / dim<1>(sub);
        std::size_t i_j = i % dim<1>(sub);
        return sub.read_flat((dim<0>(sub) - 1 - i_i) * dim<1>(sub) + i_j);
    }

    /*!
     * \brief Access to the value at the given (i) position
     * \param i The index
     * \return The value at the position (i)
     */
    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(rows(sub) - 1 - i, j);
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }
};

/*!
 * \brief Transform (dynamic) that flips a matrix vertically and horizontally.
 * \tparam T The type on which the transformer is applied
 */
template <typename T>
struct fflip_transformer {
    using sub_type   = T;           ///< The type on which the expression works
    using value_type = value_t<T>;  ///< The type of valuie

    sub_type sub; ///< The subexpression

    /*!
     * \brief Construct a new transformer around the given expression
     * \param expr The sub expression
     */
    explicit fflip_transformer(sub_type expr)
            : sub(expr) {}

    /*!
     * \brief Returns the value at the given index
     * \param i The index
     * \return the value at the given index.
     */
    value_type operator[](std::size_t i) const {
        if (dimensions(sub) == 1) {
            return sub[i];
        } else {
            return sub[size(sub) - i - 1];
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        if (dimensions(sub) == 1) {
            return sub.read_flat(i);
        } else {
            return sub.read_flat(size(sub) - i - 1);
        }
    }

    /*!
     * \brief Access to the value at the given (i) position
     * \param i The index
     * \return The value at the position (i)
     */
    value_type operator()(std::size_t i) const {
        return sub(i);
    }

    /*!
     * \brief Access to the value at the given (i, j) position
     * \param i The first index
     * \param j The second index
     * \return The value at the position (i, j)
     */
    value_type operator()(std::size_t i, std::size_t j) const {
        return sub(rows(sub) - 1 - i, columns(sub) - 1 - j);
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }
};

/*!
 * \brief Display the transformer on the given stream
 * \param os The output stream
 * \param transformer The transformer to print
 * \return the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const hflip_transformer<T>& transformer) {
    return os << "hflip(" << transformer.sub << ")";
}

/*!
 * \brief Display the transformer on the given stream
 * \param os The output stream
 * \param transformer The transformer to print
 * \return the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const vflip_transformer<T>& transformer) {
    return os << "vflip(" << transformer.sub << ")";
}

/*!
 * \brief Display the transformer on the given stream
 * \param os The output stream
 * \param transformer The transformer to print
 * \return the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const fflip_transformer<T>& transformer) {
    return os << "fflip(" << transformer.sub << ")";
}

} //end of namespace etl
