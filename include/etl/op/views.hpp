//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the views implementations of the expressions
 */

#pragma once

namespace etl {

/*!
 * Builder to construct the type returned by a view.
 */
template <typename T, typename S>
using return_helper =
    std::conditional_t<
        std::is_const<std::remove_reference_t<S>>::value,
        const value_t<T>&,
        std::conditional_t<
            cpp::and_u<
                std::is_lvalue_reference<S>::value,
                cpp::not_u<std::is_const<T>::value>::value>::value,
            value_t<T>&,
            value_t<T>>>;

/*!
 * Builder to construct the const type returned by a view.
 */
template <typename T, typename S>
using const_return_helper = std::conditional_t<
    std::is_lvalue_reference<S>::value,
    const value_t<T>&,
    value_t<T>>;

/*!
 * \brief View that shows one dimension of a matrix
 * \tparam T The type of expression on which the view is made
 * \tparam D The dimension to show
 */
template <typename T, std::size_t D>
struct dim_view {
    T sub;               ///< The Sub expression
    const std::size_t i; ///< The index

    static_assert(D == 1 || D == 2, "Invalid dimension");

    using sub_type          = T;                                                  ///< The sub type
    using value_type        = value_t<sub_type>;                                  ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                                 ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                           ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub(0, 0))>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub(0, 0))>; ///< The const type return by the view

    dim_view(sub_type sub, std::size_t i)
            : sub(sub), i(i) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator()(std::size_t j) const {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator()(std::size_t j) {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
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

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value && D == 1, "This expression does not have direct memory access");
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value && D == 1, "This expression does not have direct memory access");
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value && D == 1, "This expression does not have direct memory access");
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value && D == 1, "This expression does not have direct memory access");
        return sub.memory_start() + (i + 1) * subsize(sub);
    }
};

/*!
 * \brief View that shows a sub matrix of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T>
struct sub_view {
    T sub;               ///< The Sub expression
    const std::size_t i; ///< The index

    using sub_type          = T;                                               ///< The sub type
    using value_type        = value_t<sub_type>;                               ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                              ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                        ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>; ///< The const type return by the view

    sub_view(sub_type sub, std::size_t i)
            : sub(sub), i(i) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub[i * subsize(sub) + j]
                   : sub[i + dim<0>(sub) * j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub[i * subsize(sub) + j]
                   : sub[i + dim<0>(sub) * j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub.read_flat(i * subsize(sub) + j)
                   : sub.read_flat(i + dim<0>(sub) * j);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    const_return_type operator()(S... args) const {
        return sub(i, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    return_type operator()(S... args) {
        return sub(i, static_cast<std::size_t>(args)...);
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

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + (i + 1) * subsize(sub);
    }
};

namespace fast_matrix_view_detail {

template <typename M, std::size_t I, typename Enable = void>
struct matrix_subsize : std::integral_constant<std::size_t, M::template dim<I + 1>() * matrix_subsize<M, I + 1>::value> {};

template <typename M, std::size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == M::n_dimensions - 1>> : std::integral_constant<std::size_t, 1> {};

template <typename M, std::size_t I>
inline constexpr std::size_t compute_index(std::size_t first) noexcept {
    return first;
}

template <typename M, std::size_t I, typename... S, cpp_enable_if((sizeof...(S) > 0))>
inline constexpr std::size_t compute_index(std::size_t first, S... args) noexcept {
    return matrix_subsize<M, I>::value * first + compute_index<M, I + 1>(args...);
}

} //end of namespace fast_matrix_view_detail

/*!
 * \brief View to represent a fast matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 * \tparam Dims The dimensios of the view
 */
template <typename T, std::size_t... Dims>
struct fast_matrix_view {
    T sub; ///< The Sub expression

    using sub_type          = T;                                               ///< The sub type
    using value_type        = value_t<sub_type>;                               ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                              ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                        ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>; ///< The const type return by the view

    static constexpr std::size_t n_dimensions = sizeof...(Dims);

    explicit fast_matrix_view(sub_type sub)
            : sub(sub) {}

    template <typename... S>
    static constexpr std::size_t index(S... args) {
        return fast_matrix_view_detail::compute_index<fast_matrix_view<T, Dims...>, 0>(args...);
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        return sub[j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        return sub.read_flat(j);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    return_type& operator()(S... args) noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return sub[index(static_cast<std::size_t>(args)...)];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    const_return_type& operator()(S... args) const noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return sub[index(static_cast<std::size_t>(args)...)];
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() noexcept {
        return nth_size<D, 0, Dims...>::value;
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

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }
};

/*!
 * \brief View to represent a dyn vector in top of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T>
struct dyn_vector_view {
    T sub;            ///< The Sub expression
    std::size_t rows; ///< The number of rows

    using sub_type          = T;                                               ///< The sub type
    using value_type        = value_t<sub_type>;                               ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                              ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                        ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>; ///< The const type return by the view

    dyn_vector_view(sub_type sub, std::size_t rows)
            : sub(sub), rows(rows) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        return sub[j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        return sub.read_flat(j);
    }

    /*!
     * \brief Access to the element at the given position
     * \param j The index
     * \return a reference to the element at the given position.
     */
    return_type operator()(std::size_t j) {
        return sub[j];
    }

    /*!
     * \brief Access to the element at the given position
     * \param j The index
     * \return a reference to the element at the given position.
     */
    const_return_type operator()(std::size_t j) const {
        return sub[j];
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

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }
};

/*!
 * \brief View to represent a dyn matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T>
struct dyn_matrix_view {
    T sub;               ///< The sub expression
    std::size_t rows;    ///< The number of rows
    std::size_t columns; ///< The number columns

    using sub_type          = T;                                               ///< The sub type
    using value_type        = value_t<sub_type>;                               ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                              ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                        ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>; ///< The const type return by the view

    dyn_matrix_view(sub_type sub, std::size_t rows, std::size_t columns)
            : sub(sub), rows(rows), columns(columns) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        return sub[j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        return sub[j];
    }

    /*!
     * \brief Access to the element at the given position
     * \param j The index
     * \return a reference to the element at the given position.
     */
    const_return_type operator()(std::size_t j) const {
        return sub[j];
    }

    /*!
     * \brief Access to the element at the given position
     * \param i The first index
     * \param j The second index
     * \return a reference to the element at the given position.
     */
    const_return_type operator()(std::size_t i, std::size_t j) const {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub[i * columns + j]
                   : sub[i + rows * j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        return sub.read_flat(j);
    }

    /*!
     * \brief Access to the element at the given position
     * \param j The index
     * \return a reference to the element at the given position.
     */
    return_type operator()(std::size_t j) {
        return sub[j];
    }

    /*!
     * \brief Access to the element at the given (i,j) position
     * \param i The first index
     * \param j The second index
     * \return a reference to the element at the given position.
     */
    return_type operator()(std::size_t i, std::size_t j) {
        return decay_traits<sub_type>::storage_order == order::RowMajor
                   ? sub[i * columns + j]
                   : sub[i + rows * j];
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

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value, "This expression does not have direct memory access");
        return sub.memory_end();
    }
};

/*!
 * \brief Specialization for dim_view
 */
template <typename T, std::size_t D>
struct etl_traits<etl::dim_view<T, D>> {
    using expr_t     = etl::dim_view<T, D>; ///< The expression type
    using sub_expr_t = std::decay_t<T>; ///< The sub expression type

    static constexpr const bool is_etl                  = true;  ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;  ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true; ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false; ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast; ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = false; ///< Indicates if the expression is linear
    static constexpr const bool is_value                = false; ///< Indicates if the expression is of value type
    static constexpr const bool is_generator            = false; ///< Indicates if the expression is a generator
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<typename V>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        if (D == 1) {
            return etl_traits<sub_expr_t>::dim(v.sub, 1);
        } else {
            return etl_traits<sub_expr_t>::dim(v.sub, 0);
        }
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        cpp_assert(d == 0, "Invalid dimension");
        cpp_unused(d);

        return size(v);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return D == 1 ? etl_traits<sub_expr_t>::template dim<1>() : etl_traits<sub_expr_t>::template dim<0>();
    }

    /*!
     * \brief Returns the D2th dimension of an expression of this type
     * \tparam D2 The dimension to get
     * \return the D2th dimension of an expression of this type
     */
    template <std::size_t D2>
    static constexpr std::size_t dim() {
        static_assert(D2 == 0, "Invalid dimension");

        return size();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Specialization for sub_view
 */
template <typename T>
struct etl_traits<etl::sub_view<T>> {
    using expr_t     = etl::sub_view<T>; ///< The expression type
    using sub_expr_t = std::decay_t<T>; ///< The sub expression type

    static constexpr const bool is_etl                  = true;  ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;  ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true; ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false; ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast; ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear; ///< Indicates if the expression is linear
    static constexpr const bool is_value                = false; ///< Indicates if the expression is of value type
    static constexpr const bool is_generator            = false; ///< Indicates if the expression is a generator
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<typename V>
    using vectorizable = cpp::bool_constant<has_direct_access<sub_expr_t>::value && storage_order == order::RowMajor>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.sub) / etl_traits<sub_expr_t>::dim(v.sub, 0);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.sub, d + 1);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size() / etl_traits<sub_expr_t>::template dim<0>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D + 1>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions() - 1;
    }
};

/*!
 * \brief Specialization for fast_matrix_view.
 */
template <typename T, std::size_t... Dims>
struct etl_traits<etl::fast_matrix_view<T, Dims...>> {
    using expr_t     = etl::fast_matrix_view<T, Dims...>; ///< The expression type
    using sub_expr_t = std::decay_t<T>; ///< The sub expression type

    static constexpr const bool is_etl                  = true;  ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;  ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true; ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false; ///< Indicates if the type is a magic view
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear; ///< Indicates if the expression is linear
    static constexpr const bool is_fast                 = true; ///< Indicates if the expression is fast
    static constexpr const bool is_value                = false; ///< Indicates if the expression is of value type
    static constexpr const bool is_generator            = false; ///< Indicates if the expression is a generator
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<typename V>
    using vectorizable = std::false_type;

    static constexpr std::size_t size(const expr_t& /*unused*/) {
        return mul_all<Dims...>::value;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        cpp_unused(v);
        return dyn_nth_size<Dims...>(d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return mul_all<Dims...>::value;
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return nth_size<D, 0, Dims...>::value;
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return sizeof...(Dims);
    }
};

/*!
 * \brief Specialization for dyn_matrix_view.
 */
template <typename T>
struct etl_traits<etl::dyn_matrix_view<T>> {
    using expr_t     = etl::dyn_matrix_view<T>; ///< The expression type
    using sub_expr_t = std::decay_t<T>; ///< The sub expression type

    static constexpr const bool is_etl                  = true;  ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;  ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true; ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false; ///< Indicates if the type is a magic view
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear; ///< Indicates if the expression is linear
    static constexpr const bool is_fast                 = false; ///< Indicates if the expression is fast
    static constexpr const bool is_value                = false; ///< Indicates if the expression is of value type
    static constexpr const bool is_generator            = false; ///< Indicates if the expression is a generator
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<typename V>
    using vectorizable = std::false_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return v.rows * v.columns;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return d == 0 ? v.rows : v.columns;
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
 * \brief Specialization for dyn_vector_view.
 */
template <typename T>
struct etl_traits<etl::dyn_vector_view<T>> {
    using expr_t     = etl::dyn_vector_view<T>; ///< The expression type
    using sub_expr_t = std::decay_t<T>; ///< The sub expression type

    static constexpr const bool is_etl                  = true;  ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;  ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true; ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false; ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = false; ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear; ///< Indicates if the expression is linear
    static constexpr const bool is_value                = false; ///< Indicates if the expression is of value type
    static constexpr const bool is_generator            = false; ///< Indicates if the expression is a generator
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<typename V>
    using vectorizable = std::false_type;


    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return v.rows;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        cpp_unused(d);
        return v.rows;
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return 1;
    }
};


template <typename T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const dim_view<T, D>& v) {
    return os << "dim[" << D << "](" << v.sub << ", " << v.i << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const sub_view<T>& v) {
    return os << "sub(" << v.sub << ", " << v.i << ")";
}

template <typename T, std::size_t Rows, std::size_t Columns>
std::ostream& operator<<(std::ostream& os, const fast_matrix_view<T, Rows, Columns>& v) {
    return os << "reshape[" << Rows << "," << Columns << "](" << v.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const dyn_matrix_view<T>& v) {
    return os << "reshape[" << v.rows << "," << v.columns << "](" << v.sub << ")";
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const dyn_vector_view<T>& v) {
    return os << "reshape[" << v.rows << "](" << v.sub << ")";
}

} //end of namespace etl
