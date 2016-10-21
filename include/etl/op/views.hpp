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

    /*!
     * \brief Construct a new dim_view over the given sub expression
     * \param sub The sub expression
     * \param i The sub index
     */
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
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
        return sub;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
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

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_type>;

    /*!
     * \brief Construct a new sub_view over the given sub expression
     * \param sub The sub expression
     * \param i The sub index
     */
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
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
        return sub;
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t x) noexcept {
        return value().template storeu<V>(in, x + i * subsize(sub));
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t x) noexcept {
        return value().template storeu<V>(in, x + i * subsize(sub));
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t x) noexcept {
        return value().template storeu<V>(in, x + i * subsize(sub));
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(std::size_t x) const noexcept {
        return sub.template loadu<V>(x + i * subsize(sub));
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(std::size_t x) const noexcept {
        return sub.template loadu<V>(x + i * subsize(sub));
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
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

    /*!
     * \brief Returns a reference to the ith dimension value.
     *
     * This should only be used internally and with care
     *
     * \return a refernece to the ith dimension value.
     */
    std::size_t& unsafe_dimension_access(std::size_t i) {
        return sub.unsafe_dimension_access(i + 1);
    }
};

//TODO Make slice view works with vector!

/*!
 * \brief View that shows a slice of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T>
struct slice_view {
    T sub;                   ///< The Sub expression
    const std::size_t first; ///< The index
    const std::size_t last;  ///< The last index

    using sub_type          = T;                                               ///< The sub type
    using value_type        = value_t<sub_type>;                               ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                              ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                        ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>; ///< The const type return by the view

    /*!
     * \brief Construct a new sub_view over the given sub expression
     * \param sub The sub expression
     * \param first The first index
     * \param last The last index
     */
    slice_view(sub_type sub, std::size_t first, std::size_t last)
            : sub(sub), first(first), last(last) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        if(decay_traits<sub_type>::storage_order == order::RowMajor){
            return sub[first * (size(sub) / dim<0>(sub)) + j];
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (size(sub) / sa);
            return sub[(j % ss) * sa + j / ss + first];
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        if(decay_traits<sub_type>::storage_order == order::RowMajor){
            return sub[first * (size(sub) / dim<0>(sub)) + j];
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (size(sub) / sa);
            return sub[(j % ss) * sa + j / ss + first];
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        if(decay_traits<sub_type>::storage_order == order::RowMajor){
            return sub.read_flat(first * (size(sub) / dim<0>(sub)) + j);
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (size(sub) / sa);
            return sub.read_flat((j % ss) * sa + j / ss + first);
        }
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    const_return_type operator()(std::size_t i, S... args) const {
        return sub(i + first, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    return_type operator()(std::size_t i, S... args) {
        return sub(i + first, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
        return sub;
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(std::size_t x) const noexcept {
        return sub.template loadu<V>(x + first * subsize(sub));
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(std::size_t x) const noexcept {
        return sub.template loadu<V>(x + first * subsize(sub));
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + first * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + first * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + last * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + last * subsize(sub);
    }
};

/*!
 * \brief View that shows a slice of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T>
struct memory_slice_view {
    T sub;                   ///< The Sub expression
    const std::size_t first; ///< The index
    const std::size_t last;  ///< The last index

    using sub_type          = T;                                               ///< The sub type
    using value_type        = value_t<sub_type>;                               ///< The value contained in the expression
    using memory_type       = value_type*;                                     ///< The memory acess type
    using const_memory_type = const value_type*;                               ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>; ///< The const type return by the view

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_type>;

    /*!
     * \brief Construct a new sub_view over the given sub expression
     * \param sub The sub expression
     * \param first The first index
     * \param last The last index
     */
    memory_slice_view(sub_type sub, std::size_t first, std::size_t last)
            : sub(sub), first(first), last(last) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        return sub[first + j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        return sub[first + j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        return sub[first + j];
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
        return sub;
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start.
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(std::size_t x) const noexcept {
        return sub.template loadu<V>(x + first );
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start.
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(std::size_t x) const noexcept {
        return sub.template loadu<V>(x + first );
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t i) noexcept {
        sub.template storeu<V>(in, first + i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t i) noexcept {
        sub.template storeu<V>(in, first + i);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t i) noexcept {
        sub.template storeu<V>(in, first + i);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        return sub.memory_start() + first ;
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        return sub.memory_start() + first ;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return sub.memory_start() + last ;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return sub.memory_start() + last;
    }
};

namespace fast_matrix_view_detail {

/*!
 * \brief Constant traits to get the subsize of a matrix for a fast_matrix_view
 */
template <typename M, std::size_t I, typename Enable = void>
struct matrix_subsize : std::integral_constant<std::size_t, M::template dim<I + 1>() * matrix_subsize<M, I + 1>::value> {};

/*!
 * \copydoc matrix_subsize
 */
template <typename M, std::size_t I>
struct matrix_subsize<M, I, std::enable_if_t<I == M::n_dimensions - 1>> : std::integral_constant<std::size_t, 1> {};

/*!
 * \brief Compute the index for a fast matrix of the given type
 * \param first The first index
 * \return the index inside the matrix
 */
template <typename M, std::size_t I>
inline constexpr std::size_t compute_index(std::size_t first) noexcept {
    return first;
}

/*!
 * \brief Compute the index for a fast matrix of the given type
 * \param first The first index
 * \param args The following indices
 * \return the index inside the matrix
 */
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

    static constexpr std::size_t n_dimensions = sizeof...(Dims); ///< The number of dimensions of the view

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_type>;

    /*!
     * \brief Construct a new fast_matrix_view over the given sub expression
     * \param sub The sub expression
     */
    explicit fast_matrix_view(sub_type sub)
            : sub(sub) {}

    /*!
     * \brief Compute the flat index for the given position.
     * \param args The indices
     * \return the index inside the matrix
     */
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
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
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
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(std::size_t x) const noexcept {
        return sub.template load<V>(x);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(std::size_t x) const noexcept {
        return sub.template loadu<V>(x);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t i) noexcept {
        sub.template store<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t i) noexcept {
        sub.template storeu<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t i) noexcept {
        sub.template stream<V>(in, i);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
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
template <typename T, size_t D>
struct dyn_matrix_view {
    T sub;               ///< The sub expression
    std::array<std::size_t, D> dimensions;
    size_t _size;

    using sub_type          = T;                                               ///< The sub type
    using value_type        = value_t<sub_type>;                               ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                              ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                        ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>; ///< The const type return by the view

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_type>;

    /*!
     * \brief Construct a new dyn_matrix_view over the given sub expression
     * \param dims The dimensions
     */
    template<typename... S>
    dyn_matrix_view(sub_type sub, S... dims)
            : sub(sub), dimensions{{dims...}}, _size(etl::size(sub)) {}

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
     * \param f1 The first index
     * \param f2 The second index
     * \param sizes The following indices
     * \return a reference to the element at the given position.
     */
    template<typename... S>
    const_return_type operator()(size_t f1, size_t f2, S... sizes) const {
        return sub[index(f1, f2, sizes...)];
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
     * \param f1 The first index
     * \param f2 The second index
     * \param sizes The following indices
     * \return a reference to the element at the given position.
     */
    template<typename... S>
    return_type operator()(size_t f1, size_t f2, S... sizes) {
        return sub[index(f1, f2, sizes...)];
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    sub_type& value() {
        return sub;
    }

    /*!
     * \brief Returns the value on which the transformer is working.
     * \return A reference  to the value on which the transformer is working.
     */
    const sub_type& value() const {
        return sub;
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(std::size_t x) const noexcept {
        return sub.template load<V>(x);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(std::size_t x) const noexcept {
        return sub.template loadu<V>(x);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t i) noexcept {
        sub.template stream<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t i) noexcept {
        sub.template store<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t i) noexcept {
        sub.template storeu<V>(in, i);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
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

private:
    /*!
     * \brief Return the flat index for the element at the given position
     * \param sizes The indices
     * \return The flat index
     */
    template <typename... S>
    std::size_t index(S... sizes) const noexcept(assert_nothrow) {
        //Note: Version with sizes moved to a std::array and accessed with
        //standard loop may be faster, but need some stack space (relevant ?)

        std::size_t index = 0;

        if (decay_traits<sub_type>::storage_order == order::RowMajor) {
            std::size_t subsize = _size;
            std::size_t i       = 0;

            cpp::for_each_in(
                [&subsize, &index, &i, this](std::size_t s) {
                    cpp_assert(s < dimensions[i], "Out of bounds");
                    subsize /= dimensions[i++];
                    index += subsize * s;
                },
                sizes...);
        } else {
            std::size_t subsize = 1;
            std::size_t i       = 0;

            cpp::for_each_in(
                [&subsize, &index, &i, this](std::size_t s) {
                    cpp_assert(s < dimensions[i], "Out of bounds");
                    index += subsize * s;
                    subsize *= dimensions[i++];
                },
                sizes...);
        }

        return index;
    }
};

/*!
 * \brief Specialization for dim_view
 */
template <typename T, std::size_t D>
struct etl_traits<etl::dim_view<T, D>> {
    using expr_t     = etl::dim_view<T, D>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;     ///< The sub expression type

    static constexpr const bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;                 ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = false;                                           ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe          = etl_traits<sub_expr_t>::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr const bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr const bool is_direct               = etl_traits<sub_expr_t>::is_direct && D == 1;     ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr const bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
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
    using sub_expr_t = std::decay_t<T>;  ///< The sub expression type

    static constexpr const bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;                 ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear;               ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe          = etl_traits<sub_expr_t>::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr const bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr const bool is_direct               = etl_traits<sub_expr_t>::is_direct && etl_traits<sub_expr_t>::storage_order == order::RowMajor;               ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr const bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = cpp::bool_constant<decay_traits<sub_expr_t>::template vectorizable<V>::value && storage_order == order::RowMajor>;

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
 * \brief Specialization for slice_view
 */
template <typename T>
struct etl_traits<etl::slice_view<T>> {
    using expr_t     = etl::slice_view<T>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;    ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>; ///< The traits of the sub expression

    static constexpr const bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = false;                                           ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = sub_traits::is_linear;               ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe          = sub_traits::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr const bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr const bool is_direct               = sub_traits::is_direct && sub_traits::storage_order == order::RowMajor;               ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr const bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool needs_temporary_visitor = sub_traits::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = sub_traits::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = sub_traits::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = cpp::bool_constant<sub_traits::template vectorizable<V>::value && storage_order == order::RowMajor>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return (sub_traits::size(v.sub) / sub_traits::dim(v.sub, 0)) * (v.last - v.first);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        if (d == 0) {
            return v.last - v.first;
        } else {
            return sub_traits::dim(v.sub, d);
        }
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return sub_traits::dimensions();
    }
};

/*!
 * \brief Specialization for memory_slice_view
 */
template <typename T>
struct etl_traits<etl::memory_slice_view<T>> {
    using expr_t     = etl::memory_slice_view<T>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;    ///< The sub expression type

    static constexpr const bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = false;                                           ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear;               ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe          = etl_traits<sub_expr_t>::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr const bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr const bool is_direct               = etl_traits<sub_expr_t>::is_direct;               ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr const bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = cpp::bool_constant<decay_traits<sub_expr_t>::template vectorizable<V>::value>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return v.last - v.first;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        cpp_unused(d);
        return v.last - v.first;
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
 * \brief Specialization for fast_matrix_view.
 */
template <typename T, std::size_t... Dims>
struct etl_traits<etl::fast_matrix_view<T, Dims...>> {
    using expr_t     = etl::fast_matrix_view<T, Dims...>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;                   ///< The sub expression type

    static constexpr const bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear;               ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe          = etl_traits<sub_expr_t>::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr const bool is_fast                 = true;                                            ///< Indicates if the expression is fast
    static constexpr const bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr const bool is_direct               = etl_traits<sub_expr_t>::is_direct;               ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr const bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = cpp::bool_constant<etl_traits<sub_expr_t>::template vectorizable<V>::value && storage_order == order::RowMajor>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static cpp14_constexpr std::size_t size(const expr_t& v) {
        cpp_unused(v);
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
template <typename T, size_t D>
struct etl_traits<etl::dyn_matrix_view<T, D>> {
    using expr_t     = etl::dyn_matrix_view<T, D>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;         ///< The sub expression type

    static constexpr const bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear;               ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe          = etl_traits<sub_expr_t>::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr const bool is_fast                 = false;                                           ///< Indicates if the expression is fast
    static constexpr const bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr const bool is_direct               = etl_traits<sub_expr_t>::is_direct;               ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr const bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor; ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = cpp::bool_constant<etl_traits<sub_expr_t>::template vectorizable<V>::value && storage_order == order::RowMajor>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return v._size;
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return v.dimensions[d];
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Print a representation of the view on the given stream
 * \param os The output stream
 * \param v The view to print
 * \return the output stream
 */
template <typename T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const dim_view<T, D>& v) {
    return os << "dim[" << D << "](" << v.sub << ", " << v.i << ")";
}

/*!
 * \brief Print a representation of the view on the given stream
 * \param os The output stream
 * \param v The view to print
 * \return the output stream
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const sub_view<T>& v) {
    return os << "sub(" << v.sub << ", " << v.i << ")";
}

/*!
 * \brief Print a representation of the view on the given stream
 * \param os The output stream
 * \param v The view to print
 * \return the output stream
 */
template <typename T, std::size_t Rows, std::size_t Columns>
std::ostream& operator<<(std::ostream& os, const fast_matrix_view<T, Rows, Columns>& v) {
    return os << "reshape[" << Rows << "," << Columns << "](" << v.sub << ")";
}

/*!
 * \brief Print a representation of the view on the given stream
 * \param os The output stream
 * \param v The view to print
 * \return the output stream
 */
template <typename T, size_t D>
std::ostream& operator<<(std::ostream& os, const dyn_matrix_view<T, D>& v) {
    return os << "reshape[" << D << "D](" << v.sub << ")";
}

} //end of namespace etl
