//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        value().visit(visitor);
        visitor.need_value = old_need_value;
    }
};

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
     * \brief Construct a new slice_view over the given sub expression
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
        return sub.template loadu<V>(x + first * (etl::size(sub) / etl::dim<0>(sub)));
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(std::size_t x) const noexcept {
        return sub.template loadu<V>(x + first * (etl::size(sub) / etl::dim<0>(sub)));
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
        return sub.memory_start() + first * (etl::size(sub) / etl::dim<0>(sub));
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + first * (etl::size(sub) / etl::dim<0>(sub));
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + last * (etl::size(sub) / etl::dim<0>(sub));
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + last * (etl::size(sub) / etl::dim<0>(sub));
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        value().visit(visitor);
        visitor.need_value = old_need_value;
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
     * \brief Construct a new memory_slice_view over the given sub expression
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

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        value().visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        value().visit(visitor);
        visitor.need_value = old_need_value;
    }
};

/*!
 * \brief Specialization for dim_view
 */
template <typename T, std::size_t D>
struct etl_traits<etl::dim_view<T, D>> {
    using expr_t     = etl::dim_view<T, D>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;     ///< The sub expression type
    using value_type = typename etl_traits<sub_expr_t>::value_type; ///< The value type

    static constexpr bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = etl_traits<sub_expr_t>::is_fast;                 ///< Indicates if the expression is fast
    static constexpr bool is_linear               = false;                                           ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = etl_traits<sub_expr_t>::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = etl_traits<sub_expr_t>::is_direct && D == 1;     ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

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
 * \brief Specialization for slice_view
 */
template <typename T>
struct etl_traits<etl::slice_view<T>> {
    using expr_t     = etl::slice_view<T>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;    ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>; ///< The traits of the sub expression
    using value_type = typename etl_traits<sub_expr_t>::value_type; ///< The value type

    static constexpr bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = false;                                           ///< Indicates if the expression is fast
    static constexpr bool is_linear               = sub_traits::is_linear;               ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = sub_traits::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = sub_traits::is_direct && sub_traits::storage_order == order::RowMajor;               ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = sub_traits::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr order storage_order          = sub_traits::storage_order;           ///< The expression's storage order

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
    using value_type = typename etl_traits<sub_expr_t>::value_type; ///< The value type

    static constexpr bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = false;                                           ///< Indicates if the expression is fast
    static constexpr bool is_linear               = etl_traits<sub_expr_t>::is_linear;               ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = etl_traits<sub_expr_t>::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = etl_traits<sub_expr_t>::is_direct;               ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr order storage_order          = etl_traits<sub_expr_t>::storage_order;           ///< The expression's storage order

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
 * \brief Print a representation of the view on the given stream
 * \param os The output stream
 * \param v The view to print
 * \return the output stream
 */
template <typename T, std::size_t D>
std::ostream& operator<<(std::ostream& os, const dim_view<T, D>& v) {
    return os << "dim[" << D << "](" << v.sub << ", " << v.i << ")";
}

} //end of namespace etl
