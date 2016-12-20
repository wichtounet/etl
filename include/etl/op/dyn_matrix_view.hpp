//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Implementation of dyn_matrix_view
 */

#pragma once

namespace etl {

namespace detail {

/*!
 * \brief Return the flat index for the element at the given position
 * \param sizes The indices
 * \return The flat index
 */
template <order O, typename Dim, typename... S>
std::size_t index(const Dim& dimensions, size_t size, S... sizes) noexcept(assert_nothrow) {
    //Note: Version with sizes moved to a std::array and accessed with
    //standard loop may be faster, but need some stack space (relevant ?)

    std::size_t index = 0;

    if (O == order::RowMajor) {
        std::size_t subsize = size;
        std::size_t i       = 0;

        cpp::for_each_in(
            [&subsize, &index, &i, &dimensions](std::size_t s) {
                cpp_assert(s < dimensions[i], "Out of bounds");
                subsize /= dimensions[i++];
                index += subsize * s;
            },
            sizes...);
    } else {
        std::size_t subsize = 1;
        std::size_t i       = 0;

        cpp::for_each_in(
            [&subsize, &index, &i, &dimensions](std::size_t s) {
                cpp_assert(s < dimensions[i], "Out of bounds");
                index += subsize * s;
                subsize *= dimensions[i++];
            },
            sizes...);
    }

    return index;
}

} // end of namespace detail

/*!
 * \brief View to represent a dyn matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T, size_t D, typename Enable>
struct dyn_matrix_view;

/*!
 * \brief View to represent a dyn matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T, size_t D>
struct dyn_matrix_view <T, D, std::enable_if_t<!all_dma<T>::value>> final : iterable<dyn_matrix_view<T, D>, false> {
    static_assert(is_etl_expr<T>::value, "dyn_matrix_view only works with ETL expressions");

    using this_type          = dyn_matrix_view<T, D>;                                                ///< The type of this expression
    using iterable_base_type = iterable<this_type, false>;                                           ///< The iterable base type
    using sub_type           = T;                                                                    ///< The sub type
    using value_type         = value_t<sub_type>;                                                    ///< The value contained in the expression
    using memory_type        = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type  = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type        = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type  = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view
    using iterator           = etl::iterator<this_type>;                                             ///< The iterator type
    using const_iterator     = etl::iterator<const this_type>;                                       ///< The const iterator type

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_type>;

    using iterable_base_type::begin;
    using iterable_base_type::end;

private:
    T sub;                                 ///< The sub expression
    std::array<std::size_t, D> dimensions; ///< The dimensions of the view
    size_t _size;                          ///< The size of the view

    static constexpr order storage_order = decay_traits<sub_type>::storage_order; ///< The matrix storage order

    friend struct etl_traits<etl::dyn_matrix_view<T, D>>;

public:
    /*!
     * \brief Construct a new dyn_matrix_view over the given sub expression
     * \param dims The dimensions
     */
    template<typename... S>
    dyn_matrix_view(sub_type sub, S... dims)
            : sub(sub), dimensions{{dims...}}, _size(etl::size(sub)) {}

    /*!
     * \brief Assign the given expression to the unary expression
     * \param e The expression to get the values from
     * \return the unary expression
     */
    template <typename E, cpp_enable_if(is_etl_expr<E>::value)>
    dyn_matrix_view& operator=(E&& e) {
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
        return *this;
    }

    /*!
     * \brief Assign the given expression to the unary expression
     * \param v The expression to get the values from
     * \return the unary expression
     */
    dyn_matrix_view& operator=(const value_type& v) {
        std::fill(begin(), end(), v);
        return *this;
    }

    /*!
     * \brief Assign the given container to the unary expression
     * \param vec The container to get the values from
     * \return the unary expression
     */
    template <typename Container, cpp_enable_if(!is_etl_expr<Container>::value, std::is_convertible<typename Container::value_type, value_type>::value)>
    dyn_matrix_view& operator=(const Container& vec) {
        validate_assign(*this, vec);

        std::copy(vec.begin(), vec.end(), begin());

        return *this;
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
        return sub[detail::index<storage_order>(dimensions, _size, f1, f2, sizes...)];
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
        return sub[detail::index<storage_order>(dimensions, _size, f1, f2, sizes...)];
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

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        sub.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::gpu_clean_visitor& visitor){
        sub.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        sub.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        sub.visit(visitor);
        visitor.need_value = old_need_value;
    }

    /*!
     * \brief Print a representation of the view on the given stream
     * \param os The output stream
     * \param v The view to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const dyn_matrix_view& v) {
        return os << "reshape[" << D << "D](" << v.sub << ")";
    }
};

/*!
 * \brief View to represent a dyn matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T, size_t D>
struct dyn_matrix_view <T, D, std::enable_if_t<all_dma<T>::value>> final : iterable<dyn_matrix_view<T, D>, true> {
    static_assert(is_etl_expr<T>::value, "dyn_matrix_view only works with ETL expressions");

    using this_type          = dyn_matrix_view<T, D>;                                                ///< The type of this expression
    using iterable_base_type = iterable<this_type, true>;                                            ///< The iterable base type
    using sub_type           = T;                                                                    ///< The sub type
    using value_type         = value_t<sub_type>;                                                    ///< The value contained in the expression
    using memory_type        = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type  = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type        = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type  = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view
    using iterator           = value_type*;                                                          ///< The iterator type
    using const_iterator     = const value_type*;                                                    ///< The const iterator type

    /*!
     * \brief The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_type>;

    using iterable_base_type::begin;
    using iterable_base_type::end;

private:
    T sub;                                 ///< The sub expression
    std::array<std::size_t, D> dimensions; ///< The dimensions of the view
    size_t _size;                          ///< The size of the view

    mutable memory_type memory;

    //TODO Should be shared with the sub expression
    mutable gpu_handler<value_t<T>> _gpu_memory_handler; ///< The GPU memory handler

    static constexpr order storage_order = decay_traits<sub_type>::storage_order; ///< The matrix storage order

    friend struct etl_traits<etl::dyn_matrix_view<T, D>>;

public:
    /*!
     * \brief Construct a new dyn_matrix_view over the given sub expression
     * \param dims The dimensions
     */
    template<typename... S>
    dyn_matrix_view(sub_type sub, S... dims) : sub(sub), dimensions{{dims...}}, _size(etl::size(sub)) {
        if(!decay_traits<sub_type>::needs_evaluator_visitor){
            this->memory = sub.memory_start();
        }
    }

    /*!
     * \brief Assign the given expression to the unary expression
     * \param e The expression to get the values from
     * \return the unary expression
     */
    template <typename E, cpp_enable_if(is_etl_expr<E>::value)>
    dyn_matrix_view& operator=(E&& e) {
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
        return *this;
    }

    /*!
     * \brief Assign the given expression to the unary expression
     * \param v The expression to get the values from
     * \return the unary expression
     */
    dyn_matrix_view& operator=(const value_type& v) {
        std::fill(begin(), end(), v);
        return *this;
    }

    /*!
     * \brief Assign the given container to the unary expression
     * \param vec The container to get the values from
     * \return the unary expression
     */
    template <typename Container, cpp_enable_if(!is_etl_expr<Container>::value, std::is_convertible<typename Container::value_type, value_type>::value)>
    dyn_matrix_view& operator=(const Container& vec) {
        validate_assign(*this, vec);

        std::copy(vec.begin(), vec.end(), begin());

        return *this;
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        return memory[j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        return memory[j];
    }

    /*!
     * \brief Access to the element at the given position
     * \param j The index
     * \return a reference to the element at the given position.
     */
    const_return_type operator()(std::size_t j) const {
        return memory[j];
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
        return memory[detail::index<storage_order>(dimensions, _size, f1, f2, sizes...)];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        return memory[j];
    }

    /*!
     * \brief Access to the element at the given position
     * \param j The index
     * \return a reference to the element at the given position.
     */
    return_type operator()(std::size_t j) {
        return memory[j];
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
        return memory[detail::index<storage_order>(dimensions, _size, f1, f2, sizes...)];
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
        return V::loadu(memory + x);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t i) noexcept {
        return V::stream(memory + i, in);
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
        return V::storeu(memory + i, in);
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
        return memory;
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        return memory;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return memory + _size;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return memory + _size;
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        sub.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::gpu_clean_visitor& visitor){
        sub.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        sub.visit(visitor);

        // It's only interesting if the sub expression is not direct
        if(decay_traits<sub_type>::needs_evaluator_visitor){
            this->memory = sub.memory_start();
        }
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        sub.visit(visitor);
        visitor.need_value = old_need_value;
    }

    /*!
     * \brief Return an opaque (type-erased) access to the memory of the matrix
     * \return a structure containing the dimensions, the storage order and the memory pointers of the matrix
     */
    opaque_memory<value_type, decay_traits<this_type>::dimensions()> direct() const {
        return {memory, _size, dimensions, _gpu_memory_handler, decay_traits<this_type>::storage_order};
    }

    /*!
     * \brief Returns all the Ith... dimensions in array
     * \return an array containing the Ith... dimensions of the expression.
     */
    template<std::size_t... I>
    std::array<std::size_t, decay_traits<this_type>::dimensions()> dim_array(std::index_sequence<I...>) const {
        return dimensions;
    }

    /*!
     * \brief Print a representation of the view on the given stream
     * \param os The output stream
     * \param v The view to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const dyn_matrix_view& v) {
        return os << "reshape[" << D << "D](" << v.sub << ")";
    }
};

/*!
 * \brief Specialization for dyn_matrix_view.
 */
template <typename T, size_t D>
struct etl_traits<etl::dyn_matrix_view<T, D>> {
    using expr_t     = etl::dyn_matrix_view<T, D>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;            ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;     ///< The sub traits

    static constexpr bool is_etl                  = true;                                ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                               ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = true;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                               ///< Indicates if the type is a magic view
    static constexpr bool is_linear               = sub_traits::is_linear;               ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = sub_traits::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr bool is_fast                 = false;                               ///< Indicates if the expression is fast
    static constexpr bool is_value                = false;                               ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = sub_traits::is_direct;               ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                               ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                               ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = false;                               ///< Indicates if the expression is padded
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

} //end of namespace etl
