//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the fast_matrix_view implementation.
 */

#pragma once

namespace etl {

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

template <typename T, bool DMA, std::size_t... Dims>
struct fast_matrix_view;

/*!
 * \brief View to represent a fast matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 * \tparam Dims The dimensios of the view
 */
template <typename T, std::size_t... Dims>
struct fast_matrix_view <T, false, Dims...> final :
    iterable<fast_matrix_view<T, Dims...>, false>
{
    using sub_type          = T;                                                                    ///< The sub type
    using value_type        = value_t<sub_type>;                                                    ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view

    sub_type sub; ///< The Sub expression

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
    template <typename... S, cpp_enable_if((sizeof...(S) == sizeof...(Dims)))>
    return_type operator()(S... args) noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return sub[index(static_cast<std::size_t>(args)...)];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_if((sizeof...(S) == sizeof...(Dims)))>
    const_return_type operator()(S... args) const noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return sub[index(static_cast<std::size_t>(args)...)];
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_if(B)>
    auto operator()(size_t i) noexcept {
        return etl::sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_if(B)>
    auto operator()(size_t i) const noexcept {
        return etl::sub(*this, i);
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
    void visit(const detail::gpu_clean_visitor& visitor){
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
 * \brief View to represent a fast matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 * \tparam Dims The dimensios of the view
 */
template <typename T, std::size_t... Dims>
struct fast_matrix_view <T, true, Dims...> final :
    iterable<fast_matrix_view<T, true, Dims...>, true>
{
    using this_type         = fast_matrix_view<T, true, Dims...>;                                   ///< The type of this expression
    using sub_type          = T;                                                                    ///< The sub type
    using value_type        = value_t<sub_type>;                                                    ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view

    sub_type sub; ///< The Sub expression

    mutable memory_type memory;

    //TODO Should be shared with the sub expression
    mutable gpu_handler<value_type> _gpu_memory_handler; ///< The GPU memory handler

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
    explicit fast_matrix_view(sub_type sub): sub(sub) {
        if(!decay_traits<sub_type>::needs_evaluator_visitor){
            this->memory = sub.memory_start();
        }
    }

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
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        return memory[j];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_if((sizeof...(S) == sizeof...(Dims)))>
    return_type operator()(S... args) noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return memory[index(static_cast<std::size_t>(args)...)];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_if((sizeof...(S) == sizeof...(Dims)))>
    const_return_type operator()(S... args) const noexcept {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return memory[index(static_cast<std::size_t>(args)...)];
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_if(B)>
    auto operator()(size_t i) noexcept {
        return etl::sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_if(B)>
    auto operator()(size_t i) const noexcept {
        return etl::sub(*this, i);
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
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        return sub.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return sub.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return sub.memory_end();
    }

    /*!
     * \brief Return an opaque (type-erased) access to the memory of the matrix
     * \return a structure containing the dimensions, the storage order and the memory pointers of the matrix
     */
    opaque_memory<value_type, sizeof...(Dims)> direct() const {
        return {memory, mul_all<Dims...>::value, {{Dims...}}, _gpu_memory_handler, decay_traits<this_type>::storage_order};
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
    void visit(const detail::gpu_clean_visitor& visitor){
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
 * \brief Specialization for fast_matrix_view.
 */
template <typename T, bool DMA, std::size_t... Dims>
struct etl_traits<etl::fast_matrix_view<T, DMA, Dims...>> {
    using expr_t     = etl::fast_matrix_view<T, DMA, Dims...>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;                        ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                 ///< The sub traits
    using value_type = typename sub_traits::value_type;        ///< The value type

    static constexpr bool is_etl                  = true;                                ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                               ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = true;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                               ///< Indicates if the type is a magic view
    static constexpr bool is_linear               = sub_traits::is_linear;               ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = sub_traits::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr bool is_fast                 = true;                                ///< Indicates if the expression is fast
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
 * \brief Print a representation of the view on the given stream
 * \param os The output stream
 * \param v The view to print
 * \return the output stream
 */
template <typename T, bool DMA, std::size_t Rows, std::size_t Columns>
std::ostream& operator<<(std::ostream& os, const fast_matrix_view<T, DMA, Rows, Columns>& v) {
    return os << "reshape[" << Rows << "," << Columns << "](" << v.sub << ")";
}

} //end of namespace etl
