//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains unary expression implementations
 */

#pragma once

namespace etl {

/*!
 * \brief Simple unary op for identity operations
 *
 * Such an unary expr does not apply the operator but delegates to its sub expression.
 */
struct identity_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;
};

/*!
 * \brief Simple unary op for transform operations
 *
 * Such an unary expr does not apply the operator but delegates to its sub expression.
 */
struct transform_op {
    static constexpr bool linear = false; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::false_type;
};

/*!
 * \brief Simple unary op for stateful operations
 *
 * The sub operation (the real unary operation) is constructed from the
 * arguments of the unary_expr constructor. Such an unary expr does not apply
 * the operator but delegates to its sub expression.
 */
template <typename Sub>
struct stateful_op {
    static constexpr bool linear      = Sub::linear;      ///< Indicates if the operator is linear
    static constexpr bool thread_safe = Sub::thread_safe; ///< Indicates if the operator is thread safe

    using op = Sub; ///< The sub operator type

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = typename Sub::template vectorizable<V>;
};

/*!
 * \brief An unary expression
 *
 * This expression applies an unary operator on each element of a sub expression
 */
template <typename T, typename Expr, typename UnaryOp>
struct unary_expr final :
        comparable<unary_expr<T, Expr, UnaryOp>>,
        value_testable<unary_expr<T, Expr, UnaryOp>>,
        dim_testable<unary_expr<T, Expr, UnaryOp>>,
        iterable<unary_expr<T, Expr, UnaryOp>>
{
private:
    static_assert(
        is_etl_expr<Expr>::value || std::is_same<Expr, etl::scalar<T>>::value,
        "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, UnaryOp>; ///< The type of this expression

    Expr _value; ///< The sub expression

public:
    using value_type        = T;                              ///< The value type
    using memory_type       = void;                           ///< The memory type
    using const_memory_type = void;                           ///< The const memory type
    using expr_t            = Expr;                           ///< The sub expression type
    using iterator          = etl::iterator<this_type>;       ///< The iterator type
    using const_iterator    = etl::iterator<const this_type>; ///< The const iterator type

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Construct a new unary_expr with the given sub expression
     * \param l The sub expression
     */
    explicit unary_expr(Expr l)
            : _value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& rhs) = delete;
    unary_expr& operator=(unary_expr&& rhs) = delete;

    /*!
     * \brief Return the value on which this expression operates
     * \return The value on which this expression operates.
     */
    std::add_lvalue_reference_t<Expr> value() noexcept {
        return _value;
    }

    /*!
     * \brief Return the value on which this expression operates
     * \return The value on which this expression operates.
     */
    cpp::add_const_lvalue_t<Expr> value() const noexcept {
        return _value;
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) const {
        return UnaryOp::apply(value()[i]);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const noexcept {
        return UnaryOp::apply(value().read_flat(i));
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> load(std::size_t i) const {
        return UnaryOp::template load<V>(value().template load<V>(i));
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> loadu(std::size_t i) const {
        return UnaryOp::template load<V>(value().template loadu<V>(i));
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return UnaryOp::apply(value()(args...));
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return _value.alias(rhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template<typename V>
    void visit(V&& visitor){
        _value.visit(std::forward<V>(visitor));
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        _value.visit(visitor);
        visitor.need_value = old_need_value;
    }
};

/*!
 * \brief Specialization of unary expression for identity op.
 *
 * This unary expression keeps access to data (can be edited)
 */
template <typename T, typename Expr>
struct unary_expr<T, Expr, identity_op> :
        inplace_assignable<unary_expr<T, Expr, identity_op>>,
        comparable<unary_expr<T, Expr, identity_op>>,
        value_testable<unary_expr<T, Expr, identity_op>>,
        dim_testable<unary_expr<T, Expr, identity_op>>,
        iterable<unary_expr<T, Expr, identity_op>, has_direct_access<Expr>::value>
{
private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, identity_op>; ///< The type of this expression

    Expr _value; ///< The sub expression

    mutable gpu_handler<T> _gpu_memory_handler; ///< The GPU memory handler

    static constexpr bool dma = has_direct_access<Expr>::value;

    /*!
     * \brief Indicates if the non-const functions returns a reference
     */
    static constexpr bool non_const_return_ref =
        cpp::and_c<
            std::is_lvalue_reference<decltype(_value[0])>,
            cpp::not_c<std::is_const<std::remove_reference_t<decltype(_value[0])>>>>::value;

    /*!
     * \brief Indicates if the const functions returns a reference
     */
    static constexpr bool const_return_ref =
        std::is_lvalue_reference<decltype(_value[0])>::value; ///< Indicates if the const functions returns a reference

public:
    using value_type        = T;                                                                          ///< The value type
    using memory_type       = memory_t<Expr>;                                                             ///< The memory type
    using const_memory_type = const_memory_t<Expr>;                                                       ///< The const memory type
    using return_type       = std::conditional_t<non_const_return_ref, value_type&, value_type>;          ///< The type returned by the functions
    using const_return_type = std::conditional_t<const_return_ref, const value_type&, value_type>;        ///< The const type returned by the const functions
    using expr_t            = Expr;                                                                       ///< The sub expression type
    using iterator          = std::conditional_t<dma, value_type*, etl::iterator<this_type>>;             ///< The iterator type
    using const_iterator    = std::conditional_t<dma, const value_type*, etl::iterator<const this_type>>; ///< The const iterator type

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<value_type>;

    /*!
     * \brief Construct a new unary expression
     * \param l The sub expression
     */
    explicit unary_expr(Expr l) noexcept : _value(std::forward<Expr>(l)) {
        //Nothing to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    /*!
     * \brief Assign the given expression to the unary expression
     * \param e The expression to get the values from
     * \return the unary expression
     */
    template <typename E, cpp_enable_if(non_const_return_ref, is_etl_expr<E>::value)>
    unary_expr& operator=(E&& e) {
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
        return *this;
    }

    /*!
     * \brief Assign the given value to each eleemnt of the unary expression
     * \param e The value
     * \return the unary expression
     */
    unary_expr& operator=(const value_type& e) {
        static_assert(non_const_return_ref, "Impossible to modify read-only unary_expr");

        memory_set(e);

        return *this;
    }

    /*!
     * \brief Assign the given container to the unary expression
     * \param vec The container to get the values from
     * \return the unary expression
     */
    template <typename Container, cpp_enable_if(!is_etl_expr<Container>::value, std::is_convertible<typename Container::value_type, value_type>::value)>
    unary_expr& operator=(const Container& vec) {
        validate_assign(*this, vec);

        for (std::size_t i = 0; i < size(*this); ++i) {
            (*this)[i] = vec[i];
        }

        return *this;
    }

    /*!
     * \brief Return the value on which this expression operates
     * \return The value on which this expression operates.
     */
    std::add_lvalue_reference_t<Expr> value() noexcept {
        return _value;
    }

    /*!
     * \brief Return the value on which this expression operates
     * \return The value on which this expression operates.
     */
    cpp::add_const_lvalue_t<Expr> value() const noexcept {
        return _value;
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t i) {
        return value()[i];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t i) const {
        return value()[i];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const noexcept {
        return value().read_flat(i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t i) noexcept {
        return value().template store<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t i) noexcept {
        return value().template storeu<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t i) noexcept {
        return value().template stream<V>(in, i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>) load(std::size_t i) const noexcept {
        return _value.template load<V>(i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>) loadu(std::size_t i) const noexcept {
        return _value.template loadu<V>(i);
    }

    /*!
     * \brief Returns a reference to the ith dimension value.
     *
     * This should only be used internally and with care
     *
     * \return a refernece to the ith dimension value.
     */
    std::size_t& unsafe_dimension_access(std::size_t i) {
        return value().unsafe_dimension_access(i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (sub_size_compare<this_type>::value > 1), cpp_enable_if(B)>
    auto operator()(std::size_t i) {
        return sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (sub_size_compare<this_type>::value > 1), cpp_enable_if(B)>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(std::size_t first, std::size_t last) noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(std::size_t first, std::size_t last) const noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S, cpp_enable_if((sizeof...(S) == sub_size_compare<this_type>::value))>
    ETL_STRONG_INLINE(return_type) operator()(S... args) noexcept(noexcept(_value(args...))) {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return _value(args...);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S, cpp_enable_if((sizeof...(S) == sub_size_compare<this_type>::value))>
    ETL_STRONG_INLINE(const_return_type) operator()(S... args) const noexcept(noexcept(_value(args...))) {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return _value(args...);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E, cpp_enable_if(has_direct_access<Expr>::value, all_dma<E>::value)>
    bool alias(const E& rhs) const noexcept {
        return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E, cpp_disable_if(has_direct_access<Expr>::value&& all_dma<E>::value)>
    bool alias(const E& rhs) const noexcept {
        return _value.alias(rhs);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<Expr>::value, "This expression does not have direct memory access");
        return value().memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<Expr>::value, "This expression does not have direct memory access");
        return value().memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<Expr>::value, "This expression does not have direct memory access");
        return value().memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<Expr>::value, "This expression does not have direct memory access");
        return value().memory_end();
    }

    /*!
     * \brief Returns the DDth dimension of the matrix
     * \return The DDth dimension of the matrix
     */
    template <std::size_t DD, typename TT = this_type, cpp_enable_if(all_fast<TT>::value)>
    static constexpr std::size_t dim() {
        return etl_traits<TT>::template dim<DD>();
    }

    /*!
     * \brief Returns the DDth dimension of the matrix
     * \return The DDth dimension of the matrix
     */
    template <std::size_t DD, typename TT = this_type, cpp_disable_if(all_fast<TT>::value)>
    std::size_t dim() const {
        return etl_traits<TT>::dim(*this, DD);
    }

    /*!
     * \brief Returns all the Ith... dimensions in array
     * \return an array containing the Ith... dimensions of the expression.
     */
    template<std::size_t... I>
    std::array<std::size_t, decay_traits<this_type>::dimensions()> dim_array(std::index_sequence<I...>) const {
        return {{this->template dim<I>()...}};
    }

    /*!
     * \brief Return an opaque (type-erased) access to the memory of the matrix
     * \return a structure containing the dimensions, the storage order and the memory pointers of the matrix
     */
    opaque_memory<T, decay_traits<this_type>::dimensions()> direct() const {
        return {memory_start(), etl::size(*this),
            dim_array(std::make_index_sequence<decay_traits<this_type>::dimensions()>()),
            _gpu_memory_handler, decay_traits<this_type>::storage_order};
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template<typename V>
    void visit(V&& visitor){
        _value.visit(std::forward<V>(visitor));
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        _value.visit(visitor);
        visitor.need_value = old_need_value;
    }

private:
    /*!
     * \brief Assign the given value to each eleemnt of the unary expression
     * \param e The value
     */
    template<cpp_enable_if_cst(all_dma<Expr>::value)>
    void memory_set(const value_type& e){
        direct_fill(memory_start(), memory_end(), e);
    }

    /*!
     * \brief Assign the given value to each eleemnt of the unary expression
     * \param e The value
     */
    template<cpp_enable_if_cst(!all_dma<Expr>::value)>
    void memory_set(const value_type& e){
        for (std::size_t i = 0; i < size(*this); ++i) {
            (*this)[i] = e;
        }
    }
};

/*!
 * \brief Specialization of unary expression for transform op.
 */
template <typename T, typename Expr>
struct unary_expr<T, Expr, transform_op> :
        comparable<unary_expr<T, Expr, transform_op>>,
        value_testable<unary_expr<T, Expr, transform_op>>,
        dim_testable<unary_expr<T, Expr, transform_op>>,
        iterable<unary_expr<T, Expr, transform_op>>
{
private:
    using this_type = unary_expr<T, Expr, transform_op>; ///< The type of this expression

    Expr _value; ///< The sub expression

public:
    using value_type        = T;                              ///< The value type of the expression
    using memory_type       = void;                           ///< The memory type of the expression
    using const_memory_type = void;                           ///< The const memory type of the expression
    using expr_t            = Expr;                           ///< The sub expression type
    using iterator          = etl::iterator<this_type>;       ///< The iterator type
    using const_iterator    = etl::iterator<const this_type>; ///< The const iterator type

    /*!
     * \brief Construct a new unary_expr from the given sub-expression
     * \param l The sub expression
     */
    explicit unary_expr(Expr l)
            : _value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) noexcept = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& e) = delete;
    unary_expr& operator=(unary_expr&& e) = delete;

    /*!
     * \brief Return the value on which this expression operates
     * \return The value on which this expression operates.
     */
    std::add_lvalue_reference_t<Expr> value() noexcept {
        return _value;
    }

    /*!
     * \brief Return the value on which this expression operates
     * \return The value on which this expression operates.
     */
    cpp::add_const_lvalue_t<Expr> value() const noexcept {
        return _value;
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) const {
        return value()[i];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const noexcept {
        return value().read_flat(i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (sub_size_compare<this_type>::value > 1), cpp_enable_if(B)>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return _value.alias(rhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template<typename V>
    void visit(V&& visitor){
        _value.visit(std::forward<V>(visitor));
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        _value.visit(visitor);
        visitor.need_value = old_need_value;
    }
};

/*!
 * \brief Specialization of unary expression for stateful op.
 *
 * This operator has some state and is constructed directly inside the expression
 */
template <typename T, typename Expr, typename Op>
struct unary_expr<T, Expr, stateful_op<Op>> :
        comparable<unary_expr<T, Expr, stateful_op<Op>>>,
        value_testable<unary_expr<T, Expr, stateful_op<Op>>>,
        dim_testable<unary_expr<T, Expr, stateful_op<Op>>>,
        iterable<unary_expr<T, Expr, stateful_op<Op>>>
{
private:
    using this_type = unary_expr<T, Expr, stateful_op<Op>>; ///< The type of this expression

    Expr _value; ///< The sub expression
    Op op;       ///< The operator state

public:
    using value_type        = T;                              ///< The value type
    using memory_type       = void;                           ///< The memory type
    using const_memory_type = void;                           ///< The const memory type
    using expr_t            = Expr;                           ///< The sub expression type
    using iterator          = etl::iterator<this_type>;       ///< The iterator type
    using const_iterator    = etl::iterator<const this_type>; ///< The const iterator type

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Construct a new unary_expr from the given sub-expression and construct the op by forwarding it the given arguments
     * \param l The sub expression
     * \param args The arguments to forward to the op constructor
     */
    template <typename... Args>
    explicit unary_expr(Expr l, Args&&... args)
            : _value(std::forward<Expr>(l)), op(std::forward<Args>(args)...) {
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& e) = delete;
    unary_expr& operator=(unary_expr&& e) = delete;

    /*!
     * \brief Return the value on which this expression operates
     * \return The value on which this expression operates.
     */
    std::add_lvalue_reference_t<Expr> value() noexcept {
        return _value;
    }

    /*!
     * \brief Return the value on which this expression operates
     * \return The value on which this expression operates.
     */
    cpp::add_const_lvalue_t<Expr> value() const noexcept {
        return _value;
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) const {
        return op.apply(value()[i]);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        return op.apply(value().read_flat(i));
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> load(std::size_t i) const {
        return op.template load<V>(value().template load<V>(i));
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> loadu(std::size_t i) const {
        return op.template load<V>(value().template loadu<V>(i));
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return op.apply(value()(args...));
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return _value.alias(rhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    template<typename V>
    void visit(V&& visitor){
        _value.visit(std::forward<V>(visitor));
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        _value.visit(visitor);
        visitor.need_value = old_need_value;
    }
};

/*!
 * \brief Specialization unary_expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct etl_traits<etl::unary_expr<T, Expr, UnaryOp>> {
    using expr_t     = etl::unary_expr<T, Expr, UnaryOp>; ///< The expression type
    using sub_expr_t = std::decay_t<Expr>;                ///< The sub expression type

    static constexpr bool is_etl                  = true;                                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                                                ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = etl_traits<sub_expr_t>::is_fast;                      ///< Indicates if the expression is fast
    static constexpr bool is_value                = false;                                                ///< Indicates if the expression is of value type
    static constexpr bool is_direct                = std::is_same<UnaryOp, identity_op>::value && etl_traits<sub_expr_t>::is_direct;                                                ///< Indicates if the expression has direct memory access
    static constexpr bool is_linear               = etl_traits<sub_expr_t>::is_linear && UnaryOp::linear; ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = etl_traits<sub_expr_t>::is_thread_safe && UnaryOp::thread_safe;                                                                ///< Indicates if the expression is linear
    static constexpr bool is_generator            = etl_traits<sub_expr_t>::is_generator;                 ///< Indicates if the expression is a generator expression
    static constexpr bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;      ///< Indicates if the expression needs a temporary visitor
    static constexpr bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;      ///< Indicaes if the expression needs an evaluator visitor
    static constexpr bool is_padded                = is_linear && etl_traits<sub_expr_t>::is_padded;                                                                                ///< Indicates if the expression is padded
    static constexpr bool is_aligned                = is_linear && etl_traits<sub_expr_t>::is_aligned;                                                                                ///< Indicates if the expression is padded
    static constexpr order storage_order          = etl_traits<sub_expr_t>::storage_order;                ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = cpp::bool_constant<
        etl_traits<sub_expr_t>::template vectorizable<V>::value && UnaryOp::template vectorizable<V>::value>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(v.value());
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(v.value(), d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Prints the type of the unary expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename T, typename Expr, typename UnaryOp>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, stateful_op<UnaryOp>>& expr) {
    return os << UnaryOp::desc() << '(' << expr.value() << ')';
}

/*!
 * \brief Prints the type of the unary expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename T, typename Expr>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, identity_op>& expr) {
    return os << expr.value();
}

/*!
 * \brief Prints the type of the unary expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename T, typename Expr>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, transform_op>& expr) {
    return os << expr.value();
}

/*!
 * \brief Prints the type of the unary expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename T, typename Expr, typename UnaryOp>
std::ostream& operator<<(std::ostream& os, const unary_expr<T, Expr, UnaryOp>& expr) {
    return os << UnaryOp::desc() << '(' << expr.value() << ')';
}

} //end of namespace etl
