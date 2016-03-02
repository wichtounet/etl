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

#include "etl/iterator.hpp"

namespace etl {

/*!
 * \brief Simple unary op for identity operations
 *
 * Such an unary expr does not apply the operator but delegates to its sub expression.
 */
struct identity_op {
    static constexpr const bool linear       = true; ///< Indicates if the operator is linear

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<vector_mode_t V>
    using vectorizable = std::true_type;
};

/*!
 * \brief Simple unary op for transform operations
 *
 * Such an unary expr does not apply the operator but delegates to its sub expression.
 */
struct transform_op {
    static constexpr const bool linear       = false; ///< Indicates if the operator is linear

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<vector_mode_t V>
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
    static constexpr const bool linear       = Sub::linear;       ///< Indicates if the operator is linear
    using op                                 = Sub;               ///< The sub operator type

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<vector_mode_t V>
    using vectorizable = typename Sub::template vectorizable<V>;
};

/*!
 * \brief An unary expression
 *
 * This expression applies an unary operator on each element of a sub expression
 */
template <typename T, typename Expr, typename UnaryOp>
struct unary_expr final : comparable<unary_expr<T, Expr, UnaryOp>>, value_testable<unary_expr<T, Expr, UnaryOp>>, dim_testable<unary_expr<T, Expr, UnaryOp>> {
private:
    static_assert(
        is_etl_expr<Expr>::value || std::is_same<Expr, etl::scalar<T>>::value,
        "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, UnaryOp>; ///< The type of this expression

    Expr _value; ///< The sub expression

public:
    using value_type        = T;    ///< The value type
    using memory_type       = void; ///< The memory type
    using const_memory_type = void; ///< The const memory type
    using expr_t            = Expr; ///< The sub expression type

    /*!
     * The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type          = typename V::template vec_type<T>;

    //Construct a new expression
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
    template<typename V = default_vec>
    vec_type<V> load(std::size_t i) const {
        return UnaryOp::template load<V>(value().template load<V>(i));
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
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return _value.alias(rhs);
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    iterator<const this_type> begin() const noexcept {
        return {*this, 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    iterator<const this_type> end() const noexcept {
        return {*this, size(*this)};
    }
};

/*!
 * \brief Specialization of unary expression for identity op.
 *
 * This unary expression keeps access to data (can be edited)
 */
template <typename T, typename Expr>
struct unary_expr<T, Expr, identity_op> : inplace_assignable<unary_expr<T, Expr, identity_op>>, comparable<unary_expr<T, Expr, identity_op>>, value_testable<unary_expr<T, Expr, identity_op>>, dim_testable<unary_expr<T, Expr, identity_op>>, gpu_able<T, unary_expr<T, Expr, identity_op>> {
private:
    static_assert(is_etl_expr<Expr>::value, "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, identity_op>; ///< The type of this expression

    Expr _value; ///< The sub expression

    static constexpr const bool non_const_return_ref =
        cpp::and_c<
            std::is_lvalue_reference<decltype(_value[0])>,
            cpp::not_c<std::is_const<std::remove_reference_t<decltype(_value[0])>>>>::value;

    static constexpr const bool const_return_ref =
        std::is_lvalue_reference<decltype(_value[0])>::value;

public:
    using value_type        = T;                                                                   ///< The value type
    using memory_type       = memory_t<Expr>;                                                      ///< The memory type
    using const_memory_type = const_memory_t<Expr>;                                                ///< The const memory type
    using return_type       = std::conditional_t<non_const_return_ref, value_type&, value_type>;   ///< The type returned by the functions
    using const_return_type = std::conditional_t<const_return_ref, const value_type&, value_type>; ///< The const type returned by the const functions
    using expr_t            = Expr; ///< The sub expression type

    /*!
     * The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type          = typename V::template vec_type<T>;

    /*!
     * \brief Construct a new unary expression
     * \param l The sub expression
     */
    explicit unary_expr(Expr l) noexcept
            : _value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs) = default;
    unary_expr(unary_expr&& rhs) = default;

    /*!
     * \param Assign the given expression to the unary expression
     * \brief e The expression to get the values from
     * \return the unary expression
     */
    template <typename E, cpp_enable_if(non_const_return_ref, is_etl_expr<E>::value)>
    unary_expr& operator=(E&& e) {
        validate_assign(*this, e);
        assign_evaluate(std::forward<E>(e), *this);
        return *this;
    }

    /*!
     * \param Assign the given value to each eleemnt of the unary expression
     * \brief e The value
     * \return the unary expression
     */
    unary_expr& operator=(const value_type& e) {
        static_assert(non_const_return_ref, "Impossible to modify read-only unary_expr");

        //TODO: This is not necessarily efficient
        for (std::size_t i = 0; i < size(*this); ++i) {
            (*this)[i] = e;
        }

        return *this;
    }

    /*!
     * \param Assign the given container to the unary expression
     * \brief vec The container to get the values from
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
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    auto load(std::size_t i) const noexcept {
        return value().template load<V>(i);
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
    CPP_DEBUG_AUTO_TRICK auto slice(std::size_t first, std::size_t last) noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    CPP_DEBUG_AUTO_TRICK auto slice(std::size_t first, std::size_t last) const noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, return_type> operator()(S... args) {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    std::enable_if_t<sizeof...(S) == sub_size_compare<this_type>::value, const_return_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return value()(args...);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E, cpp_enable_if(has_direct_access<Expr>::value, all_dma<E>::value)>
    bool alias(const E& rhs) const noexcept {
        return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template<typename E, cpp_disable_if(has_direct_access<Expr>::value && all_dma<E>::value)>
    bool alias(const E& rhs) const noexcept {
        return _value.alias(rhs);
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return a iterator pointing to the first element of the matrix
     */
    iterator<this_type, non_const_return_ref, false> begin() noexcept {
        return {*this, 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return an iterator pointing to the past-the-end element of the matrix
     */
    iterator<this_type, non_const_return_ref, false> end() noexcept {
        return {*this, size(*this)};
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    iterator<const this_type, true> begin() const noexcept {
        return {*this, 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    iterator<const this_type, true> end() const noexcept {
        return {*this, size(*this)};
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
};

/*!
 * \brief Specialization of unary expression for transform op.
 */
template <typename T, typename Expr>
struct unary_expr<T, Expr, transform_op> : comparable<unary_expr<T, Expr, transform_op>>, value_testable<unary_expr<T, Expr, transform_op>>, dim_testable<unary_expr<T, Expr, transform_op>> {
private:
    using this_type = unary_expr<T, Expr, transform_op>; ///< The type of this expression

    Expr _value; ///< The sub expression

public:
    using value_type        = T;
    using memory_type       = void;
    using const_memory_type = void;
    using expr_t            = Expr;

    //Construct a new expression
    explicit unary_expr(Expr l)
            : _value(std::forward<Expr>(l)) {
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
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return _value.alias(rhs);
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return a const iterator pointing to the first element of the matrix
     */
    iterator<const this_type, false, false> begin() const noexcept {
        return {*this, 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    iterator<const this_type, false, false> end() const noexcept {
        return {*this, size(*this)};
    }
};


/*!
 * \brief Specialization of unary expression for stateful op.
 *
 * This operator has some state and is constructed directly inside the expression
 */
template <typename T, typename Expr, typename Op>
struct unary_expr<T, Expr, stateful_op<Op>> : comparable<unary_expr<T, Expr, stateful_op<Op>>>, value_testable<unary_expr<T, Expr, stateful_op<Op>>>, dim_testable<unary_expr<T, Expr, stateful_op<Op>>> {
private:
    using this_type = unary_expr<T, Expr, stateful_op<Op>>; ///< The type of this expression

    Expr _value; ///< The sub expression
    Op op; ///< The operator state

public:
    using value_type        = T; ///< The value type
    using memory_type       = void; ///< The memory type
    using const_memory_type = void; ///< The const memory type
    using expr_t            = Expr; ///< The sub expression type

    /*!
     * The vectorization type for V
     */
    template<typename V = default_vec>
    using vec_type          = typename V::template vec_type<T>;

    //Construct a new expression
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
    template<typename V = default_vec>
    vec_type<V> load(std::size_t i) const {
        return op.template load<V>(value().template load<V>(i));
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
    template<typename E>
    bool alias(const E& rhs) const noexcept {
        return _value.alias(rhs);
    }

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return a const iterator pointing to the first element of the matrix
     */
    iterator<const this_type, false, false> begin() const noexcept {
        return {*this, 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    iterator<const this_type, false, false> end() const noexcept {
        return {*this, size(*this)};
    }
};

/*!
 * \brief Specialization unary_expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct etl_traits<etl::unary_expr<T, Expr, UnaryOp>> {
    using expr_t     = etl::unary_expr<T, Expr, UnaryOp>; ///< The expression type
    using sub_expr_t = std::decay_t<Expr>; ///< The sub expression type

    static constexpr const bool is_etl                  = true;                                                          ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                                                         ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;                                                         ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                                         ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;                               ///< Indicates if the expression is fast
    static constexpr const bool is_value                = false;                                                         ///< Indicates if the expression is of value type
    static constexpr const bool is_linear               = etl_traits<sub_expr_t>::is_linear && UnaryOp::linear;          ///< Indicates if the expression is linear
    static constexpr const bool is_generator            = etl_traits<sub_expr_t>::is_generator;                          ///< Indicates if the expression is a generator expression
    static constexpr const bool needs_temporary_visitor = etl_traits<sub_expr_t>::needs_temporary_visitor;               ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<sub_expr_t>::needs_evaluator_visitor;               ///< Indicaes if the expression needs an evaluator visitor
    static constexpr const order storage_order          = etl_traits<sub_expr_t>::storage_order;                         ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template<vector_mode_t V>
    using vectorizable = cpp::bool_constant<
            etl_traits<sub_expr_t>::template vectorizable<V>::value
        &&  UnaryOp::template vectorizable<V>::value>;

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
