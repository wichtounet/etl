//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
    static constexpr bool vectorizable = true;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = false;
};

/*!
 * \brief Simple unary op for transform operations
 *
 * Such an unary expr does not apply the operator but delegates to its sub expression.
 */
struct transform_op {
    static constexpr bool linear      = false; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename Expr>
    static constexpr bool gpu_computable = Expr::gpu_computable;
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
    static constexpr bool vectorizable = Sub::template vectorizable<V>;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable = Sub::template gpu_computable<E>;
};

/*!
 * \brief An unary expression
 *
 * This expression applies an unary operator on each element of a sub expression
 */
template <typename T, typename Expr, typename UnaryOp>
struct unary_expr final : value_testable<unary_expr<T, Expr, UnaryOp>>, dim_testable<unary_expr<T, Expr, UnaryOp>>, iterable<unary_expr<T, Expr, UnaryOp>> {
private:
    static_assert(is_etl_expr<Expr> || std::is_same_v<Expr, etl::scalar<T>>, "Only ETL expressions can be used in unary_expr");

    using this_type = unary_expr<T, Expr, UnaryOp>; ///< The type of this expression

    Expr value; ///< The sub expression

    friend struct etl_traits<unary_expr>;
    friend struct optimizer<unary_expr>;
    friend struct optimizable<unary_expr>;
    friend struct transformer<unary_expr>;

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
    using vec_type = typename V::template vec_type<T>;

    /*!
     * \brief Construct a new unary_expr with the given sub expression
     * \param l The sub expression
     */
    explicit unary_expr(Expr l) : value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs)     = default;
    unary_expr(unary_expr&& rhs) noexcept = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& rhs) = delete;
    unary_expr& operator=(unary_expr&& rhs) = delete;

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](size_t i) const {
        return UnaryOp::apply(value[i]);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const noexcept {
        return UnaryOp::apply(value.read_flat(i));
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> load(size_t i) const {
        return UnaryOp::template load<V>(value.template load<V>(i));
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> loadu(size_t i) const {
        return UnaryOp::template load<V>(value.template loadu<V>(i));
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return UnaryOp::apply(value(args...));
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return value.alias(rhs);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    decltype(auto) gpu_compute_hint(Y& y) const {
        return UnaryOp::gpu_compute_hint(value, y);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    decltype(auto) gpu_compute(Y& y) const {
        return UnaryOp::gpu_compute(value, y);
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        value.visit(visitor);
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // The sub value must be ensured
        value.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // The sub value must be ensured
        value.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Prints the type of the unary expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const unary_expr& expr) {
        return os << UnaryOp::desc() << '(' << expr.value << ')';
    }
};

/*!
 * \brief Specialization of unary expression for identity op.
 *
 * This unary expression keeps access to data (can be edited)
 */
template <typename T, typename Expr>
struct unary_expr<T, Expr, identity_op> : inplace_assignable<unary_expr<T, Expr, identity_op>>,
                                          assignable<unary_expr<T, Expr, identity_op>, T>,
                                          value_testable<unary_expr<T, Expr, identity_op>>,
                                          dim_testable<unary_expr<T, Expr, identity_op>>,
                                          iterable<unary_expr<T, Expr, identity_op>, is_dma<Expr>> {
private:
    static_assert(is_etl_expr<Expr>, "Only ETL expressions can be used in unary_expr");

    Expr value;                 ///< The sub expression
    gpu_memory_handler<T> _gpu; ///< The GPU memory handler

    static constexpr bool dma = is_dma<Expr>; ///< Indicates if the unary expression has direct memory access

    /*!
     * \brief Indicates if the non-const functions returns a reference
     */
    static constexpr bool non_const_return_ref =
        std::is_lvalue_reference_v<decltype(value[0])> && !std::is_const_v<std::remove_reference_t<decltype(value[0])>>;

    /*!
     * \brief Indicates if the const functions returns a reference
     */
    static constexpr bool const_return_ref = std::is_lvalue_reference_v<decltype(value[0])>;

    friend struct etl_traits<unary_expr>;
    friend struct optimizer<unary_expr>;
    friend struct optimizable<unary_expr>;
    friend struct transformer<unary_expr>;

public:
    using this_type            = unary_expr<T, Expr, identity_op>;                                           ///< The type of this expression
    using value_type           = T;                                                                          ///< The value type
    using assignable_base_type = assignable<this_type, value_type>;                                          ///< The assignable base type
    using memory_type          = memory_t<Expr>;                                                             ///< The memory type
    using const_memory_type    = const_memory_t<Expr>;                                                       ///< The const memory type
    using return_type          = std::conditional_t<non_const_return_ref, value_type&, value_type>;          ///< The type returned by the functions
    using const_return_type    = std::conditional_t<const_return_ref, const value_type&, value_type>;        ///< The const type returned by the const functions
    using expr_t               = Expr;                                                                       ///< The sub expression type
    using iterator             = std::conditional_t<dma, value_type*, etl::iterator<this_type>>;             ///< The iterator type
    using const_iterator       = std::conditional_t<dma, const value_type*, etl::iterator<const this_type>>; ///< The const iterator type

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<value_type>;

    using assignable_base_type::operator=;

    /*!
     * \brief Construct a new unary expression
     * \param l The sub expression
     */
    explicit unary_expr(Expr l) noexcept : value(std::forward<Expr>(l)) {
        //Nothing to init
    }

    /*!
     * \brief Copy construct an unary expression
     */
    unary_expr(const unary_expr& rhs) = default;

    /*!
     * \brief Move construct an unary expression
     */
    unary_expr(unary_expr&& rhs) noexcept = default;

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](size_t i) {
        return value[i];
    }

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](size_t i) const {
        return value[i];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const noexcept {
        return value.read_flat(i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, size_t i) noexcept {
        return value.template store<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, size_t i) noexcept {
        return value.template storeu<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, size_t i) noexcept {
        return value.template stream<V>(in, i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>)
    load(size_t i) const noexcept {
        return value.template load<V>(i);
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>)
    loadu(size_t i) const noexcept {
        return value.template loadu<V>(i);
    }

    /*!
     * \brief Returns a reference to the ith dimension value.
     *
     * This should only be used internally and with care
     *
     * \return a refernece to the ith dimension value.
     */
    size_t& unsafe_dimension_access(size_t i) {
        return value.unsafe_dimension_access(i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (safe_dimensions<this_type>> 1), cpp_enable_iff(B)>
    auto operator()(size_t i) {
        return sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (safe_dimensions<this_type>> 1), cpp_enable_iff(B)>
    auto operator()(size_t i) const {
        return sub(*this, i);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(size_t first, size_t last) noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(size_t first, size_t last) const noexcept {
        return etl::slice(*this, first, last);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == safe_dimensions<this_type>)>
    ETL_STRONG_INLINE(return_type)
    operator()(S... args) noexcept(noexcept(value(args...))) {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return value(args...);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == safe_dimensions<this_type>)>
    ETL_STRONG_INLINE(const_return_type)
    operator()(S... args) const noexcept(noexcept(value(args...))) {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return value(args...);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        if constexpr (all_dma<E, Expr>) {
            return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
        } else {
            return value.alias(rhs);
        }
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        return value.memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        return value.memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        return value.memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        return value.memory_end();
    }

    /*!
     * \brief Returns the DDth dimension of the matrix
     * \return The DDth dimension of the matrix
     */
    template <size_t DD, typename TT = this_type, cpp_enable_iff(is_fast<TT>)>
    static constexpr size_t dim() {
        return etl_traits<TT>::template dim<DD>();
    }

    /*!
     * \brief Returns the DDth dimension of the matrix
     * \return The DDth dimension of the matrix
     */
    template <size_t DD, typename TT = this_type, cpp_disable_iff(is_fast<TT>)>
    size_t dim() const {
        return etl_traits<TT>::dim(*this, DD);
    }

    /*!
     * \brief Returns all the Ith... dimensions in array
     * \return an array containing the Ith... dimensions of the expression.
     */
    template <size_t... I>
    std::array<size_t, decay_traits<this_type>::dimensions()> dim_array(std::index_sequence<I...>) const {
        return {{this->template dim<I>()...}};
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        value.visit(visitor);
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    T* gpu_memory() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        return _gpu.gpu_memory();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        _gpu.gpu_evict();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        _gpu.invalidate_cpu();
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        _gpu.invalidate_gpu();
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        _gpu.validate_cpu();
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        _gpu.validate_gpu();
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        _gpu.ensure_gpu_allocated(etl::size(value));
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        if constexpr (is_dma<Expr>) {
            _gpu.ensure_gpu_up_to_date(memory_start(), etl::size(value));
        } else {
            value.ensure_gpu_up_to_date();
        }
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        if constexpr (is_dma<Expr>) {
            _gpu.ensure_cpu_up_to_date(memory_start(), etl::size(value));
        } else {
            value.ensure_cpu_up_to_date();
        }
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const T* gpu_memory) const {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        _gpu.gpu_copy_from(gpu_memory, etl::size(value));
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        return _gpu.is_cpu_up_to_date();
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        static_assert(is_dma<Expr>, "This expression does not have direct memory access");
        return _gpu.is_gpu_up_to_date();
    }

    /*!
     * \brief Prints the type of the unary expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const unary_expr& expr) {
        return os << expr.value;
    }

private:
    /*!
     * \brief Assign the given value to each eleemnt of the unary expression
     * \param e The value
     */
    void memory_set(const value_type& e) {
        if constexpr (is_dma<Expr>) {
            direct_fill(memory_start(), memory_end(), e);
        } else {
            for (size_t i = 0; i < size(*this); ++i) {
                (*this)[i] = e;
            }
        }
    }
};

/*!
 * \brief Specialization of unary expression for transform op.
 */
template <typename T, typename Expr>
struct unary_expr<T, Expr, transform_op>
        : value_testable<unary_expr<T, Expr, transform_op>>, dim_testable<unary_expr<T, Expr, transform_op>>, iterable<unary_expr<T, Expr, transform_op>> {
private:
    using this_type = unary_expr<T, Expr, transform_op>; ///< The type of this expression

    Expr value; ///< The sub expression

    friend struct etl_traits<unary_expr>;
    friend struct optimizer<unary_expr>;
    friend struct optimizable<unary_expr>;
    friend struct transformer<unary_expr>;

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
    explicit unary_expr(Expr l) : value(std::forward<Expr>(l)) {
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs)     = default;
    unary_expr(unary_expr&& rhs) noexcept = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& e) = delete;
    unary_expr& operator=(unary_expr&& e) = delete;

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](size_t i) const {
        return value[i];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const noexcept {
        return value.read_flat(i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (safe_dimensions<this_type>> 1), cpp_enable_iff(B)>
    auto operator()(size_t i) const {
        return sub(*this, i);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    std::enable_if_t<sizeof...(S) == safe_dimensions<this_type>, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return value(args...);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return value.alias(rhs);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    decltype(auto) gpu_compute_hint(Y& y) const {
        return value.gpu_compute_hint(y);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    decltype(auto) gpu_compute(Y& y) const {
        return value.gpu_compute(y);
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        value.visit(visitor);
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // The sub value must be ensured
        value.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // The sub value must be ensured
        value.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Prints the type of the unary expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const unary_expr& expr) {
        return os << expr.value;
    }
};

/*!
 * \brief Specialization of unary expression for stateful op.
 *
 * This operator has some state and is constructed directly inside the expression
 */
template <typename T, typename Expr, typename Op>
struct unary_expr<T, Expr, stateful_op<Op>> : value_testable<unary_expr<T, Expr, stateful_op<Op>>>,
                                              dim_testable<unary_expr<T, Expr, stateful_op<Op>>>,
                                              iterable<unary_expr<T, Expr, stateful_op<Op>>> {
private:
    using this_type = unary_expr<T, Expr, stateful_op<Op>>; ///< The type of this expression

    Expr value; ///< The sub expression
    Op op;      ///< The operator state

    friend struct etl_traits<unary_expr>;
    friend struct optimizer<unary_expr>;
    friend struct optimizable<unary_expr>;
    friend struct transformer<unary_expr>;

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
    using vec_type = typename V::template vec_type<T>;

    /*!
     * \brief Construct a new unary_expr from the given sub-expression and construct the op by forwarding it the given arguments
     * \param l The sub expression
     * \param args The arguments to forward to the op constructor
     */
    template <typename... Args>
    explicit unary_expr(Expr l, Args&&... args) : value(std::forward<Expr>(l)), op(std::forward<Args>(args)...) {
        //Nothing else to init
    }

    unary_expr(const unary_expr& rhs)     = default;
    unary_expr(unary_expr&& rhs) noexcept = default;

    //Expression are invariant
    unary_expr& operator=(const unary_expr& e) = delete;
    unary_expr& operator=(unary_expr&& e) = delete;

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](size_t i) const {
        return op.apply(value[i]);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return op.apply(value.read_flat(i));
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> load(size_t i) const {
        return op.template load<V>(value.template load<V>(i));
    }

    /*!
     * \brief Load several elements of the matrix at once
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the matrix
     */
    template <typename V = default_vec>
    vec_type<V> loadu(size_t i) const {
        return op.template load<V>(value.template loadu<V>(i));
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    decltype(auto) gpu_compute_hint(Y& y) const {
        return op.gpu_compute_hint(value, y);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    decltype(auto) gpu_compute(Y& y) const {
        return op.gpu_compute(value, y);
    }

    /*!
     * \brief Returns the value at the position (args...)
     * \param args The indices
     * \return The computed value at the position (args...)
     */
    template <typename... S>
    std::enable_if_t<sizeof...(S) == safe_dimensions<this_type>, value_type> operator()(S... args) const {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return op.apply(value(args...));
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return value.alias(rhs);
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        value.visit(visitor);
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // The sub value must be ensured
        value.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // The sub value must be ensured
        value.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Prints the type of the unary expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const unary_expr& expr) {
        return os << Op::desc() << '(' << expr.value << ')';
    }
};

/*!
 * \brief Specialization unary_expr
 */
template <typename T, typename Expr, typename UnaryOp>
struct etl_traits<etl::unary_expr<T, Expr, UnaryOp>> {
    using expr_t     = etl::unary_expr<T, Expr, UnaryOp>; ///< The expression type
    using sub_expr_t = std::decay_t<Expr>;                ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;            ///< The traits of the sub expression
    using value_type = T;                                 ///< The value type

    static constexpr bool is_etl         = true;                ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;               ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;               ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;               ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast; ///< Indicates if the expression is fast
    static constexpr bool is_value       = false;               ///< Indicates if the expression is of value type
    static constexpr bool is_direct =
        std::is_same_v<UnaryOp, identity_op> && sub_traits::is_direct;                    ///< Indicates if the expression has direct memory access
    static constexpr bool is_linear      = sub_traits::is_linear && UnaryOp::linear;           ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = sub_traits::is_thread_safe && UnaryOp::thread_safe; ///< Indicates if the expression is linear
    static constexpr bool is_generator   = sub_traits::is_generator;                           ///< Indicates if the expression is a generator expression
    static constexpr bool is_temporary   = sub_traits::is_temporary;                           ///< Indicaes if the expression needs an evaluator visitor
    static constexpr bool is_padded      = is_linear && sub_traits::is_padded;                 ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = is_linear && sub_traits::is_aligned;                ///< Indicates if the expression is padded
    static constexpr order storage_order = sub_traits::storage_order;                          ///< The expression storage order

    /*!
     * \brief Indicates if the expression can be computed on GPU
     */
    static constexpr bool gpu_computable = is_gpu_computable<Expr> && UnaryOp::template gpu_computable<Expr> && (is_floating<Expr> || is_complex<Expr>);

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = sub_traits::template vectorizable<V>&& UnaryOp::template vectorizable<V>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) {
        return sub_traits::size(v.value);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, size_t d) {
        return sub_traits::dim(v.value, d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() {
        return sub_traits::size();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <size_t D>
    static constexpr size_t dim() {
        return sub_traits::template dim<D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return sub_traits::dimensions();
    }
};

} //end of namespace etl
