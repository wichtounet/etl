//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the fast_matrix_view implementation.
 */

#pragma once

#include "etl/index.hpp"

namespace etl {

template <typename T, bool DMA, size_t... Dims>
struct fast_matrix_view;

/*!
 * \brief View to represent a fast matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 * \tparam Dims The dimensios of the view
 */
template <typename T, size_t... Dims>
struct fast_matrix_view<T, false, Dims...> final : iterable<fast_matrix_view<T, false, Dims...>, false>,
                                                   value_testable<fast_matrix_view<T, false, Dims...>>,
                                                   assignable<fast_matrix_view<T, false, Dims...>, value_t<T>> {
    using sub_type             = T;                                                                    ///< The sub type
    using this_type            = fast_matrix<T, false, Dims...>;                                       ///< The type of this expression
    using value_type           = value_t<sub_type>;                                                    ///< The value contained in the expression
    using iterable_base_type   = iterable<this_type, false>;                                           ///< The iterable base type
    using assignable_base_type = assignable<this_type, value_type>;                                    ///< The assignable base type
    using memory_type          = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type    = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type          = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type    = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<value_type>;

    using assignable_base_type::operator=;
    using iterable_base_type::begin;
    using iterable_base_type::end;

private:
    sub_type sub; ///< The Sub expression

    static constexpr size_t n_dimensions = sizeof...(Dims); ///< The number of dimensions of the view

    friend struct etl_traits<fast_matrix_view>;

public:
    /*!
     * \brief Construct a new fast_matrix_view over the given sub expression
     * \param sub The sub expression
     */
    explicit fast_matrix_view(sub_type sub) : sub(sub) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](size_t j) const {
        return sub[j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](size_t j) {
        return sub[j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t j) const noexcept {
        return sub.read_flat(j);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == sizeof...(Dims))>
    return_type operator()(S... args) noexcept {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return sub[etl::fast_index<this_type>(static_cast<size_t>(args)...)];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == sizeof...(Dims))>
    const_return_type operator()(S... args) const noexcept {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return sub[etl::fast_index<this_type>(static_cast<size_t>(args)...)];
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_iff(B)>
    auto operator()(size_t i) noexcept {
        return etl::sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_iff(B)>
    auto operator()(size_t i) const noexcept {
        return etl::sub(*this, i);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(size_t x) const noexcept {
        return sub.template load<V>(x);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(size_t x) const noexcept {
        return sub.template loadu<V>(x);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, size_t i) noexcept {
        sub.template store<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, size_t i) noexcept {
        sub.template storeu<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, size_t i) noexcept {
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
        sub.visit(visitor);
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // Need to ensure sub value
        sub.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // Need to ensure both LHS and RHS
        sub.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Print a representation of the view on the given stream
     * \param os The output stream
     * \param v The view to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const fast_matrix_view& v) {
        return os << "reshape(" << v.sub << ")";
    }
};

/*!
 * \brief View to represent a fast matrix in top of an expression
 * \tparam T The type of expression on which the view is made
 * \tparam Dims The dimensios of the view
 */
template <typename T, size_t... Dims>
struct fast_matrix_view<T, true, Dims...> final : iterable<fast_matrix_view<T, true, Dims...>, true>,
                                                  value_testable<fast_matrix_view<T, true, Dims...>>,
                                                  assignable<fast_matrix_view<T, true, Dims...>, value_t<T>> {
    using this_type            = fast_matrix_view<T, true, Dims...>;                                   ///< The type of this expression
    using sub_type             = T;                                                                    ///< The sub type
    using value_type           = value_t<sub_type>;                                                    ///< The value contained in the expression
    using iterable_base_type   = iterable<this_type, true>;                                            ///< The iterable base type
    using assignable_base_type = assignable<this_type, value_type>;                                    ///< The assignable base type
    using memory_type          = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type    = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type          = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type    = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<value_type>;

    using assignable_base_type::operator=;
    using iterable_base_type::begin;
    using iterable_base_type::end;

private:
    sub_type sub; ///< The Sub expression

    mutable memory_type memory; ///< Pointer to the memory of expression

    static constexpr size_t n_dimensions = sizeof...(Dims); ///< The number of dimensions of the view

    friend struct etl_traits<fast_matrix_view>;

public:
    /*!
     * \brief Construct a new fast_matrix_view over the given sub expression
     * \param sub The sub expression
     */
    explicit fast_matrix_view(sub_type sub) : sub(sub) {
        // Accessing the memory through fast sub views means evaluation
        if constexpr (decay_traits<sub_type>::is_temporary) {
            standard_evaluator::pre_assign_rhs(*this);
        }

        this->memory = this->sub.memory_start();

        cpp_assert(this->memory, "Invalid memory");
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](size_t j) const {
        cpp_assert(memory, "Memory has not been initialized");
        ensure_cpu_up_to_date();
        return memory[j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](size_t j) {
        cpp_assert(memory, "Memory has not been initialized");
        ensure_cpu_up_to_date();
        invalidate_gpu();
        return memory[j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t j) const noexcept(assert_nothrow) {
        cpp_assert(memory, "Memory has not been initialized");
        ensure_cpu_up_to_date();
        return memory[j];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == sizeof...(Dims))>
    return_type operator()(S... args) noexcept(assert_nothrow) {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");
        cpp_assert(memory, "Memory has not been initialized");
        ensure_cpu_up_to_date();
        invalidate_gpu();
        return memory[etl::fast_index<this_type>(static_cast<size_t>(args)...)];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == sizeof...(Dims))>
    const_return_type operator()(S... args) const noexcept(assert_nothrow) {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");
        cpp_assert(memory, "Memory has not been initialized");
        ensure_cpu_up_to_date();
        return memory[etl::fast_index<this_type>(static_cast<size_t>(args)...)];
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_iff(B)>
    auto operator()(size_t i) noexcept {
        return etl::sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <bool B = (n_dimensions > 1), cpp_enable_iff(B)>
    auto operator()(size_t i) const noexcept {
        return etl::sub(*this, i);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(size_t x) const noexcept {
        return V::loadu(memory + x);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(size_t x) const noexcept {
        return V::loadu(memory + x);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, size_t i) noexcept {
        return V::stream(memory + i, in);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, size_t i) noexcept {
        sub.template store<V>(in, i);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param i The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, size_t i) noexcept {
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
        sub.visit(visitor);
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return sub.gpu_memory();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        sub.gpu_evict();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        sub.invalidate_cpu();
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        sub.invalidate_gpu();
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        sub.validate_cpu();
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        sub.validate_gpu();
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        sub.ensure_gpu_allocated();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        sub.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        sub.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const value_type* gpu_memory) const {
        sub.gpu_copy_from(gpu_memory);
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return sub.is_cpu_up_to_date();
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return sub.is_gpu_up_to_date();
    }

    /*!
     * \brief Print a representation of the view on the given stream
     * \param os The output stream
     * \param v The view to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const fast_matrix_view& v) {
        return os << "reshape(" << v.sub << ")";
    }
};

/*!
 * \brief Specialization for fast_matrix_view.
 */
template <typename T, bool DMA, size_t... Dims>
struct etl_traits<etl::fast_matrix_view<T, DMA, Dims...>> {
    using expr_t     = etl::fast_matrix_view<T, DMA, Dims...>; ///< The expression type
    using sub_expr_t = std::decay_t<T>;                        ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                 ///< The sub traits
    using value_type = typename sub_traits::value_type;        ///< The value type

    static constexpr bool is_etl         = true;                       ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                      ///< Indicates if the type is a transformer
    static constexpr bool is_view        = true;                       ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                      ///< Indicates if the type is a magic view
    static constexpr bool is_linear      = sub_traits::is_linear;      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = sub_traits::is_thread_safe; ///< Indicates if the expression is thread safe
    static constexpr bool is_fast        = true;                       ///< Indicates if the expression is fast
    static constexpr bool is_value       = false;                      ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = sub_traits::is_direct;      ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                      ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                      ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = false;                      ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = sub_traits::is_temporary;   ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr bool gpu_computable = false;                      ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = sub_traits::storage_order;  ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = sub_traits::template vectorizable<V>;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static constexpr size_t size([[maybe_unused]] const expr_t& v) {
        return (Dims * ...);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim([[maybe_unused]] const expr_t& v, size_t d) {
        return dyn_nth_size<Dims...>(d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() {
        return (Dims * ...);
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <size_t D>
    static constexpr size_t dim() {
        return nth_size<D, 0, Dims...>;
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return sizeof...(Dims);
    }

    /*!
     * \brief Estimate the complexity of computation
     * \return An estimation of the complexity of the expression
     */
    static constexpr int complexity() noexcept {
        return -1;
    }
};

} //end of namespace etl
