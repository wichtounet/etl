//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the sub_view implementation
 */

#pragma once

#ifdef ETL_CUDA
#include "etl/impl/cublas/cuda.hpp"
#endif

namespace etl {

/*!
 * \brief View that shows a sub matrix of an expression
 * \tparam T The type of expression on which the view is made
 */
template <etl_expr T, bool Aligned>
requires(!fast_sub_view_able<T>)
struct sub_view<T, Aligned> final : iterable<sub_view<T, Aligned>, false>,
                                                                              assignable<sub_view<T, Aligned>, value_t<T>>,
                                                                              value_testable<sub_view<T, Aligned>>,
                                                                              inplace_assignable<sub_view<T, Aligned>> {
    using this_type            = sub_view<T, Aligned>;                                                 ///< The type of this expression
    using iterable_base_type   = iterable<this_type, false>;                                           ///< The iterable base type
    using assignable_base_type = assignable<this_type, value_t<T>>;                                    ///< The iterable base type
    using sub_type             = T;                                                                    ///< The sub type
    using value_type           = value_t<sub_type>;                                                    ///< The value contained in the expression
    using memory_type          = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type    = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type          = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type    = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view
    using iterator             = etl::iterator<this_type>;                                             ///< The iterator type
    using const_iterator       = etl::iterator<const this_type>;                                       ///< The const iterator type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<value_type>;

    using assignable_base_type::operator=;
    using iterable_base_type::begin;
    using iterable_base_type::end;

private:
    sub_type sub_expr;       ///< The Sub expression
    const size_t i;          ///< The index
    const size_t sub_offset; ///< The sub size

    friend struct etl_traits<sub_view>;

    static constexpr order storage_order = decay_traits<sub_type>::storage_order; ///< The storage order

public:
    /*!
     * \brief Construct a new sub_view over the given sub expression
     * \param sub_expr The sub expression
     * \param i The sub index
     */
    sub_view(sub_type sub_expr, size_t i) : sub_expr(sub_expr), i(i), sub_offset(i * subsize(sub_expr)) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](size_t j) const {
        return storage_order == order::RowMajor ? sub_expr[sub_offset + j] : sub_expr[i + dim<0>(sub_expr) * j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](size_t j) {
        return storage_order == order::RowMajor ? sub_expr[sub_offset + j] : sub_expr[i + dim<0>(sub_expr) * j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t j) const noexcept {
        return storage_order == order::RowMajor ? sub_expr.read_flat(sub_offset + j) : sub_expr.read_flat(i + dim<0>(sub_expr) * j);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    ETL_STRONG_INLINE(const_return_type)
    operator()(S... args) const requires(sizeof...(S) + 1 == decay_traits<sub_type>::dimensions()) {
        return sub_expr(i, static_cast<size_t>(args)...);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    ETL_STRONG_INLINE(return_type)
    operator()(S... args) requires(sizeof...(S) + 1 == decay_traits<sub_type>::dimensions()) {
        return sub_expr(i, static_cast<size_t>(args)...);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param x The index to use
     * \return a sub view of the matrix at position x.
     */
    auto operator()(size_t x) const requires(decay_traits<sub_type>::dimensions() > 2) {
        return sub(*this, x);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, size_t x) noexcept {
        return sub_expr.template storeu<V>(in, x + sub_offset);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, size_t x) noexcept {
        return sub_expr.template storeu<V>(in, x + sub_offset);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, size_t x) noexcept {
        return sub_expr.template storeu<V>(in, x + sub_offset);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>)
    load(size_t x) const noexcept {
        return sub_expr.template loadu<V>(x + sub_offset);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>)
    loadu(size_t x) const noexcept {
        return sub_expr.template loadu<V>(x + sub_offset);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return sub_expr.alias(rhs);
    }

    /*!
     * \brief Returns a reference to the ith dimension value.
     *
     * This should only be used internally and with care
     *
     * \return a refernece to the ith dimension value.
     */
    size_t& unsafe_dimension_access(size_t x) {
        return sub_expr.unsafe_dimension_access(x + 1);
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, std::forward<L>(lhs));
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        sub_expr.visit(visitor);
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // The sub value must be ensured
        sub_expr.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // The sub value must be ensured
        sub_expr.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Print a representation of the view on the given stream
     * \param os The output stream
     * \param v The view to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const sub_view& v) {
        return os << "sub(" << v.sub_expr << ", " << v.i << ")";
    }
};

/*!
 * \brief View that shows a sub matrix of an expression
 * \tparam T The type of expression on which the view is made
 */
template <etl_expr T, bool Aligned>
requires(fast_sub_view_able<T>)
struct sub_view<T, Aligned> : iterable<sub_view<T, Aligned>, true>,
                                                                       assignable<sub_view<T, Aligned>, value_t<T>>,
                                                                       value_testable<sub_view<T, Aligned>>,
                                                                       inplace_assignable<sub_view<T, Aligned>> {
    static_assert(decay_traits<T>::dimensions() > 1, "sub_view<T, true> should only be done with Matrices >1D");
    static_assert(decay_traits<T>::storage_order == order::RowMajor, "sub_view<T, true> should only be done with RowMajor");

    using this_type            = sub_view<T, Aligned>;                                                 ///< The type of this expression
    using iterable_base_type   = iterable<this_type, true>;                                            ///< The iterable base type
    using assignable_base_type = assignable<this_type, value_t<T>>;                                    ///< The iterable base type
    using sub_type             = T;                                                                    ///< The sub type
    using value_type           = value_t<sub_type>;                                                    ///< The value contained in the expression
    using memory_type          = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type    = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type          = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type    = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view
    using iterator             = value_type*;                                                          ///< The iterator type
    using const_iterator       = const value_type*;                                                    ///< The const iterator type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<value_type>;

    using iterable_base_type::begin;
    using iterable_base_type::end;
    using assignable_base_type::operator=;

private:
    T sub_expr;            ///< The Sub expression
    const size_t i;        ///< The indbex
    const size_t sub_size; ///< The sub size

    static constexpr size_t n_dimensions = decay_traits<sub_type>::dimensions() - 1; ///< The Number of dimensions
    static constexpr order storage_order = decay_traits<sub_type>::storage_order;    ///< The storage order

    mutable memory_type memory; ///< Pointer to the CPU memory

    mutable bool cpu_up_to_date; ///< Indicates if the CPU is up to date
    mutable bool gpu_up_to_date; ///< Indicates if the GPU is up to date

    friend struct etl_traits<sub_view>;

public:
    /*!
     * \brief Construct a new sub_view over the given sub expression
     * \param sub_expr The sub expression
     * \param i The sub index
     */
    sub_view(sub_type sub_expr, size_t i) : sub_expr(sub_expr), i(i), sub_size(subsize(sub_expr)) {
        // Accessing the memory through fast sub views means evaluation
        if constexpr (decay_traits<sub_type>::is_temporary) {
            standard_evaluator::pre_assign_rhs(*this);
        }

        this->memory = this->sub_expr.memory_start() + i * sub_size;

        // A sub view inherits the CPU/GPU from parent
        this->cpu_up_to_date = this->sub_expr.is_cpu_up_to_date();
        this->gpu_up_to_date = this->sub_expr.is_gpu_up_to_date();

        cpp_assert(this->memory, "Invalid memory");
    }

    ~sub_view() {
        if (this->memory) {
            // Propagate the status on the parent
            if (!this->cpu_up_to_date) {
                if (sub_expr.is_gpu_up_to_date()) {
                    sub_expr.invalidate_cpu();
                } else {
                    // If the GPU is not up to date, cannot invalidate the CPU too
                    ensure_cpu_up_to_date();
                }
            }

            if (!this->gpu_up_to_date) {
                if (sub_expr.is_cpu_up_to_date()) {
                    sub_expr.invalidate_gpu();
                } else {
                    // If the GPU is not up to date, cannot invalidate the CPU too
                    ensure_gpu_up_to_date();
                }
            }
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](size_t j) const {
        ensure_cpu_up_to_date();
        return memory[j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](size_t j) {
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
    value_type read_flat(size_t j) const noexcept {
        ensure_cpu_up_to_date();
        return memory[j];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    ETL_STRONG_INLINE(const_return_type)
    operator()(S... args) const requires (sizeof...(S) == n_dimensions) {
        ensure_cpu_up_to_date();
        return memory[dyn_index(*this, args...)];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    ETL_STRONG_INLINE(return_type)
    operator()(S... args) requires (sizeof...(S) == n_dimensions) {
        ensure_cpu_up_to_date();
        return memory[dyn_index(*this, args...)];
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param x The index to use
     * \return a sub view of the matrix at position x.
     */
    auto operator()(size_t x) const requires(n_dimensions > 1) {
        return sub(*this, x);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, size_t x) noexcept {
        return V::storeu(memory + x, in);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, size_t x) noexcept {
        return V::storeu(memory + x, in);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, size_t x) noexcept {
        //TODO If the sub view is aligned (at compile-time), use stream store here
        return V::storeu(memory + x, in);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    vec_type<V> load(size_t x) const noexcept {
        return V::loadu(memory + x);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    vec_type<V> loadu(size_t x) const noexcept {
        return V::loadu(memory + x);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        if constexpr (is_dma<E>) {
            return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
        } else {
            return sub_expr.alias(rhs);
        }
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    ETL_STRONG_INLINE(memory_type) memory_start() noexcept {
        return memory;
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    ETL_STRONG_INLINE(const_memory_type) memory_start() const noexcept {
        return memory;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    ETL_STRONG_INLINE(memory_type) memory_end() noexcept {
        return memory + sub_size;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    ETL_STRONG_INLINE(const_memory_type) memory_end() const noexcept {
        return memory + sub_size;
    }

    /*!
     * \brief Returns a reference to the ith dimension value.
     *
     * This should only be used internally and with care
     *
     * \return a refernece to the ith dimension value.
     */
    size_t& unsafe_dimension_access(size_t x) {
        return sub_expr.unsafe_dimension_access(x + 1);
    }

    /*!
     * \brief Returns all the Ith... dimensions in array
     * \return an array containing the Ith... dimensions of the expression.
     */
    template <size_t... I>
    std::array<size_t, decay_traits<this_type>::dimensions()> dim_array(std::index_sequence<I...>) const {
        return {{decay_traits<this_type>::dim(*this, I)...}};
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        std_assign_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    auto& gpu_compute_hint([[maybe_unused]] Y& y) {
        this->ensure_gpu_up_to_date();
        return *this;
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    const auto& gpu_compute_hint([[maybe_unused]] Y& y) const {
        this->ensure_gpu_up_to_date();
        return *this;
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        sub_expr.visit(visitor);
    }

    // GPU functions

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return sub_expr.gpu_memory() + i * sub_size;
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        sub_expr.gpu_evict();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        this->cpu_up_to_date = false;
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        this->gpu_up_to_date = false;
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        this->cpu_up_to_date = true;
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        this->gpu_up_to_date = true;
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        // Allocate is done by the sub
        sub_expr.ensure_gpu_allocated();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        sub_expr.ensure_gpu_allocated();

#ifdef ETL_CUDA
        if (!this->gpu_up_to_date) {
            cuda_check_assert(cudaMemcpy(const_cast<std::remove_const_t<value_type>*>(gpu_memory()),
                                         const_cast<std::remove_const_t<value_type>*>(memory_start()), sub_size * sizeof(value_type), cudaMemcpyHostToDevice));

            this->gpu_up_to_date = true;

            inc_counter("gpu:sub:cpu_to_gpu");
        }
#endif
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
#ifdef ETL_CUDA
        if (!this->cpu_up_to_date) {
            cuda_check_assert(cudaMemcpy(const_cast<std::remove_const_t<value_type>*>(memory_start()),
                                         const_cast<std::remove_const_t<value_type>*>(gpu_memory()), sub_size * sizeof(value_type), cudaMemcpyDeviceToHost));

            inc_counter("gpu:sub:gpu_to_cpu");
        }
#endif

        this->cpu_up_to_date = true;
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param new_gpu_memory Pointer to CPU memory from which to copy
     */
    void gpu_copy_from([[maybe_unused]] const value_type* new_gpu_memory) const {
        cpp_assert(sub_expr.gpu_memory(), "GPU must be allocated before copy");

#ifdef ETL_CUDA
        cuda_check_assert(cudaMemcpy(const_cast<std::remove_const_t<value_type>*>(gpu_memory()), const_cast<std::remove_const_t<value_type>*>(new_gpu_memory),
                                     sub_size * sizeof(value_type), cudaMemcpyDeviceToDevice));
#endif

        gpu_up_to_date = true;
        cpu_up_to_date = false;
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return cpu_up_to_date;
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return gpu_up_to_date;
    }

    /*!
     * \brief Print a representation of the view on the given stream
     * \param os The output stream
     * \param v The view to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const sub_view& v) {
        return os << "sub(" << v.sub_expr << ", " << v.i << ")";
    }
};

/*!
 * \brief Specialization for sub_view
 */
template <typename T, bool Aligned>
struct etl_traits<etl::sub_view<T, Aligned>> {
    using expr_t     = etl::sub_view<T, Aligned>;       ///< The expression type
    using sub_expr_t = std::decay_t<T>;                 ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;          ///< The sub traits
    using value_type = typename sub_traits::value_type; ///< The value type of the expression

    static constexpr bool is_etl         = true;                       ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                      ///< Indicates if the type is a transformer
    static constexpr bool is_view        = true;                       ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                      ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast;        ///< Indicates if the expression is fast
    static constexpr bool is_linear      = sub_traits::is_linear;      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = sub_traits::is_thread_safe; ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                      ///< Indicates if the expression is of value type
    static constexpr bool is_direct =
        sub_traits::is_direct && sub_traits::storage_order == order::RowMajor; ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                              ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                              ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = false;                              ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = sub_traits::is_temporary;           ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr bool gpu_computable = fast_sub_view_able<T>;              ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = sub_traits::storage_order;          ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = sub_traits::template vectorizable<V>&& storage_order == order::RowMajor;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) noexcept {
        return sub_traits::size(v.sub_expr) / sub_traits::dim(v.sub_expr, 0);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, size_t d) noexcept {
        return sub_traits::dim(v.sub_expr, d + 1);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() noexcept {
        return sub_traits::size() / sub_traits::template dim<0>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <size_t D>
    static constexpr size_t dim() noexcept {
        return sub_traits::template dim<D + 1>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() noexcept {
        return sub_traits::dimensions() - 1;
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
