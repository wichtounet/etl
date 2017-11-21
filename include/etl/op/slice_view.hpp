//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief slice_view expression implementation
 */

#pragma once

namespace etl {

/*!
 * \brief View that shows a slice of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T, typename Enable>
struct slice_view;

/*!
 * \brief Specialization of slice_view for non-DMA types
 */
template <typename T>
struct slice_view  <T, std::enable_if_t<!fast_slice_view_able<T>>>
: assignable<slice_view<T>, value_t<T>>, value_testable<slice_view<T>>, iterable<slice_view<T>, fast_slice_view_able<T>> {
    using this_type            = slice_view<T>;                                                        ///< The type of this expression
    using iterable_base_type   = iterable<this_type, fast_slice_view_able<T>>;                  ///< The iterable base type
    using assignable_base_type = assignable<this_type, value_t<T>>;                                    ///< The assignable base type
    using sub_type             = T;                                                                    ///< The sub type
    using value_type           = value_t<sub_type>;                                                    ///< The value contained in the expression
    using memory_type          = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type    = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type          = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type    = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view
    using iterator             = etl::iterator<this_type>;                                             ///< The iterator type
    using const_iterator       = etl::iterator<const this_type>;                                       ///< The const iterator type

    using assignable_base_type::operator=;
    using iterable_base_type::begin;
    using iterable_base_type::end;

private:
    T sub;                   ///< The Sub expression
    const size_t first; ///< The index
    const size_t last;  ///< The last index

    friend struct etl_traits<slice_view>;

public:

    /*!
     * \brief Construct a new slice_view over the given sub expression
     * \param sub The sub expression
     * \param first The first index
     * \param last The last index
     */
    slice_view(sub_type sub, size_t first, size_t last)
            : sub(sub), first(first), last(last) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](size_t j) const {
        if /*constexpr*/ (decay_traits<sub_type>::storage_order == order::RowMajor) {
            return sub[first * (etl::size(sub) / dim<0>(sub)) + j];
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (etl::size(sub) / sa);
            return sub[(j % ss) * sa + j / ss + first];
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](size_t j) {
        if /*constexpr*/ (decay_traits<sub_type>::storage_order == order::RowMajor) {
            return sub[first * (etl::size(sub) / dim<0>(sub)) + j];
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (etl::size(sub) / sa);
            return sub[(j % ss) * sa + j / ss + first];
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t j) const noexcept {
        if /*constexpr*/ (decay_traits<sub_type>::storage_order == order::RowMajor) {
            return sub.read_flat(first * (etl::size(sub) / dim<0>(sub)) + j);
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (etl::size(sub) / sa);
            return sub.read_flat((j % ss) * sa + j / ss + first);
        }
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) + 1 == decay_traits<sub_type>::dimensions())>
    const_return_type operator()(size_t i, S... args) const {
        return sub(i + first, static_cast<size_t>(args)...);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) + 1 == decay_traits<sub_type>::dimensions())>
    return_type operator()(size_t i, S... args) {
        return sub(i + first, static_cast<size_t>(args)...);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param x The index to use
     * \return a sub view of the matrix at position x.
     */
    template <typename TT = sub_type, cpp_enable_iff(decay_traits<TT>::dimensions() > 1)>
    auto operator()(size_t x) const {
        return etl::sub(*this, x);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(size_t x) const noexcept {
        return sub.template loadu<V>(x + first * (etl::size(sub) / etl::dim<0>(sub)));
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(size_t x) const noexcept {
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

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_add_to(L&& lhs)  const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_sub_to(L&& lhs)  const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mul_to(L&& lhs)  const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_div_to(L&& lhs)  const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mod_to(L&& lhs)  const {
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
};

/*!
 * \brief Specialization of slice_view for DMA types
 */
template <typename T>
struct slice_view  <T, std::enable_if_t<fast_slice_view_able<T>>>
: assignable<slice_view<T>, value_t<T>>, value_testable<slice_view<T>>, iterable<slice_view<T>, true> {
    using this_type            = slice_view<T>;                                                        ///< The type of this expression
    using iterable_base_type   = iterable<this_type, true>;                                            ///< The iterable base type
    using assignable_base_type = assignable<this_type, value_t<T>>;                                    ///< The assignable base type
    using sub_type             = T;                                                                    ///< The sub type
    using value_type           = value_t<sub_type>;                                                    ///< The value contained in the expression
    using memory_type          = memory_t<sub_type>;                                                   ///< The memory acess type
    using const_memory_type    = const_memory_t<sub_type>;                                             ///< The const memory access type
    using return_type          = return_helper<sub_type, decltype(std::declval<sub_type>()[0])>;       ///< The type returned by the view
    using const_return_type    = const_return_helper<sub_type, decltype(std::declval<sub_type>()[0])>; ///< The const type return by the view
    using iterator             = memory_type;                                                          ///< The iterator type
    using const_iterator       = const_memory_type;                                                    ///< The const iterator type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<value_type>;

    using assignable_base_type::operator=;
    using iterable_base_type::begin;
    using iterable_base_type::end;

private:
    T sub;                   ///< The Sub expression
    const size_t first;      ///< The index
    const size_t last;       ///< The last index
    const size_t sub_size;   ///< The sub size

    mutable memory_type memory; ///< Pointer to the CPU memory

    mutable bool cpu_up_to_date; ///< Indicates if the CPU is up to date
    mutable bool gpu_up_to_date; ///< Indicates if the GPU is up to date

    friend struct etl_traits<slice_view>;

public:

    /*!
     * \brief Construct a new slice_view over the given sub expression
     * \param sub The sub expression
     * \param first The first index
     * \param last The last index
     */
    slice_view(sub_type sub, size_t first, size_t last)
            : sub(sub), first(first), last(last), sub_size((etl::size(sub) / etl::dim<0>(sub)) * (last - first)) {
        // Accessing the memory through fast sub views means evaluation
        if /*constexpr*/ (decay_traits<sub_type>::is_temporary){
            standard_evaluator::pre_assign_rhs(*this);
        }

        this->memory = this->sub.memory_start() + first * (etl::size(this->sub) / etl::dim<0>(this->sub));

        // A sub view inherits the CPU/GPU from parent
        this->cpu_up_to_date = this->sub.is_cpu_up_to_date();
        this->gpu_up_to_date = this->sub.is_gpu_up_to_date();

        cpp_assert(this->memory, "Invalid memory");
    }

    /*!
     * \brief Destructs the slice view
     */
    ~slice_view(){
        if (this->memory) {
            // Propagate the status on the parent
            if (!this->cpu_up_to_date) {
                if (sub.is_gpu_up_to_date()) {
                    sub.invalidate_cpu();
                } else {
                    // If the GPU is not up to date, cannot invalidate the CPU too
                    ensure_cpu_up_to_date();
                }
            }

            if (!this->gpu_up_to_date) {
                if (sub.is_cpu_up_to_date()) {
                    sub.invalidate_gpu();
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
    template <typename... S, cpp_enable_iff(sizeof...(S) + 1 == decay_traits<sub_type>::dimensions())>
    const_return_type operator()(size_t i, S... args) const {
        ensure_cpu_up_to_date();
        return memory[dyn_index(*this, i, args...)];
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) + 1 == decay_traits<sub_type>::dimensions())>
    return_type operator()(size_t i, S... args) {
        ensure_cpu_up_to_date();
        invalidate_gpu();
        return memory[dyn_index(*this, i, args...)];
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param x The index to use
     * \return a sub view of the matrix at position x.
     */
    template <typename TT = sub_type, cpp_enable_iff(decay_traits<TT>::dimensions() > 1)>
    auto operator()(size_t x) const {
        return etl::sub(*this, x);
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
        //TODO If the slice view is aligned (at compile-time), use stream store here
        return V::storeu(memory + x, in);
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
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E, cpp_enable_iff(is_dma<E>)>
    bool alias(const E& rhs) const noexcept {
        return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E, cpp_disable_iff(is_dma<E>)>
    bool alias(const E& rhs) const noexcept {
        return sub.alias(rhs);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    auto& gpu_compute_hint(Y& y){
        cpp_unused(y);
        this->ensure_gpu_up_to_date();
        return *this;
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    const auto& gpu_compute_hint(Y& y) const {
        cpp_unused(y);
        this->ensure_gpu_up_to_date();
        return *this;
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
        return memory + sub_size;
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return memory + sub_size;
    }

    // Assignment functions

    /*!
     * \brief Assign to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_to(L&& lhs)  const {
        std_assign_evaluate(*this, lhs);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_add_to(L&& lhs)  const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_sub_to(L&& lhs)  const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mul_to(L&& lhs)  const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_div_to(L&& lhs)  const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template<typename L>
    void assign_mod_to(L&& lhs)  const {
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

    // GPU functions

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return sub.gpu_memory() + first * (etl::size(sub) / etl::dim<0>(sub));
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
        sub.ensure_gpu_allocated();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        sub.ensure_gpu_allocated();

#ifdef ETL_CUDA
        if(!this->gpu_up_to_date){
            cuda_check_assert(cudaMemcpy(
                const_cast<std::remove_const_t<value_type>*>(gpu_memory()),
                const_cast<std::remove_const_t<value_type>*>(memory_start()),
                sub_size * sizeof(value_type), cudaMemcpyHostToDevice));

            this->gpu_up_to_date = true;

            inc_counter("gpu:slice:cpu_to_gpu");
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
            cuda_check_assert(cudaMemcpy(
                const_cast<std::remove_const_t<value_type>*>(memory_start()),
                const_cast<std::remove_const_t<value_type>*>(gpu_memory()),
                sub_size * sizeof(value_type), cudaMemcpyDeviceToHost));

            inc_counter("gpu:slice:gpu_to_cpu");
        }
#endif

        this->cpu_up_to_date = true;
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param new_gpu_memory Pointer to CPU memory from which to copy
     */
    void gpu_copy_from(const value_type* new_gpu_memory) const {
        cpp_assert(sub.gpu_memory(), "GPU must be allocated before copy");

#ifdef ETL_CUDA
        cuda_check_assert(cudaMemcpy(
            const_cast<std::remove_const_t<value_type>*>(gpu_memory()),
            const_cast<std::remove_const_t<value_type>*>(new_gpu_memory),
            sub_size * sizeof(value_type), cudaMemcpyDeviceToDevice));
#else
        cpp_unused(new_gpu_memory);
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

    static constexpr bool is_etl         = true;                       ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                      ///< Indicates if the type is a transformer
    static constexpr bool is_view        = true;                       ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                      ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = false;                      ///< Indicates if the expression is fast
    static constexpr bool is_linear      = sub_traits::is_linear;      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = sub_traits::is_thread_safe; ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                      ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = fast_slice_view_able<T>;    ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                      ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                      ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = false;                      ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = sub_traits::is_temporary;   ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr bool gpu_computable = is_direct;                  ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = sub_traits::storage_order;  ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = sub_traits::template vectorizable<V> && storage_order == order::RowMajor;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) {
        return (sub_traits::size(v.sub) / sub_traits::dim(v.sub, 0)) * (v.last - v.first);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, size_t d) {
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
    static constexpr size_t dimensions() {
        return sub_traits::dimensions();
    }
};

} //end of namespace etl
