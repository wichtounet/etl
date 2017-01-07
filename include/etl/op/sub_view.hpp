//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
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
template <typename T, typename Enable>
struct sub_view;

/*!
 * \brief View that shows a sub matrix of an expression
 * \tparam T The type of expression on which the view is made
 */
template <typename T>
struct sub_view <T, std::enable_if_t<!fast_sub_view_able<T>::value>> final :
    iterable<sub_view<T>, false>,
    assignable<sub_view<T>, value_t<T>>,
    inplace_assignable<sub_view<T>>
{
    static_assert(is_etl_expr<T>::value, "sub_view<T> only works with ETL expressions");

    using this_type            = sub_view<T>;                                                          ///< The type of this expression
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
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_type>;

    using assignable_base_type::operator=;
    using iterable_base_type::begin;
    using iterable_base_type::end;

private:
    sub_type sub_expr;              ///< The Sub expression
    const size_t i;          ///< The index
    const size_t sub_offset; ///< The sub size

    friend struct etl_traits<this_type>;

    static constexpr order storage_order = decay_traits<sub_type>::storage_order; /// < The storage order

public:
    /*!
     * \brief Construct a new sub_view over the given sub expression
     * \param sub_expr The sub expression
     * \param i The sub index
     */
    sub_view(sub_type sub_expr, std::size_t i)
            : sub_expr(sub_expr), i(i), sub_offset(i * subsize(sub_expr)) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        return storage_order == order::RowMajor
                   ? sub_expr[sub_offset + j]
                   : sub_expr[i + dim<0>(sub_expr) * j];
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        return storage_order == order::RowMajor
                   ? sub_expr[sub_offset + j]
                   : sub_expr[i + dim<0>(sub_expr) * j];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        return storage_order == order::RowMajor
                   ? sub_expr.read_flat(sub_offset + j)
                   : sub_expr.read_flat(i + dim<0>(sub_expr) * j);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_if((sizeof...(S) + 1 == decay_traits<sub_type>::dimensions()))>
    ETL_STRONG_INLINE(const_return_type) operator()(S... args) const {
        return sub_expr(i, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_if((sizeof...(S) + 1 == decay_traits<sub_type>::dimensions()))>
    ETL_STRONG_INLINE(return_type) operator()(S... args) {
        return sub_expr(i, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param x The index to use
     * \return a sub view of the matrix at position x.
     */
    template <typename TT = sub_type, cpp_enable_if((decay_traits<TT>::dimensions() > 2))>
    auto operator()(std::size_t x) const {
        return sub(*this, x);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t x) noexcept {
        return sub_expr.template storeu<V>(in, x + sub_offset);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t x) noexcept {
        return sub_expr.template storeu<V>(in, x + sub_offset);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t x) noexcept {
        return sub_expr.template storeu<V>(in, x + sub_offset);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>) load(std::size_t x) const noexcept {
        return sub_expr.template loadu<V>(x + sub_offset);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>) loadu(std::size_t x) const noexcept {
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
    std::size_t& unsafe_dimension_access(std::size_t x) {
        return sub_expr.unsafe_dimension_access(x + 1);
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        sub_expr.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        sub_expr.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        sub_expr.visit(visitor);
        visitor.need_value = old_need_value;
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
template <typename T>
struct sub_view <T, std::enable_if_t<fast_sub_view_able<T>::value>> :
    iterable<sub_view<T>, true>,
    assignable<sub_view<T>, value_t<T>>,
    inplace_assignable<sub_view<T>>
{
    static_assert(is_etl_expr<T>::value, "sub_view<T> only works with ETL expressions");
    static_assert(decay_traits<T>::dimensions() > 1, "sub_view<T, true> should only be done with Matrices >1D");
    static_assert(decay_traits<T>::storage_order == order::RowMajor, "sub_view<T, true> should only be done with RowMajor");

    using this_type            = sub_view<T>;                                                          ///< The type of this expression
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
    template<typename V = default_vec>
    using vec_type               = typename V::template vec_type<value_type>;

    using iterable_base_type::begin;
    using iterable_base_type::end;
    using assignable_base_type::operator=;

private:
    T sub_expr;            ///< The Sub expression
    const size_t i;        ///< The indbex
    const size_t sub_size; ///< The sub size

    static constexpr size_t n_dimensions = decay_traits<sub_type>::dimensions() - 1; ///< The Number of dimensions
    static constexpr order storage_order = decay_traits<sub_type>::storage_order;    ///< The storage order

    mutable memory_type memory;

    mutable bool cpu_up_to_date;
    mutable bool gpu_up_to_date;

    friend struct etl_traits<this_type>;

public:
    /*!
     * \brief Construct a new sub_view over the given sub expression
     * \param sub_expr The sub expression
     * \param i The sub index
     */
    sub_view(sub_type sub_expr, std::size_t i) : sub_expr(sub_expr), i(i), sub_size(subsize(sub_expr)) {
        if(!decay_traits<sub_type>::needs_evaluator_visitor){
            this->memory = sub_expr.memory_start() + i * sub_size;
        } else {
            this->memory = nullptr;
        }

        // A sub view inherits the CPU/GPU from parent
        this->cpu_up_to_date = sub_expr.is_cpu_up_to_date();
        this->gpu_up_to_date = sub_expr.is_gpu_up_to_date();
    }

    ~sub_view(){
        //TODO Reflect the status of up to date
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

    //TODO These two following operators should also be handled in lighter weight way!

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_if((sizeof...(S) == n_dimensions && sizeof...(S) > 2))>
    ETL_STRONG_INLINE(const_return_type) operator()(S... args) const {
        return sub_expr(i, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S, cpp_enable_if((sizeof...(S) == n_dimensions && sizeof...(S) > 2))>
    ETL_STRONG_INLINE(return_type) operator()(S... args) {
        return sub_expr(i, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Access to the element at position (x)
     * \param x The index of the first dimension
     * \return A reference to the element at position x
     */
    template <cpp_enable_if_cst((n_dimensions == 1))>
    value_type& operator()(size_t x) noexcept {
        return memory[x];
    }

    /*!
     * \brief Access to the element at position (x)
     * \param x The index of the first dimension
     * \return A const reference to the element at position x
     */
    template <cpp_enable_if_cst((n_dimensions == 1))>
    const value_type& operator()(size_t x) const noexcept {
        return memory[x];
    }

    /*!
     * \brief Access to the element at position (x)
     * \param x The index of the first dimension
     * \return A reference to the element at position x
     */
    template <cpp_enable_if_cst((n_dimensions == 2))>
    value_type& operator()(size_t x, size_t j) noexcept {
        return memory[x * etl::dim<2>(sub_expr) + j];
    }

    /*!
     * \brief Access to the element at position (x)
     * \param x The index of the first dimension
     * \return A const reference to the element at position x
     */
    template <cpp_enable_if_cst((n_dimensions == 2))>
    const value_type& operator()(size_t x, size_t j) const noexcept {
        return memory[x * etl::dim<2>(sub_expr) + j];
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param x The index to use
     * \return a sub view of the matrix at position x.
     */
    template <cpp_enable_if_cst((n_dimensions > 1))>
    auto operator()(std::size_t x) const {
        return sub(*this, x);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void store(vec_type<V> in, std::size_t x) noexcept {
        return V::storeu(memory + x, in);
    }

    /*!
     * \brief Store several elements in the matrix at once
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void storeu(vec_type<V> in, std::size_t x) noexcept {
        return V::storeu(memory + x, in);
    }

    /*!
     * \brief Store several elements in the matrix at once, using non-temporal store
     * \param in The several elements to store
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     */
    template <typename V = default_vec>
    void stream(vec_type<V> in, std::size_t x) noexcept {
        return V::stream(memory + x, in);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    vec_type<V> load(std::size_t x) const noexcept {
        return V::loadu(memory + x);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    vec_type<V> loadu(std::size_t x) const noexcept {
        return V::loadu(memory + x);
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E, cpp_enable_if(all_dma<E>::value)>
    bool alias(const E& rhs) const noexcept {
        return memory_alias(memory_start(), memory_end(), rhs.memory_start(), rhs.memory_end());
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E, cpp_disable_if(all_dma<E>::value)>
    bool alias(const E& rhs) const noexcept {
        return sub_expr.alias(rhs);
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
    std::size_t& unsafe_dimension_access(std::size_t x) {
        return sub_expr.unsafe_dimension_access(x + 1);
    }

    /*!
     * \brief Returns all the Ith... dimensions in array
     * \return an array containing the Ith... dimensions of the expression.
     */
    template<std::size_t... I>
    std::array<std::size_t, decay_traits<this_type>::dimensions()> dim_array(std::index_sequence<I...>) const {
        return {{decay_traits<this_type>::dim(*this, I)...}};
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        sub_expr.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        sub_expr.visit(visitor);

        // It's only interesting if the sub expression is not direct
        if(decay_traits<sub_type>::needs_evaluator_visitor){
            this->memory = sub_expr.memory_start() + i * sub_size;
        }
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        sub_expr.visit(visitor);
        visitor.need_value = old_need_value;
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
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        // Allocate is done by the sub
        sub_expr.ensure_gpu_allocated();
        // TODO This may be dangerous because it sets gpu_up_to_date of the parent

        this->gpu_up_to_date = true;
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
#ifdef ETL_CUDA
        if(!this->gpu_up_to_date){
            cuda_check(cudaMemcpy(
                const_cast<std::remove_const_t<value_type>*>(gpu_memory()),
                const_cast<std::remove_const_t<value_type>*>(memory_start()),
                sub_size * sizeof(value_type), cudaMemcpyHostToDevice));
        }
#endif

        gpu_up_to_date = true;
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
#ifdef ETL_CUDA
        if (!this->cpu_up_to_date) {
            cuda_check(cudaMemcpy(
                const_cast<std::remove_const_t<value_type>*>(memory_start()),
                const_cast<std::remove_const_t<value_type>*>(gpu_memory()),
                sub_size * sizeof(value_type), cudaMemcpyDeviceToHost));
        }
#endif

        cpu_up_to_date = true;
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const value_type* new_gpu_memory) const {
        cpp_assert(is_gpu_allocated(), "GPU must be allocated before copy");

#ifdef ETL_CUDA
        cuda_check(cudaMemcpy(
            const_cast<std::remove_const_t<value_type>*>(gpu_memory()),
            const_cast<std::remove_const_t<value_type>*>(new_gpu_memory),
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
template <typename T>
struct etl_traits<etl::sub_view<T>> {
    using expr_t     = etl::sub_view<T>;       ///< The expression type
    using sub_expr_t = std::decay_t<T>;        ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>; ///< The sub traits
    using value_type = typename sub_traits::value_type;

    static constexpr bool is_etl                  = true;                                                                  ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                                                                 ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = true;                                                                  ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                                                 ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = sub_traits::is_fast;                                                   ///< Indicates if the expression is fast
    static constexpr bool is_linear               = sub_traits::is_linear;                                                 ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = sub_traits::is_thread_safe;                                            ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                                                 ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = sub_traits::is_direct && sub_traits::storage_order == order::RowMajor; ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                                                 ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                                                                 ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = false;                                                                 ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = sub_traits::needs_evaluator_visitor;                                   ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr order storage_order          = sub_traits::storage_order;                                             ///< The expression's storage order

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
    static std::size_t size(const expr_t& v) noexcept {
        return sub_traits::size(v.sub_expr) / sub_traits::dim(v.sub_expr, 0);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) noexcept {
        return sub_traits::dim(v.sub_expr, d + 1);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() noexcept {
        return sub_traits::size() / sub_traits::template dim<0>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() noexcept {
        return sub_traits::template dim<D + 1>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() noexcept {
        return sub_traits::dimensions() - 1;
    }
};

} //end of namespace etl
