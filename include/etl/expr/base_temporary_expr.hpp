//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <memory> //For shared_ptr

#include "etl/iterator.hpp"

namespace etl {

namespace temporary_detail {

/*!
 * \brief Traits to build the result type of a temporary expression
 * \tparam E The temporary expression type
 * \tparam Fast Indicates if the result is fast or dynamic
 * \tparam Subs The sub expressions
 */
template <typename E, bool Fast, typename... Subs>
struct expr_result;

/*!
 * \copydoc expr_result
 */
template <typename E>
struct expr_result<E, false> {
    /*!
     * \brief The built type for the given Subs
     */
    using type = dyn_matrix_impl<typename decay_traits<E>::value_type, decay_traits<E>::storage_order, decay_traits<E>::dimensions()>;
};

/*!
 * \copydoc expr_result
 */
template <typename E>
struct expr_result<E, true> {
    /*!
     * \brief The built type for the given Subs
     */
    using type = typename detail::build_fast_dyn_matrix_type<E, std::make_index_sequence<decay_traits<E>::dimensions()>>::type;
};

/*!
 * \brief Helper traits to directly get the result type for an impl_expr
 * \tparam E The temporary expression type
 */
template <typename E, typename... Subs>
using expr_result_t = typename expr_result<E, all_fast<E>::value, Subs...>::type;

} // end of temporary_detail

/*!
 * \brief A temporary expression base
 *
 * \tparam D The derived type
 * \tparam R The result type
 *
 * A temporary expression computes the expression directly and stores it into a temporary.
 */
template <typename D>
struct base_temporary_expr : value_testable<D>, dim_testable<D>, iterable<const D, true> {
    using derived_t         = D;                                    ///< The derived type
    using value_type        = typename decay_traits<D>::value_type; ///< The value type
    using result_type       = temporary_detail::expr_result_t<D>;   ///< The result type
    using memory_type       = value_type*;                          ///< The memory type
    using const_memory_type = const value_type*;                    ///< The const memory type

protected:
    mutable bool allocated = false; ///< Indicates if the temporary has been allocated
    mutable bool evaluated = false; ///< Indicates if the expression has been evaluated

    mutable std::shared_ptr<result_type> _c;           ///< The result reference

private:
    gpu_memory_handler<value_type> _gpu;                 ///< The GPU memory handler

public:
    /*!
     * \brief Construct a new base_temporary_expr
     */
    base_temporary_expr() = default;

    /*!
     * \brief Copy construct a new base_temporary_expr
     */
    base_temporary_expr(const base_temporary_expr& expr) = default;

    /*!
     * \brief Move construct a base_temporary_expr
     * The right hand side cannot be used anymore after ths move.
     * \param rhs The expression to move from.
     */
    base_temporary_expr(base_temporary_expr&& rhs) : allocated(rhs.allocated), evaluated(rhs.evaluated), _c(std::move(rhs._c)) {
        rhs.evaluated = false;
    }

    //Expressions are invariant
    base_temporary_expr& operator=(const base_temporary_expr& /*e*/) = delete;
    base_temporary_expr& operator=(base_temporary_expr&& /*e*/) = delete;

    /*!
     * \brief The vectorization type for VV
     */
    template <typename VV = default_vec>
    using vec_type        = typename VV::template vec_type<value_type>;

protected:
    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    derived_t& as_derived() noexcept {
        return *static_cast<derived_t*>(this);
    }

    /*!
     * \brief Returns a reference to the derived object, i.e. the object using the CRTP injector.
     * \return a reference to the derived object.
     */
    const derived_t& as_derived() const noexcept {
        return *static_cast<const derived_t*>(this);
    }

protected:
    /*!
     * \brief Evaluate the expression, if not evaluated
     *
     * Will fail if not previously allocated
     */
    void evaluate(){
        if (!evaluated) {
            cpp_assert(allocated, "The result has not been allocated");
            as_derived().apply_base(*_c);
            evaluated = true;
        }
    }

    /*!
     * \brief Allocate the necessary temporaries, if necessary
     */
    void allocate_temporary() const {
        if (!_c) {
            _c.reset(allocate());
        }

        allocated = true;
    }

    /*!
     * \brief Allocate the temporary
     */
    template <cpp_enable_if_cst(all_fast<derived_t>::value)>
    result_type* allocate() const {
        return new result_type;
    }

    /*!
     * \brief Allocate the dynamic temporary
     */
    template <std::size_t... I>
    result_type* dyn_allocate(std::index_sequence<I...> /*seq*/) const {
        return new result_type(decay_traits<derived_t>::dim(as_derived(), I)...);
    }

    /*!
     * \brief Allocate the temporary
     */
    template <cpp_disable_if_cst(all_fast<derived_t>::value)>
    result_type* allocate() const {
        return dyn_allocate(std::make_index_sequence<decay_traits<derived_t>::dimensions()>());
    }

public:
    //Apply the expression

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) const {
        return result()[i];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        return result().read_flat(i);
    }

    /*!
     * \brief Returns the value at the given position (args...)
     * \param args The position indices
     * \return The value at the given position (args...)
     */
    template <typename... S, cpp_enable_if(sizeof...(S) == safe_dimensions<derived_t>::value)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return result()(args...);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <typename DD = D, cpp_enable_if((safe_dimensions<DD>::value > 1))>
    auto operator()(std::size_t i) const {
        return sub(as_derived(), i);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(std::size_t first, std::size_t last) noexcept {
        return slice(*this, first, last);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(std::size_t first, std::size_t last) const noexcept {
        return slice(*this, first, last);
    }

    /*!
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam VV The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename VV = default_vec>
    vec_type<VV> load(std::size_t i) const noexcept {
        return VV::loadu(memory_start() + i);
    }

    /*!
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam VV The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename VV = default_vec>
    vec_type<VV> loadu(std::size_t i) const noexcept {
        return VV::loadu(memory_start() + i);
    }

    // Direct memory access

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        return result().memory_start();
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        return result().memory_start();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        return result().memory_end();
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        return result().memory_end();
    }

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        return _gpu.gpu_memory();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        _gpu.gpu_evict();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        _gpu.invalidate_cpu();
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        _gpu.invalidate_gpu();
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        _gpu.validate_cpu();
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        _gpu.validate_gpu();
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        _gpu.ensure_gpu_allocated(etl::size(result()));
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        _gpu.ensure_gpu_up_to_date(memory_start(), etl::size(result()));
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        _gpu.ensure_cpu_up_to_date(memory_start(), etl::size(result()));
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const value_type* gpu_memory) const {
        _gpu.gpu_copy_from(gpu_memory, etl::size(result()));
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return _gpu.is_cpu_up_to_date();
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return _gpu.is_gpu_up_to_date();
    }

private:
    /*!
     * \brief Returns the expression containing the result of the expression.
     * \return a reference to the expression containing the result of the expression
     */
    result_type& result() {
        cpp_assert(evaluated, "The result has not been evaluated");
        cpp_assert(allocated, "The result has not been allocated");
        return *_c;
    }

    /*!
     * \brief Returns the expression containing the result of the expression.
     * \return a const reference to the expression containing the result of the expression
     */
    const result_type& result() const {
        cpp_assert(evaluated, "The result has not been evaluated");
        cpp_assert(allocated, "The result has not been allocated");
        return *_c;
    }
};

/*!
 * \brief Abstrct base class for temporary unary expression
 * \tparam D The derived type
 * \tparam T The value type
 * \tparam A The sub type
 * \tparam R The result type, if forced
 */
template <typename D, typename A>
struct base_temporary_expr_un : base_temporary_expr<D> {
    static_assert(is_etl_expr<A>::value, "The argument must be an ETL expr");

    using this_type = base_temporary_expr_un<D, A>; ///< This type
    using base_type = base_temporary_expr<D>;       ///< The base type

    A _a;                       ///< The sub expression reference

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit base_temporary_expr_un(A a) : _a(a) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new expression by copy
     * \param e The expression to copy
     */
    base_temporary_expr_un(const base_temporary_expr_un& e) : base_type(e), _a(e._a) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new expression by move
     * \param e The expression to move
     */
    base_temporary_expr_un(base_temporary_expr_un&& e) noexcept : base_type(std::move(e)), _a(e._a){
        //Nothing else to init
    }

    /*!
     * \brief Apply the op and store the result in result
     * \param result The expressio where to store the result
     */
    template <typename Result>
    void apply_base(Result&& result){
        this->as_derived().apply(_a, std::forward<Result>(result));
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const {
        return _a.alias(rhs);
    }

    /*!
     * \brief Returns the sub expression
     * \return a reference to the sub expression
     */
    std::add_lvalue_reference_t<A> a() {
        return _a;
    }

    /*!
     * \brief Returns the sub expression
     * \return a reference to the sub expression
     */
    cpp::add_const_lvalue_t<A> a() const {
        return _a;
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        this->allocate_temporary();

        _a.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        _a.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;

        visitor.need_value = decay_traits<D>::is_gpu;
        _a.visit(visitor);

        this->evaluate();

        if (old_need_value) {
            this->ensure_cpu_up_to_date();
        }

        visitor.need_value = old_need_value;
    }
};

/*!
 * \brief Abstrct base class for temporary binary expression
 * \tparam D The derived type
 * \tparam T The value type
 * \tparam A The left sub expression type
 * \tparam B The right sub expression type
 * \tparam R The result type, if forced (void otherwise)
 */
template <typename D, typename A, typename B>
struct base_temporary_expr_bin : base_temporary_expr<D> {
    static_assert(is_etl_expr<A>::value, "The argument must be an ETL expr");
    static_assert(is_etl_expr<B>::value, "The argument must be an ETL expr");

    using this_type = base_temporary_expr_bin<D, A, B>; ///< This type
    using base_type = base_temporary_expr<D>;           ///< The base type

    A _a;                       ///< The sub expression reference
    B _b;                       ///< The sub expression reference

    /*!
     * \brief Construct a new expression
     * \param a The left sub expression
     * \param b The right sub expression
     */
    explicit base_temporary_expr_bin(A a, B b) : _a(a), _b(b) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new expression by copy
     * \param e The expression to copy
     */
    base_temporary_expr_bin(const base_temporary_expr_bin& e) : base_type(e), _a(e._a), _b(e._b) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new expression by move
     * \param e The expression to move
     */
    base_temporary_expr_bin(base_temporary_expr_bin&& e) noexcept : base_type(std::move(e)), _a(e._a), _b(e._b) {
        //Nothing else to init
    }

    /*!
     * \brief Apply the op and store the result in result
     * \param result The expressio where to store the result
     */
    template <typename Result>
    void apply_base(Result&& result){
        this->as_derived().apply(_a, _b, std::forward<Result>(result));
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const {
        return _a.alias(rhs) || _b.alias(rhs);
    }

    /*!
     * \brief Returns the sub expression
     * \return a reference to the sub expression
     */
    std::add_lvalue_reference_t<A> a() {
        return _a;
    }

    /*!
     * \brief Returns the sub expression
     * \return a reference to the sub expression
     */
    cpp::add_const_lvalue_t<A> a() const {
        return _a;
    }

    /*!
     * \brief Returns the sub expression
     * \return a reference to the sub expression
     */
    std::add_lvalue_reference_t<B> b() {
        return _b;
    }

    /*!
     * \brief Returns the sub expression
     * \return a reference to the sub expression
     */
    cpp::add_const_lvalue_t<B> b() const {
        return _b;
    }

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor){
        this->allocate_temporary();

        _a.visit(visitor);
        _b.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor){
        _a.visit(visitor);
        _b.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor){
        bool old_need_value = visitor.need_value;

        visitor.need_value = decay_traits<D>::is_gpu;
        _a.visit(visitor);

        visitor.need_value = decay_traits<D>::is_gpu;
        _b.visit(visitor);

        this->evaluate();

        if (old_need_value) {
            this->ensure_cpu_up_to_date();
        }

        visitor.need_value = old_need_value;
    }
};

} //end of namespace etl
