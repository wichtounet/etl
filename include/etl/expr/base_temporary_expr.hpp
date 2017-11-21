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
 */
template <typename E, bool Fast>
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
template <bool Fast, typename E>
using expr_result_t = typename expr_result<E, Fast && is_fast<E>>::type;

} // end of temporary_detail

/*!
 * \brief A temporary expression base
 *
 * \tparam D The derived type
 *
 * A temporary expression computes the expression directly and stores it into a temporary.
 */
template <typename D, bool Fast>
struct base_temporary_expr : value_testable<D>, dim_testable<D>, iterable<const D, true> {
    using derived_t         = D;                                        ///< The derived type
    using value_type        = typename decay_traits<D>::value_type;     ///< The value type
    using result_type       = temporary_detail::expr_result_t<Fast, D>; ///< The result type
    using memory_type       = value_type*;                              ///< The memory type
    using const_memory_type = const value_type*;                        ///< The const memory type

protected:
    mutable std::shared_ptr<bool> evaluated; ///< Indicates if the expression has been evaluated
    mutable std::shared_ptr<result_type> _c; ///< The result reference

public:
    /*!
     * \brief Construct a new base_temporary_expr
     */
    base_temporary_expr() : evaluated(std::make_shared<bool>(false)) {
        // Nothing else to init
    }

    /*!
     * \brief Copy construct a new base_temporary_expr
     */
    base_temporary_expr(const base_temporary_expr& expr) = default;

    /*!
     * \brief Move construct a base_temporary_expr
     * The right hand side cannot be used anymore after ths move.
     * \param rhs The expression to move from.
     */
    base_temporary_expr(base_temporary_expr&& rhs) : evaluated(std::move(rhs.evaluated)), _c(std::move(rhs._c)) {
        //Nothing else to change
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

    /*!
     * \brief Indicates if the temporary has been allocated
     * \return true if the temporary has been allocated, false
     * otherwise
     */
    bool is_allocated() const noexcept {
        return _c.get();
    }

    /*!
     * \brief Indicates if the temporary has been evaluated
     *
     * \return true if the temporary has been evaluted, false otherwise
     */
    bool is_evaluated() const noexcept {
        return *evaluated;
    }

protected:
    /*!
     * \brief Evaluate the expression, if not evaluated
     *
     * Will fail if not previously allocated
     */
    void evaluate() const {
        if (!*evaluated) {
            cpp_assert(is_allocated(), "The result has not been allocated");
            as_derived().assign_to(*_c);
            *evaluated = true;
        }
    }

    /*!
     * \brief Allocate the necessary temporaries, if necessary
     */
    void allocate_temporary() const {
        if (!_c) {
            _c.reset(allocate());
        }
    }

    /*!
     * \brief Allocate the temporary
     */
    template <bool B = is_fast<derived_t>, cpp_enable_iff(B)>
    result_type* allocate() const {
        return new result_type;
    }

    /*!
     * \brief Allocate the dynamic temporary
     */
    template <size_t... I>
    result_type* dyn_allocate(std::index_sequence<I...> /*seq*/) const {
        return new result_type(decay_traits<derived_t>::dim(as_derived(), I)...);
    }

    /*!
     * \brief Allocate the temporary
     */
    template <bool B = is_fast<derived_t>, cpp_disable_iff(B)>
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
    value_type operator[](size_t i) const {
        return result()[i];
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return result().read_flat(i);
    }

    /*!
     * \brief Returns the value at the given position (args...)
     * \param args The position indices
     * \return The value at the given position (args...)
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == safe_dimensions<derived_t>)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return result()(args...);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <typename DD = D, cpp_enable_iff(safe_dimensions<DD> > 1)>
    auto operator()(size_t i) const {
        return sub(as_derived(), i);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(size_t first, size_t last) noexcept {
        return slice(*this, first, last);
    }

    /*!
     * \brief Creates a slice view of the matrix, effectively reducing the first dimension.
     * \param first The first index to use
     * \param last The last index to use
     * \return a slice view of the matrix at position i.
     */
    auto slice(size_t first, size_t last) const noexcept {
        return slice(*this, first, last);
    }

    /*!
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam VV The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename VV = default_vec>
    vec_type<VV> load(size_t i) const noexcept {
        return VV::loadu(memory_start() + i);
    }

    /*!
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam VV The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename VV = default_vec>
    vec_type<VV> loadu(size_t i) const noexcept {
        return VV::loadu(memory_start() + i);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    auto& gpu_compute_hint(Y& y){
        cpu_unused(y);
        this->ensure_gpu_up_to_date();
        return as_derived();
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    const auto& gpu_compute_hint(Y& y) const {
        cpu_unused(y);
        this->ensure_gpu_up_to_date();
        return as_derived();
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
        return result().gpu_memory();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        result().gpu_evict();
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        result().invalidate_cpu();
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        result().invalidate_gpu();
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        result().validate_cpu();
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        result().validate_gpu();
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        result().ensure_gpu_allocated();
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        result().ensure_gpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        result().ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const value_type* gpu_memory) const {
        result().gpu_copy_from(gpu_memory);
    }

    /*!
     * \brief Indicates if the CPU memory is up to date.
     * \return true if the CPU memory is up to date, false otherwise.
     */
    bool is_cpu_up_to_date() const noexcept {
        return result().is_cpu_up_to_date();
    }

    /*!
     * \brief Indicates if the GPU memory is up to date.
     * \return true if the GPU memory is up to date, false otherwise.
     */
    bool is_gpu_up_to_date() const noexcept {
        return result().is_gpu_up_to_date();
    }

protected:
    /*!
     * \brief Returns the expression containing the result of the expression.
     * \return a reference to the expression containing the result of the expression
     */
    result_type& result() {
        cpp_assert(is_allocated(), "The result has not been allocated");
        cpp_assert(*evaluated, "The result has not been evaluated");
        return *_c;
    }

    /*!
     * \brief Returns the expression containing the result of the expression.
     * \return a const reference to the expression containing the result of the expression
     */
    const result_type& result() const {
        cpp_assert(is_allocated(), "The result has not been allocated");
        cpp_assert(*evaluated, "The result has not been evaluated");
        return *_c;
    }
};

/*!
 * \brief Abstract base class for temporary unary expression
 * \tparam D The derived type
 * \tparam A The sub type
 */
template <typename D, typename A, bool Fast = true>
struct base_temporary_expr_un : base_temporary_expr<D, Fast> {
    static_assert(is_etl_expr<A>, "The argument must be an ETL expr");

    using this_type = base_temporary_expr_un<D, A>; ///< This type
    using base_type = base_temporary_expr<D, Fast>;       ///< The base type

    A _a;                       ///< The sub expression reference

    using base_type::evaluated;

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
    void visit(detail::evaluator_visitor& visitor) const {
        // If the expression is already evaluated, no need to
        // recurse through the tree
        if(*evaluated){
            return;
        }

        this->allocate_temporary();

        _a.visit(visitor);

        this->evaluate();
    }
};

/*!
 * \brief Abstract base class for temporary binary expression
 * \tparam D The derived type
 * \tparam A The left sub expression type
 * \tparam B The right sub expression type
 */
template <typename D, typename A, typename B, bool Fast = true>
struct base_temporary_expr_bin : base_temporary_expr<D, Fast> {
    static_assert(is_etl_expr<A>, "The argument must be an ETL expr");
    static_assert(is_etl_expr<B>, "The argument must be an ETL expr");

    using this_type = base_temporary_expr_bin<D, A, B>; ///< This type
    using base_type = base_temporary_expr<D, Fast>;           ///< The base type

    A _a;                       ///< The sub expression reference
    B _b;                       ///< The sub expression reference

    using base_type::evaluated;

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
    void visit(detail::evaluator_visitor& visitor) const {
        // If the expression is already evaluated, no need to
        // recurse through the tree
        if(*evaluated){
            return;
        }

        this->allocate_temporary();

        _a.visit(visitor);
        _b.visit(visitor);

        this->evaluate();
    }
};

/*!
 * \brief Abstract base class for temporary ternary expression
 * \tparam D The derived type
 * \tparam A The left sub expression type
 * \tparam B The right sub expression type
 */
template <typename D, typename A, typename B, typename C, bool Fast = true>
struct base_temporary_expr_tern : base_temporary_expr<D, Fast> {
    static_assert(is_etl_expr<A>, "The argument must be an ETL expr");
    static_assert(is_etl_expr<B>, "The argument must be an ETL expr");
    static_assert(is_etl_expr<C>, "The argument must be an ETL expr");

    using this_type = base_temporary_expr_tern<D, A, B, C>; ///< This type
    using base_type = base_temporary_expr<D, Fast>;         ///< The base type

    A _a;                       ///< The first sub expression reference
    B _b;                       ///< The second sub expression reference
    C _c;                       ///< The third sub expression reference

    using base_type::evaluated;

public:

    /*!
     * \brief Construct a new expression
     * \param a The first sub expression
     * \param b The second sub expression
     * \param c The third sub expression
     */
    base_temporary_expr_tern(A a, B b, C c) : _a(a), _b(b), _c(c) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new expression by copy
     * \param e The expression to copy
     */
    base_temporary_expr_tern(const base_temporary_expr_tern& e) : base_type(e), _a(e._a), _b(e._b), _c(e._c) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new expression by move
     * \param e The expression to move
     */
    base_temporary_expr_tern(base_temporary_expr_tern&& e) noexcept : base_type(std::move(e)), _a(e._a), _b(e._b), _c(e._c) {
        //Nothing else to init
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const {
        return _a.alias(rhs) || _b.alias(rhs) || _c.alias(rhs);
    }

protected:

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

    /*!
     * \brief Returns the sub expression
     * \return a reference to the sub expression
     */
    std::add_lvalue_reference_t<C> c() {
        return _c;
    }

    /*!
     * \brief Returns the sub expression
     * \return a reference to the sub expression
     */
    cpp::add_const_lvalue_t<C> c() const {
        return _c;
    }

public:

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        // If the expression is already evaluated, no need to
        // recurse through the tree
        if(*evaluated){
            return;
        }

        this->allocate_temporary();

        _a.visit(visitor);
        _b.visit(visitor);
        _c.visit(visitor);

        this->evaluate();
    }
};

} //end of namespace etl
