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

/*!
 * \brief A temporary expression base
 *
 * A temporary expression computes the expression directly and stores it into a temporary.
 */
template <typename D, typename V, typename R>
struct temporary_expr : value_testable<D>, dim_testable<D>, iterable<const D, true> {
    using derived_t         = D;                               ///< The derived type
    using value_type        = V;                               ///< The value type
    using result_type       = R;                               ///< The result type
    using memory_type       = value_type*;                     ///< The memory type
    using const_memory_type = const value_type*;               ///< The const memory type

protected:
    mutable bool allocated = false; ///< Indicates if the temporary has been allocated
    mutable bool evaluated = false; ///< Indicates if the expression has been evaluated

    mutable std::shared_ptr<result_type> _c;           ///< The result reference

public:
    temporary_expr() = default;

    temporary_expr(const temporary_expr& expr) = default;

    /*!
     * \brief Move construct a temporary_expr
     * The right hand side cannot be used anymore after ths move.
     * \param rhs The expression to move from.
     */
    temporary_expr(temporary_expr&& rhs) : allocated(rhs.allocated), evaluated(rhs.evaluated), _c(std::move(rhs._c)) {
        rhs.evaluated = false;
    }

    //Expressions are invariant
    temporary_expr& operator=(const temporary_expr& /*e*/) = delete;
    temporary_expr& operator=(temporary_expr&& /*e*/) = delete;

    /*!
     * \brief The vectorization type for V
     */
    template <typename VV = default_vec>
    using vec_type        = typename VV::template vec_type<value_type>;

private:
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
    void evaluate() const {
        if (!evaluated) {
            cpp_assert(allocated, "The result has not been allocated");
            as_derived().apply(*_c);
            evaluated = true;
        }
    }

    /*!
     * \brief Allocate the necessary temporaries, if necessary
     */
    void allocate_temporary() const {
        if (!_c) {
            _c.reset(as_derived().allocate());
        }

        allocated = true;
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
    template <typename... S, cpp_enable_if(sizeof...(S) == safe_dimensions<derived_t>::value)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to<size_t, S...>::value, "Invalid size types");

        return result()(args...);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <typename DD = D, cpp_enable_if((safe_dimensions<DD>::value > 1))>
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

    /*!
     * \brief Return GPU memory of this expression, if any.
     * \return a pointer to the GPU memory or nullptr if not allocated in GPU.
     */
    value_type* gpu_memory() const noexcept {
        cpp_assert(evaluated, "The result has not been evaluated");
        cpp_assert(allocated, "The result has not been allocated");
        return result().gpu_memory();
    }

    /*!
     * \brief Evict the expression from GPU.
     */
    void gpu_evict() const noexcept {
        if(allocated && evaluated){
            result().gpu_evict();
        }
    }

    /*!
     * \brief Invalidates the CPU memory
     */
    void invalidate_cpu() const noexcept {
        if(allocated && evaluated){
            result().invalidate_cpu();
        }
    }

    /*!
     * \brief Invalidates the GPU memory
     */
    void invalidate_gpu() const noexcept {
        if(allocated && evaluated){
            result().invalidate_gpu();
        }
    }

    /*!
     * \brief Validates the CPU memory
     */
    void validate_cpu() const noexcept {
        if(allocated && evaluated){
            result().validate_cpu();
        }
    }

    /*!
     * \brief Validates the GPU memory
     */
    void validate_gpu() const noexcept {
        if(allocated && evaluated){
            result().validate_gpu();
        }
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_gpu_allocated() const {
        if(allocated && evaluated){
            result().ensure_gpu_allocated();
        }
    }

    /*!
     * \brief Allocate memory on the GPU for the expression and copy the values into the GPU.
     */
    void ensure_gpu_up_to_date() const {
        if(allocated && evaluated){
            result().ensure_gpu_up_to_date();
        }
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_cpu_up_to_date() const {
        if(allocated && evaluated){
            result().ensure_cpu_up_to_date();
        }
    }

    /*!
     * \brief Copy from GPU to GPU
     * \param gpu_memory Pointer to CPU memory
     */
    void gpu_copy_from(const value_type* gpu_memory) const {
        if(allocated && evaluated){
            result().gpu_copy_from(gpu_memory);
        }
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
};

/*!
 * \brief Abstrct base class for temporary binary expression
 * \tparam D The derived type
 * \tparam T The value type
 * \tparam A The left sub expression type
 * \tparam B The right sub expression type
 * \tparam R The result type, if forced (void otherwise)
 */
template <typename D, typename T, typename A, typename B, typename R>
struct temporary_expr_bin : temporary_expr<D, T, R> {
    static_assert(is_etl_expr<A>::value, "The argument must be an ETL expr");
    static_assert(is_etl_expr<B>::value, "The argument must be an ETL expr");

    using value_type  = T;                                                    ///< The type of value
    using result_type = R;                                                    ///< The result type
    using this_type   = temporary_expr_bin<D, value_type, A, B, result_type>; ///< This type
    using base_type   = temporary_expr<D, T, R>;                              ///< The base type

    A _a;                       ///< The sub expression reference
    B _b;                       ///< The sub expression reference

    /*!
     * \brief Construct a new expression
     * \param a The left sub expression
     * \param b The right sub expression
     */
    explicit temporary_expr_bin(A a, B b) : _a(a), _b(b) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new expression by copy
     * \param e The expression to copy
     */
    temporary_expr_bin(const temporary_expr_bin& e) : base_type(e), _a(e._a), _b(e._b) {
        //Nothing else to init
    }

    /*!
     * \brief Construct a new expression by move
     * \param e The expression to move
     */
    temporary_expr_bin(temporary_expr_bin&& e) noexcept : base_type(std::move(e)), _a(e._a), _b(e._b) {
        //Nothing else to init
    }

    /*!
     * \brief Returns the left-hand-side expression
     * \return a reference to the left hand side expression
     */
    std::add_lvalue_reference_t<A> a() {
        return _a;
    }

    /*!
     * \brief Returns the left-hand-side expression
     * \return a reference to the left hand side expression
     */
    cpp::add_const_lvalue_t<A> a() const {
        return _a;
    }

    /*!
     * \brief Returns the right-hand-side expression
     * \return a reference to the right hand side expression
     */
    std::add_lvalue_reference_t<B> b() {
        return _b;
    }

    /*!
     * \brief Returns the right-hand-side expression
     * \return a reference to the right hand side expression
     */
    cpp::add_const_lvalue_t<B> b() const {
        return _b;
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

    // Internals

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor) const {
        this->allocate_temporary();

        _a.visit(visitor);
        _b.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::back_propagate_visitor& visitor) const {
        _a.visit(visitor);
        _b.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
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

/*!
 * \brief A temporary binary expression
 */
template <typename T, typename AExpr, typename BExpr, typename Op>
struct temporary_binary_expr final : temporary_expr_bin<temporary_binary_expr<T, AExpr, BExpr, Op>, T, AExpr, BExpr, typename Op::template result_type<AExpr, BExpr>> {
    using value_type  = T;                                                                    ///< The value type
    using result_type = typename Op::template result_type<AExpr, BExpr>;                      ///< The result type
    using this_type   = temporary_binary_expr<T, AExpr, BExpr, Op>;                           ///< The type of this expresion
    using base_type   = temporary_expr_bin<this_type, value_type, AExpr, BExpr, result_type>; ///< The base type

    using left_traits  = decay_traits<AExpr>; ///< Traits of the left-hand-side of the expression
    using right_traits = decay_traits<BExpr>; ///< Traits of the right-hand-side of the expression

    static constexpr auto storage_order = left_traits::is_generator ? right_traits::storage_order : left_traits::storage_order; ///< The storage order

    /*!
     * \brief Construct a new expression
     * \param a The left expression
     * \param b The right expression
     */
    temporary_binary_expr(AExpr a, BExpr b) : base_type(a, b) {
        //Nothing else to init
    }

    /*!
     * \brief Apply the op and store the result in result
     * \param result The expressio where to store the result
     */
    template <typename Result>
    void apply(Result&& result) const {
        Op::apply(this->a(), this->b(), std::forward<Result>(result));
    }

    /*!
     * \brief Allocate memory for this operation
     * \return an allocated ETL expression
     */
    auto allocate() const {
        return Op::allocate(this->a(), this->b());
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template<typename L, cpp_enable_if(decay_traits<L>::storage_order == storage_order)>
    void assign_to(L&& lhs)  const {
        standard_evaluator::pre_assign_rhs(this->a());
        standard_evaluator::pre_assign_rhs(this->b());
        standard_evaluator::pre_assign_lhs(lhs);

        apply(lhs);
    }

    /*!
     * \brief Assign to a matrix of a different storage order
     * \param lhs The expression to which assign
     */
    template<typename L, cpp_enable_if(decay_traits<L>::storage_order != storage_order)>
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

    /*!
     * \brief Prints a description of the temporary binary expr to the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const temporary_binary_expr& expr) {
        return os << Op::desc() << "(" << expr.a() << ", " << expr.b() << ")";
    }
};

/*!
 * \brief A temporary binary expression with state
 */
template <typename T, typename AExpr, typename BExpr, typename Op>
struct temporary_binary_expr_state final : temporary_expr_bin<temporary_binary_expr_state<T, AExpr, BExpr, Op>, T, AExpr, BExpr, typename Op::template result_type<AExpr, BExpr>> {
    using value_type  = T;                                                                    ///< The value type
    using result_type = typename Op::template result_type<AExpr, BExpr>;                      ///< The result type
    using this_type   = temporary_binary_expr_state<T, AExpr, BExpr, Op>;                     ///< The type of this expresion
    using base_type   = temporary_expr_bin<this_type, value_type, AExpr, BExpr, result_type>; ///< The base type

    using left_traits  = decay_traits<AExpr>; ///< Traits of the left-hand-side of the expression
    using right_traits = decay_traits<BExpr>; ///< Traits of the right-hand-side of the expression

    static constexpr auto storage_order = left_traits::is_generator ? right_traits::storage_order : left_traits::storage_order; ///< The storage order

    Op op; ///< The stateful operation

    /*!
     * \brief Construct a new expression
     * \param op The operator
     * \param a The left expression
     * \param b The right expression
     */
    temporary_binary_expr_state(Op op, AExpr a, BExpr b) : base_type(a, b), op(op) {
        //Nothing else to init
    }

    /*!
     * \brief Apply the op and store the result in result
     * \param result The expressio where to store the result
     */
    template <typename Result>
    void apply(Result&& result) const {
        op.apply(this->a(), this->b(), std::forward<Result>(result));
    }

    /*!
     * \brief Allocate memory for this operation
     * \return an allocated ETL expression
     */
    auto allocate() const {
        return op.allocate(this->a(), this->b());
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template<typename L, cpp_enable_if(decay_traits<L>::storage_order == storage_order)>
    void assign_to(L&& lhs)  const {
        standard_evaluator::pre_assign_rhs(this->a());
        standard_evaluator::pre_assign_rhs(this->b());
        standard_evaluator::pre_assign_lhs(lhs);

        apply(lhs);
    }

    /*!
     * \brief Assign to a matrix of a different storage order
     * \param lhs The expression to which assign
     */
    template<typename L, cpp_enable_if(decay_traits<L>::storage_order != storage_order)>
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
};

/*!
 * \brief Specialization for temporary_binary_expr.
 */
template <typename T, typename A, typename B, typename Op>
struct etl_traits<etl::temporary_binary_expr<T, A, B, Op>> {
    using expr_t     = etl::temporary_binary_expr<T, A, B, Op>; ///< The expression type
    using a_t        = std::decay_t<A>;                         ///< The decayed left expression type
    using b_t        = std::decay_t<B>;                         ///< The decayed right expression type
    using value_type = T;                                       ///< The value type

    static constexpr bool is_etl                  = true;                                                                                            ///< Indicates if the type is an ETL type
    static constexpr bool is_transformer          = false;                                                                                           ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                                                                                           ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                                                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = etl_traits<a_t>::is_fast && etl_traits<b_t>::is_fast;                                            ///< Indicates if the expression is fast
    static constexpr bool is_direct               = true;                                                                                            ///< Indicates if the expression has direct memory access
    static constexpr bool is_linear               = true;                                                                                            ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                                                                                            ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                                                                           ///< Indicates if the expression is of value type
    static constexpr bool is_generator            = false;                                                                                           ///< Indicates if the expression is a generated
    static constexpr bool needs_evaluator = true;                                                                                            ///< Indicaes if the expression needs an evaluator visitor
    static constexpr bool is_padded               = false;                                                                                           ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = true;                                                                                            ///< Indicates if the expression is padded
    static constexpr order storage_order          = etl_traits<a_t>::is_generator ? etl_traits<b_t>::storage_order : etl_traits<a_t>::storage_order; ///< The expression storage order
    static constexpr bool is_gpu                  = Op::is_gpu;                                                                                      ///< Indicate if the expression is computed on GPU

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) {
        return Op::size(v.a(), v.b());
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, size_t d) {
        return Op::dim(v.a(), v.b(), d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() {
        return Op::template size<a_t, b_t>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <size_t D>
    static constexpr size_t dim() {
        return Op::template dim<a_t, b_t, D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return Op::dimensions();
    }
};

/*!
 * \brief Specialization for temporary_binary_expr_state.
 */
template <typename T, typename A, typename B, typename Op>
struct etl_traits<etl::temporary_binary_expr_state<T, A, B, Op>> {
    using expr_t     = etl::temporary_binary_expr_state<T, A, B, Op>; ///< The type of expression
    using a_t        = std::decay_t<A>;                               ///< The decayed left expression type
    using b_t        = std::decay_t<B>;                               ///< The decayed right expression type
    using value_type = T;                                             ///< The value type

    static constexpr bool is_etl                  = true;                                                                                            ///< Indicates if the type is an ETL type
    static constexpr bool is_transformer          = false;                                                                                           ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = false;                                                                                           ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                                                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = false;                                                                                           ///< Indicates if the expression is fast
    static constexpr bool is_direct               = true;                                                                                            ///< Indicates if the expression has direct memory access
    static constexpr bool is_linear               = true;                                                                                            ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = true;                                                                                            ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                                                                           ///< Indicates if the expression is of value type
    static constexpr bool is_generator            = false;                                                                                           ///< Indicates if the expression is a generated
    static constexpr bool needs_evaluator = true;                                                                                            ///< Indicaes if the expression needs an evaluator visitor
    static constexpr bool is_padded               = false;                                                                                           ///< Indicates if the expression is padded
    static constexpr bool is_aligned              = true;                                                                                            ///< Indicates if the expression is padded
    static constexpr order storage_order          = etl_traits<a_t>::is_generator ? etl_traits<b_t>::storage_order : etl_traits<a_t>::storage_order; ///< The expression storage order
    static constexpr bool is_gpu                  = Op::is_gpu;                                                                                      ///< Indicate if the expression is computed on GPU

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) {
        return v.op.size(v.a(), v.b());
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, size_t d) {
        return v.op.dim(v.a(), v.b(), d);
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return Op::dimensions();
    }
};

} //end of namespace etl
