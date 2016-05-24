//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include <memory> //For shared_ptr

#include "etl/iterator.hpp"

namespace etl {

/*!
 * \brief Simple utility wrapper for shared_ptr that is mutable.
 *
 * This is necessary because *mutable* references are not possible and therefore
 * cannot simply put mutable DataType inside temporary expression.
 */
template<typename T>
struct mutable_shared_ptr {
private:
    mutable std::shared_ptr<T> ptr; ///< The pointer

public:
    mutable_shared_ptr() = default;
    mutable_shared_ptr(const mutable_shared_ptr& rhs) = default;
    mutable_shared_ptr(mutable_shared_ptr&& rhs) = default;
    mutable_shared_ptr& operator=(const mutable_shared_ptr& rhs) = default;
    mutable_shared_ptr& operator=(mutable_shared_ptr&& rhs) = default;

    /*!
     * \brief Constructs a new mutable_shared_ptr from a shared_ptr.
     * \param ptr The pointer to copy inside mutable_shared_ptr
     */
    mutable_shared_ptr(const std::shared_ptr<T>& ptr) : ptr(ptr) {}

    /*!
     * \brief Resets the pointer to a new value.
     * \param new_value The new value of the pointer
     */
    void reset(T* new_value) const {
        ptr.reset(new_value);
    }

    /*!
     * \brief Explicit conversion to bool
     * \return false if the pointer is nullptr, false otherwise
     */
    explicit operator bool() const {
        return static_cast<bool>(ptr);
    }

    /*!
     * \brief Returns the underlying object
     * \return a reference to the underlying object
     */
    T& operator*() const {
        return *ptr;
    }

    /*!
     * \brief Returns the underlying pointer
     * \return a pointer to the underlying object
     */
    T* operator->() const {
        return ptr.get();
    }
};


/*!
 * \brief A temporary expression base
 *
 * A temporary expression computes the expression directly and stores it into a temporary.
 */
template <typename D, typename V, typename R>
struct temporary_expr : comparable<D>, value_testable<D>, dim_testable<D> {
    using derived_t         = D;                 ///< The derived type
    using value_type        = V;                 ///< The value type
    using result_type       = R;                 ///< The result type
    using memory_type       = value_type*;       ///< The memory type
    using const_memory_type = const value_type*; ///< The const memory type
    using data_type   = mutable_shared_ptr<result_type>;                    ///< The data type

protected:
    mutable bool allocated = false; ///< Indicates if the temporary has been allocated
    mutable bool evaluated = false; ///< Indicates if the expression has been evaluated

    data_type _c;           ///< The result reference

private:
    mutable gpu_handler<V> _gpu_memory_handler;

public:
    temporary_expr() = default;

    temporary_expr(const temporary_expr& expr) = default;

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


    /*!
     * \brief Evaluate the expression directly into the given result
     *
     * Will fail if not previously allocated
     */
    template <typename Result>
    void direct_evaluate(Result&& result) const {
        as_derived().apply(std::forward<Result>(result));
    }

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
    template <typename... S, cpp_enable_if(sizeof...(S) == sub_size_compare<derived_t>::value)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return result()(args...);
    }

    /*!
     * \brief Creates a sub view of the matrix, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the matrix at position i.
     */
    template <typename DD = D, cpp_enable_if((sub_size_compare<DD>::value > 1))>
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
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam VV The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename VV = default_vec>
    vec_type<VV> load(std::size_t i) const noexcept {
        return VV::loadu(memory_start() + i);
    }

    // Iterator

    /*!
     * \brief Return an iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    iterator<const derived_t> begin() const noexcept {
        return {as_derived(), 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    iterator<const derived_t> end() const noexcept {
        return {as_derived(), size(as_derived())};
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

    auto direct() const {
        if(evaluated && allocated){
            return result().direct();
        } else {
            using result_type = decltype(result().direct());
            return result_type(nullptr, 0, {{}}, _gpu_memory_handler);
        }
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
};

template <typename D, typename T, typename A, typename R>
struct temporary_expr_un : temporary_expr<D, T, R> {
    static_assert(is_etl_expr<A>::value, "The argument must be an ETL expr");

    using value_type  = T;
    using result_type = R;
    using this_type   = temporary_expr_un<D, value_type, A, result_type>;
    using base_type   = temporary_expr<D, T, R>;

    A _a;                       ///< The sub expression reference

    //Construct a new expression
    explicit temporary_expr_un(A a) : _a(a) {
        //Nothing else to init
    }

    //Copy an expression
    temporary_expr_un(const temporary_expr_un& e) : base_type(e), _a(e._a) {
        //Nothing else to init
    }

    //Move an expression
    temporary_expr_un(temporary_expr_un&& e) noexcept : base_type(std::move(e)), _a(e._a){
        //Nothing else to init
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
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const {
        return a().alias(rhs);
    }
};

template <typename D, typename T, typename A, typename B, typename R>
struct temporary_expr_bin : temporary_expr<D, T, R> {
    static_assert(is_etl_expr<A>::value, "The argument must be an ETL expr");
    static_assert(is_etl_expr<B>::value, "The argument must be an ETL expr");

    using value_type  = T;
    using result_type = R;
    using this_type   = temporary_expr_bin<D, value_type, A, B, result_type>;
    using base_type   = temporary_expr<D, T, R>;

    A _a;                       ///< The sub expression reference
    B _b;                       ///< The sub expression reference

    //Construct a new expression
    explicit temporary_expr_bin(A a, B b) : _a(a), _b(b) {
        //Nothing else to init
    }

    //Copy an expression
    temporary_expr_bin(const temporary_expr_bin& e) : base_type(e), _a(e._a), _b(e._b) {
        //Nothing else to init
    }

    //Move an expression
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
};

/*!
 * \brief A temporary unary expression
 *
 * Evaluation is done at once, when access is made. This can be done
 * on const reference, this is the reason why several fields are
 * mutable.
 */
template <typename T, typename AExpr, typename Op>
struct temporary_unary_expr final : temporary_expr_un<temporary_unary_expr<T, AExpr, Op>, T, AExpr, typename Op::template result_type<AExpr>> {
    using value_type  = T;                                                ///< The value type
    using result_type = typename Op::template result_type<AExpr>;         ///< The result type
    using this_type   = temporary_unary_expr<T, AExpr, Op>;               ///< The type of this expression
    using base_type   = temporary_expr_un<this_type, T, AExpr, result_type>; ///< The base type

    //Construct a new expression
    explicit temporary_unary_expr(AExpr a) : base_type(a) {
        //Nothing else to init
    }

    //Accessors

    template <typename Result>
    void apply(Result&& result) const {
        Op::apply(this->a(), std::forward<Result>(result));
    }

    auto allocate() const {
        return Op::allocate(this->a());
    }
};

/*!
 * \brief A temporary binary expression
 */
template <typename T, typename AExpr, typename BExpr, typename Op>
struct temporary_binary_expr final : temporary_expr_bin<temporary_binary_expr<T, AExpr, BExpr, Op>, T, AExpr, BExpr, typename Op::template result_type<AExpr, BExpr>> {
    using value_type  = T;                                                  ///< The value type
    using result_type = typename Op::template result_type<AExpr, BExpr>;    ///< The result type
    using this_type   = temporary_binary_expr<T, AExpr, BExpr, Op>;         ///< The type of this expresion
    using base_type   = temporary_expr_bin<this_type, value_type, AExpr, BExpr, result_type>; ///< The base type

    //Construct a new expression
    temporary_binary_expr(AExpr a, BExpr b) : base_type(a, b) {
        //Nothing else to init
    }

    template <typename Result>
    void apply(Result&& result) const {
        Op::apply(this->a(), this->b(), std::forward<Result>(result));
    }

    auto allocate() const {
        return Op::allocate(this->a(), this->b());
    }
};

/*!
 * \brief Specialization for temporary_unary_expr.
 */
template <typename T, typename A, typename Op>
struct etl_traits<etl::temporary_unary_expr<T, A, Op>> {
    using expr_t = etl::temporary_unary_expr<T, A, Op>;
    using a_t    = std::decay_t<A>;

    static constexpr const bool is_etl                  = true;                           ///< Indicates if the type is an ETL type
    static constexpr const bool is_transformer          = false;                          ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;                          ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                          ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<a_t>::is_fast;       ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = true;                           ///< Indicates if the expression is linear
    static constexpr const bool is_value                = false;                          ///< Indicates if the expression is of value type
    static constexpr const bool is_generator            = false;                          ///< Indicates if the expression is a generated
    static constexpr const bool needs_temporary_visitor = true;                           ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = true;                           ///< Indicaes if the expression needs an evaluator visitor
    static constexpr const order storage_order          = etl_traits<a_t>::storage_order; ///< The expression storage order
    static constexpr const bool is_gpu                  = Op::is_gpu; ///< Indicate if the expression is computed on GPU

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
    static std::size_t size(const expr_t& v) {
        return Op::size(v.a());
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return Op::dim(v.a(), d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return Op::template size<a_t>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return Op::template dim<a_t, D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return Op::dimensions();
    }
};

/*!
 * \brief Specialization for temporary_binary_expr.
 */
template <typename T, typename A, typename B, typename Op>
struct etl_traits<etl::temporary_binary_expr<T, A, B, Op>> {
    using expr_t = etl::temporary_binary_expr<T, A, B, Op>;
    using a_t    = std::decay_t<A>;
    using b_t    = std::decay_t<B>;

    static constexpr const bool is_etl                  = true;                                                                                            ///< Indicates if the type is an ETL type
    static constexpr const bool is_transformer          = false;                                                                                           ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;                                                                                           ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                                                                           ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<a_t>::is_fast && etl_traits<b_t>::is_fast;                                            ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = true;                                                                                            ///< Indicates if the expression is linear
    static constexpr const bool is_value                = false;                                                                                           ///< Indicates if the expression is of value type
    static constexpr const bool is_generator            = false;                                                                                           ///< Indicates if the expression is a generated
    static constexpr const bool needs_temporary_visitor = true;                                                                                            ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = true;                                                                                            ///< Indicaes if the expression needs an evaluator visitor
    static constexpr const order storage_order          = etl_traits<a_t>::is_generator ? etl_traits<b_t>::storage_order : etl_traits<a_t>::storage_order; ///< The expression storage order
    static constexpr const bool is_gpu                  = Op::is_gpu; ///< Indicate if the expression is computed on GPU

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
    static std::size_t size(const expr_t& v) {
        return Op::size(v.a(), v.b());
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return Op::dim(v.a(), v.b(), d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return Op::template size<a_t, b_t>();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return Op::template dim<a_t, b_t, D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return Op::dimensions();
    }
};

/*!
 * \brief Prints a description of the temporary unary expr to the given stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename T, typename AExpr, typename Op>
std::ostream& operator<<(std::ostream& os, const temporary_unary_expr<T, AExpr, Op>& expr) {
    return os << Op::desc() << "(" << expr.a() << ")";
}

/*!
 * \brief Prints a description of the temporary binary expr to the given stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename T, typename AExpr, typename BExpr, typename Op>
std::ostream& operator<<(std::ostream& os, const temporary_binary_expr<T, AExpr, BExpr, Op>& expr) {
    return os << Op::desc() << "(" << expr.a() << ", " << expr.b() << ")";
}

} //end of namespace etl
