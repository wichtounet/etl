//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

/*!
 * \brief A binary expression
 *
 * A binary expression has a left hand side expression and a right hand side expression and for each element applies a binary opeartor to both expressions.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct binary_expr final : dim_testable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>>,
                           value_testable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>>,
                           iterable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
private:
    static_assert((std::is_same_v<LeftExpr, scalar<T>> && std::is_same_v<RightExpr, scalar<T>>)
                      || (is_etl_expr<LeftExpr> && std::is_same_v<RightExpr, scalar<T>>)
                      || (is_etl_expr<RightExpr> && std::is_same_v<LeftExpr, scalar<T>>) || (all_etl_expr<LeftExpr, RightExpr>),
                  "One argument must be an ETL expression and the other one convertible to T");

    using this_type = binary_expr<T, LeftExpr, BinaryOp, RightExpr>; ///< This type

    LeftExpr lhs;  ///< The Left hand side expression
    RightExpr rhs; ///< The right hand side expression

    friend struct etl_traits<binary_expr>;
    friend struct optimizer<binary_expr>;
    friend struct optimizable<binary_expr>;
    friend struct transformer<binary_expr>;

public:
    using value_type        = T;                              ///< The Value type
    using memory_type       = void;                           ///< The memory type
    using const_memory_type = void;                           ///< The const memory type
    using iterator          = etl::iterator<const this_type>; ///< The iterator type
    using const_iterator    = etl::iterator<const this_type>; ///< The const iterator type
    using operator_type     = BinaryOp;                       ///< The binary operator type

    using left_type  = std::decay_t<LeftExpr>;  ///< The LHS side type
    using right_type = std::decay_t<RightExpr>; ///< The RHS side type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

    //Cannot be constructed with no args
    binary_expr() = delete;

    /*!
     * \brief Construct a new binary expression
     * \param l The left hand side of the expression
     * \param r The right hand side of the expression
     */
    binary_expr(LeftExpr l, RightExpr r) : lhs(std::forward<LeftExpr>(l)), rhs(std::forward<RightExpr>(r)) {
        //Nothing else to init
    }

    /*!
     * \brief Copy construct a new binary expression
     * \param e The expression from which to copy
     */
    binary_expr(const binary_expr& e) = default;

    /*!
     * \brief Move construct a new binary expression
     * \param e The expression from which to move
     */
    binary_expr(binary_expr&& e) noexcept = default;

    //Expressions are invariant
    binary_expr& operator=(const binary_expr& e) = delete;
    binary_expr& operator=(binary_expr&& e) = delete;

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param other The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& other) const noexcept {
        return lhs.alias(other) || rhs.alias(other);
    }

    //Apply the expression

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](size_t i) const {
        return BinaryOp::apply(lhs[i], rhs[i]);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t i) const {
        return BinaryOp::apply(lhs.read_flat(i), rhs.read_flat(i));
    }

    /*!
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam V The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>)
    load(size_t i) const {
        return BinaryOp::template load<V>(lhs.template load<V>(i), rhs.template load<V>(i));
    }

    /*!
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam V The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>)
    loadu(size_t i) const {
        return BinaryOp::template load<V>(lhs.template loadu<V>(i), rhs.template loadu<V>(i));
    }

    /*!
     * \brief Returns the value at the given position (args...)
     * \param args The position indices
     * \return The value at the given position (args...)
     */
    template <typename... S, cpp_enable_iff(sizeof...(S) == safe_dimensions<this_type>)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to_v<size_t, S...>, "Invalid size types");

        return BinaryOp::apply(lhs(args...), rhs(args...));
    }

    /*!
     * \brief Creates a sub view of the expression, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the expression at position i.
     */
    template <bool B = (safe_dimensions<this_type>> 1), cpp_enable_iff(B)>
    auto operator()(size_t i) {
        return sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the expression, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the expression at position i.
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
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    decltype(auto) gpu_compute_hint(Y& y) const {
        return BinaryOp::gpu_compute_hint(lhs, rhs, y);
    }

    /*!
     * \brief Return a GPU computed version of this expression
     * \return a GPU-computed ETL expression for this expression
     */
    template <typename Y>
    decltype(auto) gpu_compute(Y& y) const {
        return BinaryOp::gpu_compute(lhs, rhs, y);
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
        lhs.visit(visitor);
        rhs.visit(visitor);
    }

    /*!
     * \brief Ensures that the GPU memory is allocated and that the GPU memory
     * is up to date (to undefined value).
     */
    void ensure_cpu_up_to_date() const {
        // Need to ensure both LHS and RHS
        lhs.ensure_cpu_up_to_date();
        rhs.ensure_cpu_up_to_date();
    }

    /*!
     * \brief Copy back from the GPU to the expression memory if
     * necessary.
     */
    void ensure_gpu_up_to_date() const {
        // Need to ensure both LHS and RHS
        lhs.ensure_gpu_up_to_date();
        rhs.ensure_gpu_up_to_date();
    }

    /*!
     * \brief Returns a reference to the left hand side of the expression
     * \return a reference to the left hand side of the expression
     */
    const LeftExpr& get_lhs() const {
        return lhs;
    }

    /*!
     * \brief Returns a reference to the right hand side of the expression
     * \return a reference to the right hand side of the expression
     */
    const RightExpr& get_rhs() const {
        return rhs;
    }

    /*!
     * \brief Prints the type of the binary expression to the stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const binary_expr& expr) {
        if constexpr (BinaryOp::desc_func) {
            return os << BinaryOp::desc() << "(" << expr.lhs << ", " << expr.rhs << ")";
        } else {
            return os << "(" << expr.lhs << ' ' << BinaryOp::desc() << ' ' << expr.rhs << ")";
        }
    }
};

/*!
 * \brief Specialization for binary_expr.
 */
template <typename T, typename LE, typename BinaryOp, typename RE>
struct etl_traits<etl::binary_expr<T, LE, BinaryOp, RE>> {
    using expr_t       = etl::binary_expr<T, LE, BinaryOp, RE>; ///< The type of the expression
    using left_expr_t  = std::decay_t<LE>;                      ///< The type of the left expression
    using right_expr_t = std::decay_t<RE>;                      ///< The type of the right expression
    using value_type   = T;                                     ///< The value type

    static constexpr bool left_directed =
        !etl_traits<left_expr_t>::is_generator; ///< True if directed by the left expression, false otherwise

    using sub_expr_t = std::conditional_t<left_directed, left_expr_t, right_expr_t>; ///< The type of sub expression

    using sub_traits   = etl_traits<sub_expr_t>;   ///< The sub traits
    using left_traits  = etl_traits<left_expr_t>;  ///< The left traits
    using right_traits = etl_traits<right_expr_t>; ///< The right traits

    static constexpr bool is_etl         = true;                ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;               ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;               ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;               ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast; ///< Indicates if the expression is fast
    static constexpr bool is_linear      = left_traits::is_linear && right_traits::is_linear && BinaryOp::linear; ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe =
        left_traits::is_thread_safe && right_traits::is_thread_safe && BinaryOp::thread_safe;     ///< Indicates if the expression is linear
    static constexpr bool is_value     = false;                                                   ///< Indicates if the expression is of value type
    static constexpr bool is_direct    = false;                                                   ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator = left_traits::is_generator && right_traits::is_generator; ///< Indicates if the expression is a generator expression
    static constexpr bool is_temporary = left_traits::is_temporary || right_traits::is_temporary; ///< Indicates if the expression needs an evaluator visitor
    static constexpr bool is_padded    = is_linear && left_traits::is_padded && right_traits::is_padded;   ///< Indicates if the expression is padded
    static constexpr bool is_aligned   = is_linear && left_traits::is_aligned && right_traits::is_aligned; ///< Indicates if the expression is padded
    static constexpr order storage_order =
        left_traits::is_generator ? right_traits::storage_order : left_traits::storage_order; ///< The expression storage order

    /*!
     * \brief Indicates if the expression can be computed on GPU
     */
    static constexpr bool gpu_computable = all_gpu_computable<LE, RE> && BinaryOp::template gpu_computable<LE, RE> && all_homogeneous<LE, RE>;

    template <vector_mode_t V>
    static constexpr bool vectorizable =
        all_homogeneous<LE, RE>&& left_traits::template vectorizable<V>&& right_traits::template vectorizable<V>&& BinaryOp::template vectorizable<V>;

    /*!
     * \brief Get reference to the main sub expression
     * \param v The binary expr
     * \return a refernece to the main sub expression
     */
    static constexpr auto& get(const expr_t& v) {
        if constexpr (left_directed) {
            return v.lhs;
        } else {
            return v.rhs;
        }
    }

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) {
        return sub_traits::size(get(v));
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, size_t d) {
        return sub_traits::dim(get(v), d);
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
