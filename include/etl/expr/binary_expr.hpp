//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/iterator.hpp"

namespace etl {

/*!
 * \brief A binary expression
 *
 * A binary expression has a left hand side expression and a right hand side expression and for each element applies a binary opeartor to both expressions.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct binary_expr final : comparable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>>, dim_testable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>>, value_testable<binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
private:
    static_assert(cpp::or_c<
                      cpp::and_c<std::is_same<LeftExpr, scalar<T>>, std::is_same<RightExpr, scalar<T>>>,
                      cpp::and_c<is_etl_expr<LeftExpr>, std::is_same<RightExpr, scalar<T>>>,
                      cpp::and_c<is_etl_expr<RightExpr>, std::is_same<LeftExpr, scalar<T>>>,
                      cpp::and_c<is_etl_expr<LeftExpr>, is_etl_expr<RightExpr>>>::value,
                  "One argument must be an ETL expression and the other one convertible to T");

    using this_type = binary_expr<T, LeftExpr, BinaryOp, RightExpr>; ///< This type

    LeftExpr _lhs;  ///< The Left hand side expression
    RightExpr _rhs; ///< The right hand side expression

public:
    using value_type        = T;    ///< The Value type
    using memory_type       = void; ///< The memory type
    using const_memory_type = void; ///< The const memory type

    /*!
     * \brief The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    //Cannot be constructed with no args
    binary_expr() = delete;

    /*!
     * \brief Construct a new binary expression
     * \param l The left hand side of the expression
     * \param r The right hand side of the expression
     */
    binary_expr(LeftExpr l, RightExpr r)
            : _lhs(std::forward<LeftExpr>(l)), _rhs(std::forward<RightExpr>(r)) {
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
     * \brief Returns the left hand side expression on which the transformer is working.
     * \return A reference to the left hand side expression on which the transformer is working.
     */
    std::add_lvalue_reference_t<LeftExpr> lhs() {
        return _lhs;
    }

    /*!
     * \brief Returns the left hand side expression on which the transformer is working.
     * \return A reference to the left hand side expression on which the transformer is working.
     */
    cpp::add_const_lvalue_t<LeftExpr> lhs() const {
        return _lhs;
    }

    /*!
     * \brief Returns the right hand side expression on which the transformer is working.
     * \return A reference to the right hand side expression on which the transformer is working.
     */
    std::add_lvalue_reference_t<RightExpr> rhs() {
        return _rhs;
    }

    /*!
     * \brief Returns the right hand side expression on which the transformer is working.
     * \return A reference to the right hand side expression on which the transformer is working.
     */
    cpp::add_const_lvalue_t<RightExpr> rhs() const {
        return _rhs;
    }

    /*!
     * \brief Test if this expression aliases with the given expression
     * \param rhs The other expression to test
     * \return true if the two expressions aliases, false otherwise
     */
    template <typename E>
    bool alias(const E& rhs) const noexcept {
        return _lhs.alias(rhs) || _rhs.alias(rhs);
    }

    //Apply the expression

    /*!
     * \brief Returns the element at the given index
     * \param i The index
     * \return a reference to the element at the given index.
     */
    value_type operator[](std::size_t i) const {
        return BinaryOp::apply(lhs()[i], rhs()[i]);
    }

    /*!
     * \brief Returns the value at the given index
     * This function never alters the state of the container.
     * \param i The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t i) const {
        return BinaryOp::apply(lhs().read_flat(i), rhs().read_flat(i));
    }

    /*!
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam V The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>) load(std::size_t i) const {
        return BinaryOp::template load<V>(lhs().template load<V>(i), rhs().template load<V>(i));
    }

    /*!
     * \brief Perform several operations at once.
     * \param i The index at which to perform the operation
     * \tparam V The vectorization mode to use
     * \return a vector containing several results of the expression
     */
    template <typename V = default_vec>
    ETL_STRONG_INLINE(vec_type<V>) loadu(std::size_t i) const {
        return BinaryOp::template load<V>(lhs().template loadu<V>(i), rhs().template loadu<V>(i));
    }

    /*!
     * \brief Returns the value at the given position (args...)
     * \param args The position indices
     * \return The value at the given position (args...)
     */
    template <typename... S, cpp_enable_if(sizeof...(S) == sub_size_compare<this_type>::value)>
    value_type operator()(S... args) const {
        static_assert(cpp::all_convertible_to<std::size_t, S...>::value, "Invalid size types");

        return BinaryOp::apply(lhs()(args...), rhs()(args...));
    }

    /*!
     * \brief Creates a sub view of the expression, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the expression at position i.
     */
    template <bool B = (sub_size_compare<this_type>::value > 1), cpp_enable_if(B)>
    auto operator()(std::size_t i) {
        return sub(*this, i);
    }

    /*!
     * \brief Creates a sub view of the expression, effectively removing the first dimension and fixing it to the given index.
     * \param i The index to use
     * \return a sub view of the expression at position i.
     */
    template <bool B = (sub_size_compare<this_type>::value > 1), cpp_enable_if(B)>
    auto operator()(std::size_t i) const {
        return sub(*this, i);
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
     * \brief Return an iterator to the first element of the matrix
     * \return an const iterator pointing to the first element of the matrix
     */
    iterator<const this_type> begin() const noexcept {
        return {*this, 0};
    }

    /*!
     * \brief Return an iterator to the past-the-end element of the matrix
     * \return a const iterator pointing to the past-the-end element of the matrix
     */
    iterator<const this_type> end() const noexcept {
        return {*this, size(*this)};
    }
};

/*!
 * \brief Specialization for binary_expr.
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
struct etl_traits<etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>> {
    using expr_t       = etl::binary_expr<T, LeftExpr, BinaryOp, RightExpr>; ///< The type of the expression
    using left_expr_t  = std::decay_t<LeftExpr>;                             ///< The type of the left expression
    using right_expr_t = std::decay_t<RightExpr>;                            ///< The type of the right expression

    static constexpr const bool left_directed = cpp::not_u<etl_traits<left_expr_t>::is_generator>::value; ///< True if directed by the left expression, false otherwise

    using sub_expr_t = std::conditional_t<left_directed, left_expr_t, right_expr_t>; ///< The type of sub expression

    static constexpr const bool is_etl                  = true;                                                                                                                     ///< Indicates if the type is an ETL expression
    static constexpr const bool is_transformer          = false;                                                                                                                    ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;                                                                                                                    ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                                                                                                                    ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<sub_expr_t>::is_fast;                                                                                          ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = etl_traits<left_expr_t>::is_linear && etl_traits<right_expr_t>::is_linear && BinaryOp::linear;                            ///< Indicates if the expression is linear
    static constexpr const bool is_thread_safe           = etl_traits<left_expr_t>::is_thread_safe && etl_traits<right_expr_t>::is_thread_safe && BinaryOp::thread_safe;             ///< Indicates if the expression is linear
    static constexpr const bool is_value                = false;                                                                                                                    ///< Indicates if the expression is of value type
    static constexpr const bool is_direct                = false;                                                                                                                    ///< Indicates if the expression has direct memory access
    static constexpr const bool is_generator            = etl_traits<left_expr_t>::is_generator && etl_traits<right_expr_t>::is_generator;                                          ///< Indicates if the expression is a generator expression
    static constexpr const bool needs_temporary_visitor = etl_traits<left_expr_t>::needs_temporary_visitor || etl_traits<right_expr_t>::needs_temporary_visitor;                    ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = etl_traits<left_expr_t>::needs_evaluator_visitor || etl_traits<right_expr_t>::needs_evaluator_visitor;                    ///< Indicaes if the expression needs an evaluator visitor
    static constexpr const bool is_padded               = is_linear && etl_traits<left_expr_t>::is_padded && etl_traits<right_expr_t>::is_padded;                          ///< Indicates if the expression is padded
    static constexpr const bool is_aligned               = is_linear && etl_traits<left_expr_t>::is_aligned && etl_traits<right_expr_t>::is_aligned;                          ///< Indicates if the expression is padded
    static constexpr const order storage_order          = etl_traits<left_expr_t>::is_generator ? etl_traits<right_expr_t>::storage_order : etl_traits<left_expr_t>::storage_order; ///< The expression storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = cpp::bool_constant<
        etl_traits<left_expr_t>::template vectorizable<V>::value && etl_traits<right_expr_t>::template vectorizable<V>::value && BinaryOp::template vectorizable<V>::value>;

    /*!
     * \brief Get reference to the main sub expression
     * \param v The binary expr
     * \return a refernece to the main sub expression
     */
    template <bool B = left_directed, cpp_enable_if(B)>
    static constexpr auto& get(const expr_t& v) {
        return v.lhs();
    }

    /*!
     * \brief Get reference to the main sub expression
     * \param v The binary expr
     * \return a refernece to the main sub expression
     */
    template <bool B = left_directed, cpp_disable_if(B)>
    static constexpr auto& get(const expr_t& v) {
        return v.rhs();
    }

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return etl_traits<sub_expr_t>::size(get(v));
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<sub_expr_t>::dim(get(v), d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return etl_traits<sub_expr_t>::size();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam D The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t D>
    static constexpr std::size_t dim() {
        return etl_traits<sub_expr_t>::template dim<D>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return etl_traits<sub_expr_t>::dimensions();
    }
};

/*!
 * \brief Prints the type of the binary expression to the stream
 * \param os The output stream
 * \param expr The expression to print
 * \return the output stream
 */
template <typename T, typename LeftExpr, typename BinaryOp, typename RightExpr>
std::ostream& operator<<(std::ostream& os, const binary_expr<T, LeftExpr, BinaryOp, RightExpr>& expr) {
    if (BinaryOp::desc_func) {
        return os << BinaryOp::desc() << "(" << expr.lhs() << ", " << expr.rhs() << ")";
    } else {
        return os << "(" << expr.lhs() << ' ' << BinaryOp::desc() << ' ' << expr.rhs() << ")";
    }
}

} //end of namespace etl
