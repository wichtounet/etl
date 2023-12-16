//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief dim_view expression implementation
 */

#pragma once

namespace etl {

/*!
 * \brief View that shows one dimension of a matrix
 * \tparam T The type of expression on which the view is made
 * \tparam D The dimension to show
 */
template <typename T, size_t D>
struct dim_view {
    static_assert(D == 1 || D == 2, "Invalid dimension");

    using sub_type          = T;                                                                       ///< The sub type
    using value_type        = value_t<sub_type>;                                                       ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                                                      ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                                                ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(std::declval<sub_type>()(0, 0))>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(std::declval<sub_type>()(0, 0))>; ///< The const type return by the view

private:
    T sub;          ///< The Sub expression
    const size_t i; ///< The index

    friend struct etl_traits<dim_view>;

public:
    /*!
     * \brief Construct a new dim_view over the given sub expression
     * \param sub The sub expression
     * \param i The sub index
     */
    dim_view(sub_type sub, size_t i) : sub(sub), i(i) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](size_t j) const {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](size_t j) {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(size_t j) const noexcept {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator()(size_t j) const {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator()(size_t j) {
        if (D == 1) {
            return sub(i, j);
        } else { //D == 2
            return sub(j, i);
        }
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
    memory_type memory_start() noexcept requires(etl_dma<T> && D == 1) {
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept requires(etl_dma<T> && D == 1){
        return sub.memory_start() + i * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept requires(etl_dma<T> && D == 1){
        return sub.memory_start() + (i + 1) * subsize(sub);
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept requires(etl_dma<T> && D == 1){
        return sub.memory_start() + (i + 1) * subsize(sub);
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
    friend std::ostream& operator<<(std::ostream& os, const dim_view& v) {
        return os << "dim[" << D << "](" << v.sub << ", " << v.i << ")";
    }
};

/*!
 * \brief Specialization for dim_view
 */
template <typename T, size_t D>
struct etl_traits<etl::dim_view<T, D>> {
    using expr_t     = etl::dim_view<T, D>;                         ///< The expression type
    using sub_expr_t = std::decay_t<T>;                             ///< The sub expression type
    using value_type = typename etl_traits<sub_expr_t>::value_type; ///< The value type

    static constexpr bool is_etl         = true;                                        ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                       ///< Indicates if the type is a transformer
    static constexpr bool is_view        = true;                                        ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                       ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = etl_traits<sub_expr_t>::is_fast;             ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                                       ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = etl_traits<sub_expr_t>::is_thread_safe;      ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                                       ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = etl_traits<sub_expr_t>::is_direct && D == 1; ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                                       ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                                       ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = false;                                       ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = etl_traits<sub_expr_t>::is_temporary;        ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr bool gpu_computable = false;                                       ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = etl_traits<sub_expr_t>::storage_order;       ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static size_t size(const expr_t& v) {
        if (D == 1) {
            return etl_traits<sub_expr_t>::dim(v.sub, 1);
        } else {
            return etl_traits<sub_expr_t>::dim(v.sub, 0);
        }
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static size_t dim(const expr_t& v, [[maybe_unused]] size_t d) {
        cpp_assert(d == 0, "Invalid dimension");

        return size(v);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr size_t size() {
        return D == 1 ? etl_traits<sub_expr_t>::template dim<1>() : etl_traits<sub_expr_t>::template dim<0>();
    }

    /*!
     * \brief Returns the D2th dimension of an expression of this type
     * \tparam D2 The dimension to get
     * \return the D2th dimension of an expression of this type
     */
    template <size_t D2>
    static constexpr size_t dim() requires(D2 == 0) {
        return size();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr size_t dimensions() {
        return 1;
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
