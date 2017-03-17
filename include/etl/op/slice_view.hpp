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
template <typename T>
struct slice_view {
    T sub;                   ///< The Sub expression
    const std::size_t first; ///< The index
    const std::size_t last;  ///< The last index

    using sub_type          = T;                                               ///< The sub type
    using value_type        = value_t<sub_type>;                               ///< The value contained in the expression
    using memory_type       = memory_t<sub_type>;                              ///< The memory acess type
    using const_memory_type = const_memory_t<sub_type>;                        ///< The const memory access type
    using return_type       = return_helper<sub_type, decltype(sub[0])>;       ///< The type returned by the view
    using const_return_type = const_return_helper<sub_type, decltype(sub[0])>; ///< The const type return by the view

    /*!
     * \brief Construct a new slice_view over the given sub expression
     * \param sub The sub expression
     * \param first The first index
     * \param last The last index
     */
    slice_view(sub_type sub, std::size_t first, std::size_t last)
            : sub(sub), first(first), last(last) {}

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    const_return_type operator[](std::size_t j) const {
        if(decay_traits<sub_type>::storage_order == order::RowMajor){
            return sub[first * (size(sub) / dim<0>(sub)) + j];
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (size(sub) / sa);
            return sub[(j % ss) * sa + j / ss + first];
        }
    }

    /*!
     * \brief Returns the element at the given index
     * \param j The index
     * \return a reference to the element at the given index.
     */
    return_type operator[](std::size_t j) {
        if(decay_traits<sub_type>::storage_order == order::RowMajor){
            return sub[first * (size(sub) / dim<0>(sub)) + j];
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (size(sub) / sa);
            return sub[(j % ss) * sa + j / ss + first];
        }
    }

    /*!
     * \brief Returns the value at the given index
     * This function never has side effects.
     * \param j The index
     * \return the value at the given index.
     */
    value_type read_flat(std::size_t j) const noexcept {
        if(decay_traits<sub_type>::storage_order == order::RowMajor){
            return sub.read_flat(first * (size(sub) / dim<0>(sub)) + j);
        } else {
            const auto sa = dim<0>(sub);
            const auto ss = (size(sub) / sa);
            return sub.read_flat((j % ss) * sa + j / ss + first);
        }
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    const_return_type operator()(std::size_t i, S... args) const {
        return sub(i + first, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Access to the element at the given (args...) position
     * \param args The indices
     * \return a reference to the element at the given position.
     */
    template <typename... S>
    return_type operator()(std::size_t i, S... args) {
        return sub(i + first, static_cast<std::size_t>(args)...);
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto load(std::size_t x) const noexcept {
        return sub.template loadu<V>(x + first * (etl::size(sub) / etl::dim<0>(sub)));
    }

    /*!
     * \brief Load several elements of the expression at once
     * \param x The position at which to start. This will be aligned from the beginning (multiple of the vector size).
     * \tparam V The vectorization mode to use
     * \return a vector containing several elements of the expression
     */
    template <typename V = default_vec>
    auto loadu(std::size_t x) const noexcept {
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

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    memory_type memory_start() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + first * (etl::size(sub) / etl::dim<0>(sub));
    }

    /*!
     * \brief Returns a pointer to the first element in memory.
     * \return a pointer tot the first element in memory.
     */
    const_memory_type memory_start() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + first * (etl::size(sub) / etl::dim<0>(sub));
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    memory_type memory_end() noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + last * (etl::size(sub) / etl::dim<0>(sub));
    }

    /*!
     * \brief Returns a pointer to the past-the-end element in memory.
     * \return a pointer tot the past-the-end element in memory.
     */
    const_memory_type memory_end() const noexcept {
        static_assert(has_direct_access<T>::value && decay_traits<sub_type>::storage_order == order::RowMajor, "This expression does not have direct memory access");
        return sub.memory_start() + last * (etl::size(sub) / etl::dim<0>(sub));
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
    void visit(const detail::back_propagate_visitor& visitor) const {
        sub.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(const detail::temporary_allocator_visitor& visitor) const {
        sub.visit(visitor);
    }

    /*!
     * \brief Apply the given visitor to this expression and its descendants.
     * \param visitor The visitor to apply
     */
    void visit(detail::evaluator_visitor& visitor) const {
        bool old_need_value = visitor.need_value;
        visitor.need_value = true;
        sub.visit(visitor);
        visitor.need_value = old_need_value;
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

    static constexpr bool is_etl                  = true;                                            ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer          = false;                                           ///< Indicates if the type is a transformer
    static constexpr bool is_view                 = true;                                            ///< Indicates if the type is a view
    static constexpr bool is_magic_view           = false;                                           ///< Indicates if the type is a magic view
    static constexpr bool is_fast                 = false;                                           ///< Indicates if the expression is fast
    static constexpr bool is_linear               = sub_traits::is_linear;               ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe          = sub_traits::is_thread_safe;          ///< Indicates if the expression is thread safe
    static constexpr bool is_value                = false;                                           ///< Indicates if the expression is of value type
    static constexpr bool is_direct               = sub_traits::is_direct && sub_traits::storage_order == order::RowMajor;               ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator            = false;                                           ///< Indicates if the expression is a generator
    static constexpr bool is_padded               = false;                          ///< Indicates if the expression is padded
    static constexpr bool is_aligned               = false;                          ///< Indicates if the expression is padded
    static constexpr bool needs_evaluator_visitor = sub_traits::needs_evaluator_visitor; ///< Indicates if the exxpression needs a evaluator visitor
    static constexpr order storage_order          = sub_traits::storage_order;           ///< The expression's storage order

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
    static std::size_t size(const expr_t& v) {
        return (sub_traits::size(v.sub) / sub_traits::dim(v.sub, 0)) * (v.last - v.first);
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
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
    static constexpr std::size_t dimensions() {
        return sub_traits::dimensions();
    }
};

} //end of namespace etl
