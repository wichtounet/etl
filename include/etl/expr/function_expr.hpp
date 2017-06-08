//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

#include "etl/impl/functions.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, typename Impl>
struct function_expr : base_temporary_expr_un<function_expr<A, Impl>, A> {
    using value_type = value_t<A>;                           ///< The type of value of the expression
    using this_type  = function_expr<A, Impl>;               ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>; ///< The base type
    using sub_traits = decay_traits<A>;                      ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit function_expr(A a) : base_type(a) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C, cpp_enable_if(all_fast<A,C>::value)>
    static void check(const A& a, const C& c) {
        cpp_unused(a);
        cpp_unused(c);

        static constexpr etl::order order_lhs = decay_traits<C>::storage_order;
        static constexpr etl::order order_rhs = decay_traits<A>::storage_order;

        static_assert(order_lhs == order_rhs, "Cannot change storage order");
        static_assert(decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "Invalid dimensions");
        static_assert(decay_traits<A>::size() == decay_traits<C>::size(), "Invalid size");
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C, cpp_disable_if(all_fast<A,C>::value)>
    static void check(const A& a, const C& c) {
        static constexpr etl::order order_lhs = decay_traits<A>::storage_order;
        static constexpr etl::order order_rhs = decay_traits<A>::storage_order;

        static_assert(order_lhs == order_rhs, "Cannot change storage order");
        static_assert(decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "Invalid dimensions");
        cpp_assert(etl::size(a) == etl::size(c), "Invalid size");
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template<typename C, cpp_enable_if(decay_traits<C>::storage_order == storage_order)>
    void assign_to(C&& c)  const {
        static_assert(all_etl_expr<A, C>::value, "Function expression only supported for ETL expressions");

        auto& a = this->a();

        standard_evaluator::pre_assign_rhs(a);

        check(a, c);

        Impl::apply(make_temporary(a), c);
    }

    /*!
     * \brief Assign to a matrix of a different storage order
     * \param c The expression to which assign
     */
    template<typename C, cpp_enable_if(decay_traits<C>::storage_order != storage_order)>
    void assign_to(C&& c)  const {
        std_assign_evaluate(*this, c);
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
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const function_expr& expr) {
        return os << Impl::name() << "(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename Impl>
struct etl_traits<etl::function_expr<A, Impl>> {
    using expr_t     = etl::function_expr<A, Impl>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;             ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;      ///< The sub traits
    using value_type = value_t<A>;                  ///< The value type of the expression

    static constexpr bool is_etl         = true;                      ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                     ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                     ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                     ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast;       ///< Indicates if the expression is fast
    static constexpr bool is_linear      = true;                      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;                      ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                     ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = true;                      ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                     ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                     ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = true;                      ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = true;                      ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order = sub_traits::storage_order; ///< The expression's storage order

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;

    /*!
     * \brief Returns the DDth dimension of the expression
     * \return the DDth dimension of the expression
     */
    template <size_t DD>
    static constexpr size_t dim() {
        return decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        return etl::dim(e._a, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::size(e._a);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return decay_traits<A>::size();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return decay_traits<A>::dimensions();
    }
};

} //end of namespace etl
