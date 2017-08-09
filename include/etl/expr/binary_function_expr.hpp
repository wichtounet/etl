//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

namespace etl {

/*!
 * \brief An unary function expression.
 * \tparam A The unary sub type
 */
template <typename A, typename B, typename Impl>
struct binary_function_expr : base_temporary_expr_bin<binary_function_expr<A, B, Impl>, A, B> {
    using value_type  = value_t<A>;                               ///< The type of value of the expression
    using this_type   = binary_function_expr<A, B, Impl>;         ///< The type of this expression
    using base_type   = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using left_traits = decay_traits<A>;                          ///< The traits of the sub type

    static constexpr auto storage_order = left_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit binary_function_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename C, cpp_enable_iff(all_fast<A, B, C>)>
    static void check(const A& a, const B& b, const C& c){
        static constexpr etl::order order_a = decay_traits<A>::storage_order;
        static constexpr etl::order order_b = decay_traits<B>::storage_order;
        static constexpr etl::order order_c = decay_traits<C>::storage_order;

        static_assert(order_a == order_c, "Cannot change storage order");
        static_assert(order_b == order_c, "Cannot change storage order");

        static_assert(decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "Invalid dimensions");
        static_assert(decay_traits<B>::dimensions() == decay_traits<C>::dimensions(), "Invalid dimensions");

        static_assert(decay_traits<A>::size() == decay_traits<C>::size(), "Invalid size");
        static_assert(decay_traits<B>::size() == decay_traits<C>::size(), "Invalid size");

        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename C, cpp_disable_iff(all_fast<A, B, C>)>
    static void check(const A& a, const B& b, const C& c){
        static constexpr etl::order order_a = decay_traits<A>::storage_order;
        static constexpr etl::order order_b = decay_traits<B>::storage_order;
        static constexpr etl::order order_c = decay_traits<C>::storage_order;

        static_assert(order_a == order_c, "Cannot change storage order");
        static_assert(order_b == order_c, "Cannot change storage order");

        static_assert(decay_traits<A>::dimensions() == decay_traits<C>::dimensions(), "Invalid dimensions");
        static_assert(decay_traits<B>::dimensions() == decay_traits<C>::dimensions(), "Invalid dimensions");

        cpp_assert(etl::size(a) == etl::size(c), "Invalid size");
        cpp_assert(etl::size(b) == etl::size(c), "Invalid size");

        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);
    }

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template<typename C>
    void assign_to(C&& c)  const {
        static_assert(all_etl_expr<A, B, C>, "binary_function only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, c);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        Impl::apply(make_temporary(a), make_temporary(b), c);
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
    friend std::ostream& operator<<(std::ostream& os, const binary_function_expr& expr) {
        return os << Impl::name() << "(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for an unary function expression
 * \tparam A The unary sub type
 */
template <typename A, typename B, typename Impl>
struct etl_traits<etl::binary_function_expr<A, B, Impl>> {
    using expr_t       = etl::binary_function_expr<A, B, Impl>; ///< The expression type
    using left_expr_t  = std::decay_t<A>;                       ///< The left sub expression type
    using right_expr_t = std::decay_t<B>;                       ///< The right sub expression type
    using left_traits  = etl_traits<left_expr_t>;               ///< The left sub traits
    using right_traits = etl_traits<right_expr_t>;              ///< The right sub traits
    using value_type   = value_t<A>;                            ///< The value type of the expression

    static constexpr bool is_etl         = true;                       ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                      ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                      ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                      ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = all_fast<A, B>;      ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                      ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;                       ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                      ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = true;                       ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                      ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                      ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = true;                       ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = true;                       ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order = left_traits::storage_order; ///< The expression's storage order

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
        return etl::dim<DD, A>();
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
