//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/conv.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, typename B, bool Flipped>
struct conv_2d_full_multi_expr : base_temporary_expr_bin<conv_2d_full_multi_expr<A, B, Flipped>, A, B> {
    using value_type  = value_t<A>;                               ///< The type of value of the expression
    using this_type   = conv_2d_full_multi_expr<A, B, Flipped>;         ///< The type of this expression
    using base_type   = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using left_traits = decay_traits<A>;                          ///< The traits of the sub type

    static constexpr auto storage_order = left_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit conv_2d_full_multi_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C, cpp_disable_if(all_fast<A, B, C>::value)>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_full_multi");
        static_assert(etl::dimensions<K>() == 3, "Invalid number of dimensions for kernel of conv2_full_multi");
        static_assert(etl::dimensions<C>() == 3, "Invalid number of dimensions for conv of conv2_full_multi");

        cpp_assert(etl::dim(conv, 0) == etl::dim(kernel, 0), "Invalid dimensions for conv2_full_multi");
        cpp_assert(etl::dim(conv, 1) == etl::dim(input, 1) + etl::dim(kernel, 1) - 1, "Invalid dimensions for conv2_full_multi");
        cpp_assert(etl::dim(conv, 2) == etl::dim(input, 2) + etl::dim(kernel, 2) - 1, "Invalid dimensions for conv2_full_multi");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C, cpp_enable_if(all_fast<A, B, C>::value)>
    static void check(const I& input, const K& kernel, const C& conv){
        static_assert(etl::dimensions<I>() == 2, "Invalid number of dimensions for input of conv2_full_multi");
        static_assert(etl::dimensions<K>() == 3, "Invalid number of dimensions for kernel of conv2_full_multi");
        static_assert(etl::dimensions<C>() == 3, "Invalid number of dimensions for conv of conv2_full_multi");

        static_assert(etl::dim<0, C>() == etl::dim<0, K>(), "Invalid dimensions for conv2_full_multi");
        static_assert(etl::dim<1, C>() == etl::dim<0, I>() + etl::dim<1, K>() - 1, "Invalid dimensions for conv2_full_multi");
        static_assert(etl::dim<2, C>() == etl::dim<1, I>() + etl::dim<2, K>() - 1, "Invalid dimensions for conv2_full_multi");

        cpp_unused(input);
        cpp_unused(kernel);
        cpp_unused(conv);
    }

    /*!
     * \brief Assign to a matrix of the full storage order
     * \param c The expression to which assign
     */
    template<typename C>
    void assign_to(C&& c)  const {
        static_assert(all_etl_expr<A, B, C>::value, "conv2_full_multi only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, c);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);
        standard_evaluator::pre_assign_lhs(c);

        if /* constexpr */ (Flipped){
            detail::conv2_full_multi_flipped_impl::apply(make_temporary(a), make_temporary(b), c);
        } else {
            detail::conv2_full_multi_impl::apply(make_temporary(a), make_temporary(b), c);
        }
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
    friend std::ostream& operator<<(std::ostream& os, const conv_2d_full_multi_expr& expr) {
        return os << "conv2_full_multi(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B, bool Flipped>
struct etl_traits<etl::conv_2d_full_multi_expr<A, B, Flipped>> {
    using expr_t       = etl::conv_2d_full_multi_expr<A, B, Flipped>; ///< The expression type
    using left_expr_t  = std::decay_t<A>;                       ///< The left sub expression type
    using right_expr_t = std::decay_t<B>;                       ///< The right sub expression type
    using left_traits  = etl_traits<left_expr_t>;               ///< The left sub traits
    using right_traits = etl_traits<right_expr_t>;              ///< The right sub traits
    using value_type   = value_t<A>;                            ///< The value type of the expression

    static constexpr bool is_etl          = true;                       ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer  = false;                      ///< Indicates if the type is a transformer
    static constexpr bool is_view         = false;                      ///< Indicates if the type is a view
    static constexpr bool is_magic_view   = false;                      ///< Indicates if the type is a magic view
    static constexpr bool is_fast         = all_fast<A, B>::value;      ///< Indicates if the expression is fast
    static constexpr bool is_linear       = true;                       ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe  = true;                       ///< Indicates if the expression is thread safe
    static constexpr bool is_value        = false;                      ///< Indicates if the expression is of value type
    static constexpr bool is_direct       = true;                       ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator    = false;                      ///< Indicates if the expression is a generator
    static constexpr bool is_padded       = false;                      ///< Indicates if the expression is padded
    static constexpr bool is_aligned      = true;                       ///< Indicates if the expression is padded
    static constexpr bool is_gpu          = false;                      ///< Indicates if the expression can be done on GPU
    static constexpr bool needs_evaluator = true;                       ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order  = left_traits::storage_order; ///< The expression's storage order

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
        return DD == 0 ? etl::dim<0, B>()
             : DD == 1 ? etl::dim<0, A>() + etl::dim<1, B>() - 1
                       : etl::dim<1, A>() + etl::dim<2, B>() - 1;
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        if(d == 0){
            return etl::dim(e._b, 0);
        } else if(d == 1){
            return etl::dim(e._a, 0) + etl::dim(e._b, 1) - 1;
        } else if(d == 2){
            return etl::dim(e._a, 1) + etl::dim(e._b, 2) - 1;
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return (etl::dim(e._b, 0))
            * (etl::dim(e._a, 0) + etl::dim(e._b, 1) - 1)
            * (etl::dim(e._a, 1) + etl::dim(e._b, 2) - 1);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return (etl::dim<0, B>())
             * (etl::dim<0, A>() + etl::dim<1, B>() - 1)
             * (etl::dim<1, A>() + etl::dim<2, B>() - 1);
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 3;
    }
};

/*!
 * \brief Creates an expression representing the 'full' 1D convolution of a and b.
 *
 * The convolution is applied with padding so that the output has
 * the full size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the 'full' 1D convolution of a and b
 */
template <typename A, typename B>
conv_2d_full_multi_expr<A, B, false> conv_2d_full_multi(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    return conv_2d_full_multi_expr<A, B, false>{a, b};
}

/*!
 * \brief Creates an expression representing the 'full' 1D convolution of a and b, the result will be stored in c
 *
 * The convolution is applied with padding so that the output has
 * the full size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 *
 * \return an expression representing the 'full' 1D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_2d_full_multi(A&& a, B&& b, C&& c){
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_full_multi(a, b);

    return c;
}

/*!
 * \brief Creates an expression representing the 'full' 1D convolution of a and flipped b.
 *
 * The convolution is applied with padding so that the output has
 * the full size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the 'full' 1D convolution of a and b
 */
template <typename A, typename B>
conv_2d_full_multi_expr<A, B, true> conv_2d_full_multi_flipped(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>::value, "Convolution only supported for ETL expressions");

    return conv_2d_full_multi_expr<A, B, true>{a, b};
}

/*!
 * \brief Creates an expression representing the 'full' 1D convolution of a and flipped b, the result will be stored in c
 *
 * The convolution is applied with padding so that the output has
 * the full size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 *
 * \return an expression representing the 'full' 1D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_2d_full_multi_flipped(A&& a, B&& b, C&& c){
    static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

    c = conv_2d_full_multi_flipped(a, b);

    return c;
}

} //end of namespace etl
