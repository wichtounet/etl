//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/conv_deep.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, typename B, size_t S1, size_t S2, size_t P1, size_t P2, bool Flipped>
struct conv_2d_valid_deep_expr : base_temporary_expr_bin<conv_2d_valid_deep_expr<A, B, S1, S2, P1, P2, Flipped>, A, B> {
    using value_type  = value_t<A>;                               ///< The type of value of the expression
    using this_type   = conv_2d_valid_deep_expr<A, B, S1, S2, P1, P2, Flipped>;    ///< The type of this expression
    using base_type   = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using left_traits = decay_traits<A>;                          ///< The traits of the sub type

    static constexpr size_t D           = left_traits::dimensions();  ///< The dimensions of the expresions
    static constexpr auto storage_order = left_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit conv_2d_valid_deep_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check([[maybe_unused]] const I& input, [[maybe_unused]] const K& kernel, [[maybe_unused]] const C& conv){
        static_assert(etl::dimensions<I>() == D, "Invalid number of dimensions for input of conv2_valid_deep");
        static_assert(etl::dimensions<K>() == D, "Invalid number of dimensions for kernel of conv2_valid_deep");
        static_assert(etl::dimensions<C>() == D, "Invalid number of dimensions for conv of conv2_valid_deep");

        if constexpr (all_fast<A, B, C>) {
            static_assert(etl::dim<D - 2, C>() == (etl::dim<D - 2, I>() + etl::dim<D - 2, K>() + 2 * P1) / S1 - 1, "Invalid dimensions for conv2_valid_deep");
            static_assert(etl::dim<D - 1, C>() == (etl::dim<D - 1, I>() + etl::dim<D - 1, K>() + 2 * P2) / S2 - 1, "Invalid dimensions for conv2_valid_deep");
        } else {
            cpp_assert(etl::dim(conv, D - 2) == (etl::dim(input, D - 2) + etl::dim(kernel, D - 2) + 2 * P1) / S1 - 1, "Invalid dimensions for conv2_valid_deep");
            cpp_assert(etl::dim(conv, D - 1) == (etl::dim(input, D - 1) + etl::dim(kernel, D - 1) + 2 * P2) / S2 - 1, "Invalid dimensions for conv2_valid_deep");
        }
    }

    /*!
     * \brief Assign to a matrix of the valid storage order
     * \param c The expression to which assign
     */
    template<typename C>
    void assign_to(C&& c)  const {
        static_assert(all_etl_expr<A, B, C>, "conv2_valid_deep only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, c);

        if constexpr (Flipped){
            detail::conv2_valid_flipped_deep_impl<S1, S2, P1, P2>::apply(smart_forward(a), smart_forward(b), c);
        } else {
            detail::conv2_valid_deep_impl<S1, S2, P1, P2>::apply(smart_forward(a), smart_forward(b), c);
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
    friend std::ostream& operator<<(std::ostream& os, const conv_2d_valid_deep_expr& expr) {
        return os << "conv2_valid_deep(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B, size_t S1, size_t S2, size_t P1, size_t P2, bool Flipped>
struct etl_traits<etl::conv_2d_valid_deep_expr<A, B, S1, S2, P1, P2, Flipped>> {
    using expr_t       = etl::conv_2d_valid_deep_expr<A, B, S1, S2, P1, P2, Flipped>; ///< The expression type
    using this_type    = etl_traits<expr_t>;                                         ///< The type of this traits
    using left_expr_t  = std::decay_t<A>;                                            ///< The left sub expression type
    using right_expr_t = std::decay_t<B>;                                            ///< The right sub expression type
    using left_traits  = etl_traits<left_expr_t>;                                    ///< The left sub traits
    using right_traits = etl_traits<right_expr_t>;                                   ///< The right sub traits
    using value_type   = value_t<A>;                                                 ///< The value type of the expression

    static constexpr bool is_etl          = true;                       ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer  = false;                      ///< Indicates if the type is a transformer
    static constexpr bool is_view         = false;                      ///< Indicates if the type is a view
    static constexpr bool is_magic_view   = false;                      ///< Indicates if the type is a magic view
    static constexpr bool is_fast         = all_fast<A, B>;      ///< Indicates if the expression is fast
    static constexpr bool is_linear       = false;                       ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe  = true;                       ///< Indicates if the expression is thread safe
    static constexpr bool is_value        = false;                      ///< Indicates if the expression is of value type
    static constexpr bool is_direct       = true;                       ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator    = false;                      ///< Indicates if the expression is a generator
    static constexpr bool is_padded       = false;                      ///< Indicates if the expression is padded
    static constexpr bool is_aligned      = true;                       ///< Indicates if the expression is padded
    static constexpr bool is_temporary = true;                       ///< Indicates if the expression needs a evaluator visitor
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled;                                         ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order  = left_traits::storage_order; ///< The expression's storage order

    static constexpr size_t D = left_traits::dimensions(); ///< The number of dimensions

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = true;

    /*!
     * \brief Returns the DDth dimension of the expression
     * \return the DDth dimension of the expression
     */
    template <size_t DD>
    static constexpr size_t dim() {
        return DD == D - 2 ? (etl::dim<DD, A>() + etl::dim<DD, B>() + 2 * P1) / S1 - 1
             : DD == D - 1 ? (etl::dim<DD, A>() + etl::dim<DD, B>() + 2 * P2) / S2 - 1
                           : etl::dim<DD, A>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        if(d == D - 2){
            return (etl::dim(e._a, d) + etl::dim(e._b, d) + 2 * P1) / S1 - 1;
        } else if(d == D - 1){
            return (etl::dim(e._a, d) + etl::dim(e._b, d) + 2 * P2) / S2 - 1;
        } else {
            return etl::dim(e._a, d);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        size_t s = 1;
        for(size_t d = 0; d < D; ++d){
            s *= this_type::dim(e, d);
        }
        return s;
    }

    /*!
     * \brief Returns the multiplicative sum of the dimensions at the given indices
     * \return the multiplicative sum of the dimensions at the given indices
     */
    template <size_t... I>
    static constexpr size_t size_mul(const std::index_sequence<I...>& /*seq*/) {
        return (this_type::dim<I>() * ...);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return size_mul(std::make_index_sequence<dimensions()>());
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Creates an expression representing the 'valid' 1D convolution of a and b.
 *
 * The convolution is applied with padding so that the output has
 * the valid size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the 'valid' 1D convolution of a and b
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
conv_2d_valid_deep_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, false> conv_2d_valid_deep(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return conv_2d_valid_deep_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, false>{a, b};
}

/*!
 * \brief Creates an expression representing the 'valid' 1D convolution of a and b, the result will be stored in c
 *
 * The convolution is applied with padding so that the output has
 * the valid size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 *
 * \return an expression representing the 'valid' 1D convolution of a and b
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B, typename C>
auto conv_2d_valid_deep(A&& a, B&& b, C&& c){
    static_assert(all_etl_expr<A, B, C>, "Convolution only supported for ETL expressions");

    c = conv_2d_valid_deep<S1, S2, P1, P2>(a, b);

    return c;
}

/*!
 * \brief Creates an expression representing the 'valid' 1D convolution of a and flipped b.
 *
 * The convolution is applied with padding so that the output has
 * the valid size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the 'valid' 1D convolution of a and b
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B>
conv_2d_valid_deep_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true> conv_2d_valid_deep_flipped(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return conv_2d_valid_deep_expr<detail::build_type<A>, detail::build_type<B>, S1, S2, P1, P2, true>{a, b};
}

/*!
 * \brief Creates an expression representing the 'valid' 1D convolution of a and flipped b, the result will be stored in c
 *
 * The convolution is applied with padding so that the output has
 * the valid size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 *
 * \return an expression representing the 'valid' 1D convolution of a and b
 */
template <size_t S1 = 1, size_t S2 = 1, size_t P1 = 0, size_t P2 = 0, typename A, typename B, typename C>
auto conv_2d_valid_deep_flipped(A&& a, B&& b, C&& c){
    static_assert(all_etl_expr<A, B, C>, "Convolution only supported for ETL expressions");

    c = conv_2d_valid_deep_flipped<S1, S2, P1, P2>(a, b);

    return c;
}

} //end of namespace etl
