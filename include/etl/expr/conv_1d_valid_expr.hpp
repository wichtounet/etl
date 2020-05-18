//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/conv.hpp"
#include "etl/impl/conv_select.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, typename B>
struct conv_1d_valid_expr : base_temporary_expr_bin<conv_1d_valid_expr<A, B>, A, B> {
    using value_type  = value_t<A>;                               ///< The type of value of the expression
    using this_type   = conv_1d_valid_expr<A, B>;                 ///< The type of this expression
    using base_type   = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using left_traits = decay_traits<A>;                          ///< The traits of the sub type

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
    explicit conv_1d_valid_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <typename I, typename K, typename C>
    static void check([[maybe_unused]] const I& input, [[maybe_unused]] const K& kernel, [[maybe_unused]] const C& conv) {
        static_assert(etl::dimensions<I>() == 1, "Invalid number of dimensions for input of conv1_valid");
        static_assert(etl::dimensions<K>() == 1, "Invalid number of dimensions for kernel of conv1_valid");
        static_assert(etl::dimensions<C>() == 1, "Invalid number of dimensions for conv of conv1_valid");

        if constexpr (all_fast<A, B, C>) {
            static_assert(etl::dim<0, C>() == etl::dim<0, I>() - etl::dim<0, K>() + 1, "Invalid dimensions for conv1_valid");
            static_assert(etl::dim<0, I>() >= etl::dim<0, K>(), "Invalid dimensions for conv1_valid");
        } else {
            cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0) - etl::dim(kernel, 0) + 1, "Invalid dimensions for conv1_valid");
            cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv1_valid");
        }
    }

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param conv The expression to which assign
     */
    template <typename C>
    void assign_to(C&& conv) const {
        static_assert(all_etl_expr<A, B, C>, "conv1_valid only supported for ETL expressions");

        auto& input_raw  = this->a();
        auto& kernel_raw = this->b();

        check(input_raw, kernel_raw, conv);

        // Make temporaries if necessary

        // Execute the correct implementation

        constexpr_select const auto impl = detail::select_conv1_impl_new<conv_type::VALID, A, B, C>();

//CPP17: if constexpr
#ifdef ETL_PARALLEL_SUPPORT
        decltype(auto) input  = make_temporary(input_raw);
        decltype(auto) kernel = make_temporary(kernel_raw);

        bool parallel_dispatch = detail::select_parallel(input, kernel, conv);

        if
            constexpr_select(impl == etl::conv_impl::VEC) {
                engine_dispatch_1d([&](size_t first, size_t last) { impl::vec::conv1_valid(input, kernel, conv, first, last); }, 0, etl::size(conv),
                                   parallel_dispatch);
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                engine_dispatch_1d([&](size_t first, size_t last) { impl::standard::conv1_valid(input, kernel, conv, first, last); }, 0, etl::size(conv),
                                   parallel_dispatch);
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
#else
        if
            constexpr_select(impl == etl::conv_impl::VEC) {
                impl::vec::conv1_valid(smart_forward(input_raw), smart_forward(kernel_raw), conv, 0, etl::size(conv));
            }
        else if
            constexpr_select(impl == etl::conv_impl::STD) {
                impl::standard::conv1_valid(smart_forward(input_raw), smart_forward(kernel_raw), conv, 0, etl::size(conv));
            }
        else {
            cpp_unreachable("Invalid conv implementation selection");
        }
#endif
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

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const conv_1d_valid_expr& expr) {
        return os << "conv1_valid(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B>
struct etl_traits<etl::conv_1d_valid_expr<A, B>> {
    using expr_t       = etl::conv_1d_valid_expr<A, B>; ///< The expression type
    using left_expr_t  = std::decay_t<A>;               ///< The left sub expression type
    using right_expr_t = std::decay_t<B>;               ///< The right sub expression type
    using left_traits  = etl_traits<left_expr_t>;       ///< The left sub traits
    using right_traits = etl_traits<right_expr_t>;      ///< The right sub traits
    using value_type   = value_t<A>;                    ///< The value type of the expression

    static constexpr bool is_etl         = true;                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = all_fast<A, B>;                       ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                                ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;                                 ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                                ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = true;                                 ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                                ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                                ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = true;                                 ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = true;                                 ///< Indicates if the expression needs a evaluator visitor
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled; ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = left_traits::storage_order;           ///< The expression's storage order

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
        return etl::dim<0, A>() - etl::dim<0, B>() + 1;
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, [[maybe_unused]] size_t d) {
        return etl::dim(e._a, 0) - etl::dim(e._b, 0) + 1;
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::dim(e._a, 0) - etl::dim(e._b, 0) + 1;
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return etl::dim<0, A>() - etl::dim<0, B>() + 1;
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 1;
    }
};

/*!
 * \brief Creates an expression representing the valid 1D convolution of a and b
 * \param a The input expression
 * \param b The kernel expression
 * \return an expression representing the valid 1D convolution of a and b
 */
template <typename A, typename B>
conv_1d_valid_expr<detail::build_type<A>, detail::build_type<B>> conv_1d_valid(A&& a, B&& b) {
    static_assert(all_etl_expr<A, B>, "Convolution only supported for ETL expressions");

    return conv_1d_valid_expr<detail::build_type<A>, detail::build_type<B>>{a, b};
}

/*!
 * \brief Creates an expression representing the valid 1D convolution of a and b, the result will be stored in c
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 * \return an expression representing the valid 1D convolution of a and b
 */
template <typename A, typename B, typename C>
auto conv_1d_valid(A&& a, B&& b, C&& c) {
    static_assert(all_etl_expr<A, B, C>, "Convolution only supported for ETL expressions");

    c = conv_1d_valid(a, b);

    return c;
}

} //end of namespace etl
