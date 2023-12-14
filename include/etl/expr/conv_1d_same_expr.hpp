//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/conv_select.hpp"
#include "etl/impl/conv.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <etl_expr A, etl_expr B>
struct conv_1d_same_expr : base_temporary_expr_bin<conv_1d_same_expr<A, B>, A, B> {
    using value_type  = value_t<A>;                               ///< The type of value of the expression
    using this_type   = conv_1d_same_expr<A, B>;                  ///< The type of this expression
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
    explicit conv_1d_same_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assert that the convolution is done on correct dimensions
     */
    template <etl_1d I, etl_1d K, etl_1d C>
    static void check([[maybe_unused]] const I& input, [[maybe_unused]] const K& kernel, [[maybe_unused]] const C& conv) {
        if constexpr (all_fast<A, B, C>) {
            static_assert(etl::dim<0, C>() == etl::dim<0, I>(), "Invalid dimensions for conv1_same");
            static_assert(etl::dim<0, I>() >= etl::dim<0, K>(), "Invalid dimensions for conv1_same");
        } else {
            cpp_assert(etl::dim(conv, 0) == etl::dim(input, 0), "Invalid dimensions for conv1_same");
            cpp_assert(etl::dim(input, 0) >= etl::dim(kernel, 0), "Invalid dimensions for conv1_same");
        }
    }

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param conv The expression to which assign
     */
    template <etl_expr C>
    void assign_to(C&& conv) const {
        inc_counter("temp:assign");

        auto& input_raw  = this->a();
        auto& kernel_raw = this->b();

        check(input_raw, kernel_raw, conv);

        // Execute the correct implementation

        constexpr_select const auto impl = detail::select_conv1_impl_new<conv_type::SAME, A, B, C>();

        if constexpr_select (impl == etl::conv_impl::VEC) {
            inc_counter("impl:vec");

            if constexpr (parallel_support) {
                bool parallel_dispatch = detail::select_parallel(input_raw, kernel_raw, conv);

                if (parallel_dispatch) {
                    decltype(auto) input  = smart_forward(input_raw);
                    decltype(auto) kernel = smart_forward(kernel_raw);

                    engine_dispatch_1d(
                            [&](size_t first, size_t last) { impl::vec::conv1_same(input, kernel, conv, first, last); }, 0, etl::size(conv), parallel_dispatch);
                } else {
                    impl::vec::conv1_same(smart_forward(input_raw), smart_forward(kernel_raw), conv, 0, etl::size(conv));
                }
            } else {
                impl::vec::conv1_same(smart_forward(input_raw), smart_forward(kernel_raw), conv, 0, etl::size(conv));
            }
        } else if constexpr_select (impl == etl::conv_impl::STD) {
            inc_counter("impl:std");

            if constexpr (parallel_support) {
                bool parallel_dispatch = detail::select_parallel(input_raw, kernel_raw, conv);

                if (parallel_dispatch) {
                    decltype(auto) input  = smart_forward(input_raw);
                    decltype(auto) kernel = smart_forward(kernel_raw);

                    engine_dispatch_1d([&](size_t first, size_t last) { impl::standard::conv1_same(input, kernel, conv, first, last); },
                                       0,
                                       etl::size(conv),
                                       parallel_dispatch);
                } else {
                    impl::standard::conv1_same(smart_forward(input_raw), smart_forward(kernel_raw), conv, 0, etl::size(conv));
                }
            } else {
                impl::standard::conv1_same(smart_forward(input_raw), smart_forward(kernel_raw), conv, 0, etl::size(conv));
            }
        } else if constexpr_select (impl == etl::conv_impl::EGBLAS) {
            inc_counter("impl:egblas");

            if constexpr (all_homogeneous<A, B, C>) {
                decltype(auto) input  = smart_forward_gpu(input_raw);
                decltype(auto) kernel = smart_forward_gpu(kernel_raw);

                input.ensure_gpu_up_to_date();
                kernel.ensure_gpu_up_to_date();

                conv.ensure_gpu_allocated();

                impl::egblas::conv1_same(
                        etl::size(input_raw), etl::size(kernel_raw), value_type(1), input.gpu_memory(), 1, kernel.gpu_memory(), 1, conv.gpu_memory(), 1);

                conv.validate_gpu();
                conv.invalidate_cpu();
            } else {
                cpp_unreachable("Invalid conv implementation selection");
            }
        } else {
            cpp_unreachable("Invalid conv implementation selection");
        }
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const conv_1d_same_expr& expr) {
        return os << "conv1_same(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B>
struct etl_traits<etl::conv_1d_same_expr<A, B>> {
    using expr_t       = etl::conv_1d_same_expr<A, B>; ///< The expression type
    using left_expr_t  = std::decay_t<A>;              ///< The left sub expression type
    using right_expr_t = std::decay_t<B>;              ///< The right sub expression type
    using left_traits  = etl_traits<left_expr_t>;      ///< The left sub traits
    using right_traits = etl_traits<right_expr_t>;     ///< The right sub traits
    using value_type   = value_t<A>;                   ///< The value type of the expression

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
        return etl::dim<0, A>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, [[maybe_unused]] size_t d) {
        return etl::dim(e._a, 0);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::dim(e._a, 0);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return etl::dim<0, A>();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
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

/*!
 * \brief Creates an expression representing the 'same' 1D convolution of a and b.
 *
 * The convolution is applied with padding so that the output has
 * the same size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 *
 * \return an expression representing the 'same' 1D convolution of a and b
 */
template <etl_expr A, etl_expr B>
conv_1d_same_expr<detail::build_type<A>, detail::build_type<B>> conv_1d_same(A&& a, B&& b) {
    return conv_1d_same_expr<detail::build_type<A>, detail::build_type<B>>{a, b};
}

/*!
 * \brief Creates an expression representing the 'same' 1D convolution of a and b, the result will be stored in c
 *
 * The convolution is applied with padding so that the output has
 * the same size as the input.
 *
 * \param a The input expression
 * \param b The kernel expression
 * \param c The result
 *
 * \return an expression representing the 'same' 1D convolution of a and b
 */
template <etl_expr A, etl_expr B, etl_expr C>
auto conv_1d_same(A&& a, B&& b, C&& c) {
    c = conv_1d_same(a, b);

    return c;
}

} //end of namespace etl
