//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/egblas/transpose_front.hpp"

namespace etl {

/*!
 * \brief A transposition expression for the first layers.
 * \tparam A The transposed type
 */
template <etl_expr A>
struct transpose_front_expr : base_temporary_expr_un<transpose_front_expr<A>, A> {
    using value_type = value_t<A>;                           ///< The type of value of the expression
    using this_type  = transpose_front_expr<A>;                    ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>; ///< The base type
    using sub_traits = decay_traits<A>;                      ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = all_row_major<A> && all_floating<A> && impl::egblas::has_stranspose_front;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit transpose_front_expr(A a) : base_type(a) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <etl_expr C>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const C& c) {
        if constexpr (all_fast<A, C>) {
            static_assert(etl::dim<0, A>() == etl::dim<1, C>(), "Invalid dimensions for front transposition");
            static_assert(etl::dim<1, A>() == etl::dim<0, C>(), "Invalid dimensions for front transposition");
            static_assert(etl::dim<2, A>() == etl::dim<2, C>(), "Invalid dimensions for front transposition");
        } else {
            cpp_assert(etl::dim<0>(a) == etl::dim<1>(c), "Invalid dimensions for front transposition");
            cpp_assert(etl::dim<1>(a) == etl::dim<0>(c), "Invalid dimensions for front transposition");
            cpp_assert(etl::dim<2>(a) == etl::dim<2>(c), "Invalid dimensions for front transposition");
        }
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template <etl_expr C>
    void assign_to(C&& lhs) const {
        auto& a = this->a();

        check(a, lhs);

        const auto B = etl::dim<0>(a);
        const auto K = etl::dim<1>(a);

        if constexpr (all_row_major<A, C> && all_floating<A, C> && impl::egblas::has_stranspose_front) {
            decltype(auto) t1 = smart_forward_gpu(a);
            t1.ensure_gpu_up_to_date();

            lhs.ensure_gpu_allocated();

            impl::egblas::transpose_front(B, K, etl::size(a) / (B * K), t1.gpu_memory(), lhs.gpu_memory());

            lhs.validate_gpu();
            lhs.invalidate_cpu();
        } else {
            auto batch_fun_b = [&](const size_t first, const size_t last) {
                for (size_t b = first; b < last; ++b) {
                    for (size_t k = 0; k < K; ++k) {
                        lhs(k)(b) = a(b)(k);
                    }
                }
            };

            // Ideally, this should be optimized to not use hyper thread
            // for large containers, but the threshold is tricky to define
            engine_dispatch_1d_serial(batch_fun_b, 0, B, 8UL);

            a.ensure_cpu_up_to_date();

            lhs.validate_cpu();
            lhs.invalidate_gpu();
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
    friend std::ostream& operator<<(std::ostream& os, const transpose_front_expr& expr) {
        return os << "trans_front(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A>
struct etl_traits<etl::transpose_front_expr<A>> {
    using expr_t     = etl::transpose_front_expr<A>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;        ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>; ///< The sub traits
    using value_type = value_t<A>;             ///< The value type of the expression

    static constexpr bool is_etl         = true;                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast;                  ///< Indicates if the expression is fast
    static constexpr bool is_linear      = true;                                 ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;                                 ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                                ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = true;                                 ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                                ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                                ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = true;                                 ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = true;                                 ///< Indicates if the expression needs a evaluator visitor
    static constexpr bool gpu_computable = is_gpu_t<value_type> && cuda_enabled; ///< Indicates if the expression can be computed on GPU
    static constexpr order storage_order = sub_traits::storage_order;            ///< The expression's storage order

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
        if (DD == 0) {
            return sub_traits::template dim<1>();
        } else if (DD == 1) {
            return sub_traits::template dim<0>();
        } else {
            return sub_traits::template dim<DD>();
        }
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        if (d == 0) {
            return etl::dim<1>(e._a);
        } else if (d == 1) {
            return etl::dim<0>(e._a);
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
        return etl::size(e._a);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return sub_traits::size();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return sub_traits::dimensions();
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
