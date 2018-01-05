//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A, typename B>
struct batch_embedding_lookup_expr : base_temporary_expr_bin<batch_embedding_lookup_expr<A, B>, A, B> {
    using value_type = value_t<A>;                               ///< The type of value of the expression
    using this_type  = batch_embedding_lookup_expr<A, B>;        ///< The type of this expression
    using base_type  = base_temporary_expr_bin<this_type, A, B>; ///< The base type
    using sub_traits = decay_traits<A>;                          ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = false;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit batch_embedding_lookup_expr(A a, B b) : base_type(a, b) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C, cpp_enable_iff(all_fast<A, B, C>)>
    static void check(const A& a, const B& b, const C& c) {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);

        static_assert(etl::dimensions<A>() == 2, "The input of batch_embedding_lookup is a 2d matrix");
        static_assert(etl::dimensions<B>() == 2, "The vocabulary input of batch_embedding_lookup is a 2d matrix");
        static_assert(etl::dimensions<C>() == 3, "The output of batch_embedding_lookup is 3d matrix");

        static_assert(etl::dim<0, A>() == etl::dim<0, C>(), "Invalid dimensions for batch_embedding_lookup");
        static_assert(etl::dim<1, A>() == etl::dim<1, C>(), "Invalid dimensions for batch_embedding_lookup");
        static_assert(etl::dim<1, B>() == etl::dim<2, C>(), "Invalid dimensions for batch_embedding_lookup");
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C, cpp_disable_iff(all_fast<A, B, C>)>
    static void check(const A& a, const B& b, const C& c) {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);

        static_assert(etl::dimensions<A>() == 2, "The input of batch_embedding_lookup is a 2d matrix");
        static_assert(etl::dimensions<B>() == 2, "The vocabulary input of batch_embedding_lookup is a 2d matrix");
        static_assert(etl::dimensions<C>() == 3, "The output of batch_embedding_lookup is 3d matrix");

        cpp_assert(etl::dim<0>(a) == etl::dim<0>(c), "Invalid dimensions for batch_embedding_lookup");
        cpp_assert(etl::dim<1>(a) == etl::dim<1>(c), "Invalid dimensions for batch_embedding_lookup");
        cpp_assert(etl::dim<1>(b) == etl::dim<2>(c), "Invalid dimensions for batch_embedding_lookup");
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_to(L&& lhs) const {
        static_assert(all_etl_expr<A, B, L>, "batch_embedding_lookup only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        const auto BB = etl::dim<0>(a);
        const auto I = etl::dim<1>(a);

        for (size_t bb = 0; bb < BB; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                lhs(bb)(i) = b(a(bb, i));
            }
        }
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_2d only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        const auto BB = etl::dim<0>(a);
        const auto I = etl::dim<1>(a);

        for (size_t bb = 0; bb < BB; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                lhs(bb)(i) += b(a(bb, i));
            }
        }
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_2d only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        const auto BB = etl::dim<0>(a);
        const auto I = etl::dim<1>(a);

        for (size_t bb = 0; bb < BB; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                lhs(bb)(i) -= b(a(bb, i));
            }
        }
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_2d only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        const auto BB = etl::dim<0>(a);
        const auto I = etl::dim<1>(a);

        for (size_t bb = 0; bb < BB; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                lhs(bb)(i) *= b(a(bb, i));
            }
        }
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_2d only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        const auto BB = etl::dim<0>(a);
        const auto I = etl::dim<1>(a);

        for (size_t bb = 0; bb < BB; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                lhs(bb)(i) /= bb(a(b, i));
            }
        }
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        static_assert(all_etl_expr<A, L>, "bias_batch_mean_2d only supported for ETL expressions");

        auto& a = this->a();
        auto& b = this->b();

        check(a, b, lhs);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        const auto BB = etl::dim<0>(a);
        const auto I = etl::dim<1>(a);

        for (size_t bb = 0; bb < BB; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                lhs(bb)(i) %= b(a(bb, i));
            }
        }
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const batch_embedding_lookup_expr& expr) {
        return os << "embedding_lookup(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B>
struct etl_traits<etl::batch_embedding_lookup_expr<A, B>> {
    using expr_t     = etl::batch_embedding_lookup_expr<A, B>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                        ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                 ///< The sub traits
    using value_type = value_t<A>;                             ///< The value type of the expression

    static constexpr bool is_etl         = true;                      ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                     ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                     ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                     ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast;       ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                     ///< Indicates if the expression is linear
    static constexpr bool is_thread_safe = true;                      ///< Indicates if the expression is thread safe
    static constexpr bool is_value       = false;                     ///< Indicates if the expression is of value type
    static constexpr bool is_direct      = true;                      ///< Indicates if the expression has direct memory access
    static constexpr bool is_generator   = false;                     ///< Indicates if the expression is a generator
    static constexpr bool is_padded      = false;                     ///< Indicates if the expression is padded
    static constexpr bool is_aligned     = true;                      ///< Indicates if the expression is padded
    static constexpr bool is_temporary   = true;                      ///< Indicates if the expression needs a evaluator visitor
    static constexpr order storage_order = sub_traits::storage_order; ///< The expression's storage order
    static constexpr bool gpu_computable = false;                     ///< Indicates if the expression can be computed on GPU

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
        static_assert(DD < 3, "Invalid dimensions access");

        if(DD == 0){
            return decay_traits<A>::template dim<0>();
        } else if(DD == 1){
            return decay_traits<A>::template dim<1>();
        } else {
            return decay_traits<B>::template dim<1>();
        }
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        cpp_assert(d < 3, "Invalid dimensions access");
        cpp_unused(d);
        if(d == 0){
            return etl::dim<0>(e._a);
        } else if(d == 1){
            return etl::dim<1>(e._a);
        } else {
            return etl::dim<1>(e._b);
        }
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::dim<0>(e._a) * etl::dim<1>(e._a) * etl::dim<1>(e._b);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return decay_traits<A>::template dim<0>() * decay_traits<A>::template dim<1>() * decay_traits<B>::template dim<1>();
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
 * \brief Returns the embeddings for the given sequence
 * \param value The input sequence
 * \param vocab The embedding vocabulary
 * \return The embeeddings of the given sequence.
 */
template <typename I, typename V>
batch_embedding_lookup_expr<detail::build_type<I>, detail::build_type<V>> batch_embedding_lookup(const I& value, const V& vocab) {
    static_assert(all_etl_expr<I, V>, "etl::batch_embedding_lookup can only be used on ETL expressions");
    static_assert(is_2d<I>, "etl::batch_embedding_lookup is only defined for 2d input");
    static_assert(is_2d<V>, "etl::batch_embedding_lookup is only defined for 2d vocabulary");

    return batch_embedding_lookup_expr<detail::build_type<I>, detail::build_type<V>>{value, vocab};
}

} //end of namespace etl
