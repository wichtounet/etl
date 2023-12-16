//=======================================================================
// Copyright (c) 2014-2023 Baptiste Wicht
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
template <etl_2d A, etl_3d B, typename C>
struct batch_embedding_gradients_expr : base_temporary_expr_tern<batch_embedding_gradients_expr<A, B, C>, A, B, C> {
    using value_type = value_t<A>;                                   ///< The type of value of the expression
    using this_type  = batch_embedding_gradients_expr<A, B, C>;      ///< The type of this expression
    using base_type  = base_temporary_expr_tern<this_type, A, B, C>; ///< The base type
    using sub_traits = decay_traits<A>;                              ///< The traits of the sub type

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
    explicit batch_embedding_gradients_expr(A a, B b, C c) : base_type(a, b, c) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram lhs The output matrix
     */
    template <etl_2d L>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const B& b, [[maybe_unused]] const C& c, [[maybe_unused]] const L& lhs) {
        if constexpr (all_fast<A, B, C, L>) {
            static_assert(etl::dim<0, A>() == etl::dim<0, B>(), "Invalid dimensions for batch_embedding_gradients");
            static_assert(etl::dim<1, A>() == etl::dim<1, B>(), "Invalid dimensions for batch_embedding_gradients");
            static_assert(etl::dim<2, B>() == etl::dim<1, L>(), "Invalid dimensions for batch_embedding_gradients");
        } else {
            cpp_assert(etl::dim<0>(a) == etl::dim<0>(b), "Invalid dimensions for batch_embedding_gradients");
            cpp_assert(etl::dim<1>(a) == etl::dim<1>(b), "Invalid dimensions for batch_embedding_gradients");
            cpp_assert(etl::dim<2>(b) == etl::dim<1>(lhs), "Invalid dimensions for batch_embedding_gradients");
        }
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param lhs The expression to which assign
     */
    template <etl_expr L>
    void assign_to(L&& lhs) const {
        inc_counter("temp:assign");

        auto& a = this->a();
        auto& b = this->b();
        auto& c = this->c();

        check(a, b, c, lhs);

        const auto BB = etl::dim<0>(a);
        const auto I  = etl::dim<1>(a);

        standard_evaluator::pre_assign_rhs(a);
        standard_evaluator::pre_assign_rhs(b);

        lhs = 0;

        for (size_t bb = 0; bb < BB; ++bb) {
            for (size_t i = 0; i < I; ++i) {
                lhs(a(bb, i)) += b(bb)(i);
            }
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
    friend std::ostream& operator<<(std::ostream& os, const batch_embedding_gradients_expr& expr) {
        return os << "embedding_gradients(" << expr._a << ", " << expr._b << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename B, typename C>
struct etl_traits<etl::batch_embedding_gradients_expr<A, B, C>> {
    using expr_t     = etl::batch_embedding_gradients_expr<A, B, C>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;                              ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;                       ///< The sub traits
    using value_type = value_t<A>;                                   ///< The value type of the expression

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
    static constexpr size_t dim() requires(DD < 2) {
        return DD == 0 ? decay_traits<C>::template dim<0>() : decay_traits<B>::template dim<2>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, [[maybe_unused]] size_t d) {
        cpp_assert(d < 2, "Invalid dimensions access");

        return d == 0 ? etl::dim<0>(e._c) : etl::dim<2>(e._b);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return etl::dim<0>(e._c) * etl::dim<2>(e._b);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    static constexpr size_t size() {
        return decay_traits<C>::template dim<0>() * decay_traits<B>::template dim<2>();
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr size_t dimensions() {
        return 2;
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
 * \brief Returns the embeddings for the given sequence
 * \param value The input sequence
 * \param vocab The embedding vocabulary
 * \return The embeeddings of the given sequence.
 */
template <etl_2d I, etl_3d E, etl_expr W>
batch_embedding_gradients_expr<detail::build_type<I>, detail::build_type<E>, detail::build_type<W>> batch_embedding_gradients(const I& value,
                                                                                                                              const E& errors,
                                                                                                                              const W& vocab) {
    return batch_embedding_gradients_expr<detail::build_type<I>, detail::build_type<E>, detail::build_type<W>>{value, errors, vocab};
}

} //end of namespace etl
