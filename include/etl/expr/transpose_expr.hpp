//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/expr/base_temporary_expr.hpp"

//Get the implementations
#include "etl/impl/transpose.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam A The transposed type
 */
template <typename A>
struct transpose_expr : base_temporary_expr_un<transpose_expr<A>, A> {
    using value_type = value_t<A>;                           ///< The type of value of the expression
    using this_type  = transpose_expr<A>;                    ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>; ///< The base type
    using sub_traits = decay_traits<A>;                      ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = cublas_enabled && all_floating<A>;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit transpose_expr(A a) : base_type(a) {
        //Nothing else to init
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename C>
    static void check([[maybe_unused]] const A& a, [[maybe_unused]] const C& c) {
        static constexpr etl::order order_lhs = decay_traits<C>::storage_order;
        static constexpr etl::order order_rhs = decay_traits<A>::storage_order;

        [[maybe_unused]] static constexpr bool rm_to_rm = order_lhs == etl::order::RowMajor && order_rhs == etl::order::RowMajor;
        [[maybe_unused]] static constexpr bool cm_to_cm = order_lhs == etl::order::ColumnMajor && order_rhs == etl::order::ColumnMajor;
        [[maybe_unused]] static constexpr bool rm_to_cm = order_lhs == etl::order::RowMajor && order_rhs == etl::order::ColumnMajor;
        [[maybe_unused]] static constexpr bool cm_to_rm = order_lhs == etl::order::ColumnMajor && order_rhs == etl::order::RowMajor;

        if constexpr (all_fast<A, C>) {
            static constexpr size_t L1 = decay_traits<C>::template dim<0>();
            static constexpr size_t L2 = decay_traits<C>::template dim<1>();
            static constexpr size_t R1 = decay_traits<A>::template dim<0>();
            static constexpr size_t R2 = decay_traits<A>::template dim<1>();

            // Case 1: RM -> RM
            static_assert(!rm_to_rm || (L1 == R2 && L2 == R1), "Invalid dimensions for transposition");

            // Case 2: CM -> CM
            static_assert(!cm_to_cm || (L1 == R2 && L2 == R1), "Invalid dimensions for transposition");

            // Case 3: RM -> CM (two possible cases)
            static_assert(!rm_to_cm || ((L1 == R2 && L2 == R1) || (L1 == R1 && L2 == R2)), "Invalid dimensions for transposition");

            // Case 4: RM -> CM (two possible cases)
            static_assert(!cm_to_rm || ((L1 == R2 && L2 == R1) || (L1 == R1 && L2 == R2)), "Invalid dimensions for transposition");
        } else {
            [[maybe_unused]] const size_t L1 = etl::dim<0>(c);
            [[maybe_unused]] const size_t L2 = etl::dim<1>(c);
            [[maybe_unused]] const size_t R1 = etl::dim<0>(a);
            [[maybe_unused]] const size_t R2 = etl::dim<1>(a);

            // Case 1: RM -> RM
            cpp_assert(!rm_to_rm || (L1 == R2 && L2 == R1), "Invalid dimensions for transposition");

            // Case 2: CM -> CM
            cpp_assert(!cm_to_cm || (L1 == R2 && L2 == R1), "Invalid dimensions for transposition");

            // Case 3: RM -> CM (two possible cases)
            cpp_assert(!rm_to_cm || ((L1 == R2 && L2 == R1) || (L1 == R1 && L2 == R2)), "Invalid dimensions for transposition");

            // Case 4: RM -> CM (two possible cases)
            cpp_assert(!cm_to_rm || ((L1 == R2 && L2 == R1) || (L1 == R1 && L2 == R2)), "Invalid dimensions for transposition");
        }
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template <etl_expr C>
    void assign_to(C&& c) const {
        if constexpr (decay_traits<C>::storage_order == storage_order) {
            inc_counter("temp:assign");

            auto& a = this->a();

            check(a, c);

            detail::transpose::apply(a, c);
        } else {
            std_assign_evaluate(*this, c);
        }
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, std::forward<L>(lhs));
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const transpose_expr& expr) {
        return os << "trans(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A>
struct etl_traits<etl::transpose_expr<A>> {
    using expr_t     = etl::transpose_expr<A>; ///< The expression type
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
        return DD == 0 ? decay_traits<A>::template dim<1>() : decay_traits<A>::template dim<0>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        return d == 0 ? etl::dim<1>(e._a) : etl::dim<0>(e._a);
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

} //end of namespace etl
