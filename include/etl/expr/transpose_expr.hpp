//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

//Get the implementations
#include "etl/impl/transpose.hpp"

namespace etl {

/*!
 * \brief A transposition expression.
 * \tparam T The value type
 */
template <typename T>
struct transpose_expr : impl_expr<transpose_expr<T>> {
    using value_type = T;                 ///< The type of value of the expression
    using this_type  = transpose_expr<T>; ///< The type of this expression

    static constexpr bool is_gpu = cublas_enabled; ///< Indicates if the expression runs on GPU

    /*!
     * \brief The result type for given sub types
     * \tparam A the sub expression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename A, typename C, cpp_enable_if(all_fast<A,C>::value)>
    static void check(const A& a, const C& c) {
        cpp_unused(a);
        cpp_unused(c);

        static_assert(decay_traits<A>::template dim<0>() == decay_traits<C>::template dim<1>(), "Invalid dimensions for transposition");
        static_assert(decay_traits<C>::template dim<0>() == decay_traits<A>::template dim<1>(), "Invalid dimensions for transposition");
    }

    /*!
     * \brief Validate the transposition dimensions
     * \param a The input matrix
     * \þaram c The output matrix
     */
    template <typename A, typename C, cpp_disable_if(all_fast<A,C>::value)>
    static void check(const A& a, const C& c) {
        cpp_unused(a);
        cpp_unused(c);

        cpp_assert(etl::dim<0>(a) == etl::dim<1>(c), "Invalid dimensions for transposition");
        cpp_assert(etl::dim<1>(a) == etl::dim<0>(c), "Invalid dimensions for transposition");
    }

    /*!
     * \brief Apply the expression
     * \param a The input
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(all_etl_expr<A, C>::value, "Transpose only supported for ETL expressions");

        check(a, c);

        detail::transpose::apply(make_temporary(std::forward<A>(a)), std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static constexpr const char* desc() noexcept {
        return "transpose";
    }

    /*!
     * \brief Returns the DDth dimension of the expression
     * \return the DDth dimension of the expression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return DD == 0 ? decay_traits<A>::template dim<1>() : decay_traits<A>::template dim<0>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The left hand side
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        return d == 0 ? etl::dim<1>(a) : etl::dim<0>(a);
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The left hand side
     * \return the size of the expression
     */
    template <typename A>
    static std::size_t size(const A& a) {
        return etl::size(a);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A>
    static constexpr std::size_t size() {
        return decay_traits<A>::size();
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A>
    static constexpr etl::order order() {
        return etl::order::RowMajor;
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return 2;
    }
};

} //end of namespace etl
