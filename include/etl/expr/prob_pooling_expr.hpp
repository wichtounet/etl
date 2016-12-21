//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains the probabilistic pooling expressions.
*/

#pragma once

//Get the implementations
#include "etl/impl/prob_pooling.hpp"

namespace etl {

/*!
 * \brief A configurable expression for FFT
 * \tparam T The value type
 * \tparam D The number of dimensions of the FFT
 * \tparam Impl The implementation to use
 */
template <typename T, size_t D, size_t C1, size_t C2, typename Impl>
struct basic_pmp_h_expr : impl_expr<basic_pmp_h_expr<T, D, C1, C2, Impl>, T> {
    using this_type  = basic_pmp_h_expr<T, D, C1, C2, Impl>; ///< The type of this expression
    using value_type = T;                          ///< The value type

    static constexpr bool is_gpu = false; ///< Indicate if the expression is executed on GPU

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, T, A>;

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "Probabilistic Max Pooling only supported for ETL expressions");

        Impl::apply(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "pmp_h";
    }

    /*!
     * \brief Returns the DDth dimension of the expression
     * \tparam A The sub expression type
     * \tparam DD The dimension to get
     * \return the DDth dimension of the expression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return
              DD == 0 ? decay_traits<A>::template dim<0>() / C1
            : DD == 1 ? decay_traits<A>::template dim<1>() / C2
                      : decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        return
              d == 0 ? decay_traits<A>::dim(a, 0) / C1
            : d == 1 ? decay_traits<A>::dim(a, 1) / C2
                     : decay_traits<A>::dim(a, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    static std::size_t size(const A& a) {
        return etl::size(a) / (C1 * C2);
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A>
    static constexpr std::size_t size() {
        return etl::decay_traits<A>::size() / (C1 * C2);
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A>
    static constexpr etl::order order() {
        return decay_traits<A>::storage_order;
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Expression for 2D PMP (hidden)
 */
template <typename T, size_t D, size_t C1, size_t C2>
using pmp_h_expr = basic_pmp_h_expr<T, D, 1, 1, detail::pmp_h_impl<D, C1, C2>>;

/*!
 * \brief Expression for 2D PMP (hidden)
 */
template <typename T, size_t D, size_t C1, size_t C2>
using pmp_p_expr = basic_pmp_h_expr<T, D, C1, C2, detail::pmp_p_impl<D, C1, C2>>;

/*!
 * \brief A configurable expression for FFT
 * \tparam T The value type
 * \tparam D The number of dimensions of the FFT
 * \tparam Impl The implementation to use
 */
template <typename T, std::size_t D, typename Impl>
struct dyn_basic_pmp_h_expr : impl_expr<dyn_basic_pmp_h_expr<T, D, Impl>, T> {
    using this_type  = dyn_basic_pmp_h_expr<T, D, Impl>; ///< The type of this expression
    using value_type = T;                          ///< The value type

    static constexpr bool is_gpu = false; ///< Indicate if the expression is executed on GPU

    size_t d1; ///< The divisor for the first dimension
    size_t d2; ///< The divisor for the second dimension
    Impl impl; ///< The implementation operator

    /*!
     * \brief Construct a new dyn_basic_pmp_h_expr and forward the arguments to the implementation
     */
    dyn_basic_pmp_h_expr(size_t d1, size_t d2, size_t c1, size_t c2) : d1(d1), d2(d2), impl(c1, c2){
        //Nothing else to init
    }

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::dyn_expr_result_t<this_type, T, A>;

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    void apply(A&& a, C&& c) const {
        static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "Probabilistic Max Pooling only supported for ETL expressions");

        impl.apply(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "dyn_pmp_h";
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    std::size_t dim(const A& a, std::size_t d) const {
        return
              d == 0 ? etl_traits<A>::dim(a, 0) / d1
            : d == 1 ? etl_traits<A>::dim(a, 1) / d2
                     : etl_traits<A>::dim(a, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
     * \return the size of the expression
     */
    template <typename A>
    std::size_t size(const A& a) const {
        return etl::size(a) / (d1 * d2);
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A>
    static constexpr etl::order order() {
        return decay_traits<A>::storage_order;
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Expression dyn probabilistc max pooling (hidden)
 */
template <typename T, size_t D>
using dyn_pmp_h_expr = dyn_basic_pmp_h_expr<T, D, detail::dyn_pmp_h_impl<D>>;

/*!
 * \brief Expression dyn probabilistc max pooling (pooling)
 */
template <typename T, size_t D>
using dyn_pmp_p_expr = dyn_basic_pmp_h_expr<T, D, detail::dyn_pmp_p_impl<D>>;

} //end of namespace etl
