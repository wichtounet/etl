//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file fft_expr.hpp
 * \brief Contains the FFT expressions.
*/

#pragma once

//Get the implementations
#include "etl/impl/fft.hpp"

namespace etl {

/*!
 * \brief A configurable expression for FFT
 * \tparam T The value type
 * \tparam D The number of dimensions of the FFT
 * \tparam Impl The implementation to use
 */
template <typename T, std::size_t D, typename Impl>
struct basic_fft_expr : impl_expr<basic_fft_expr<T, D, Impl>> {
    using this_type  = basic_fft_expr<T, D, Impl>; ///< The type of this expression
    using value_type = T;                          ///< The value type

    static constexpr const bool is_gpu = is_cufft_enabled; ///< Indicate if the expression is executed on GPU

    /*!
     * \brief The result type for a given sub expression type
     * \tparam A The sub epxpression type
     */
    template <typename A>
    using result_type = detail::expr_result_t<this_type, A>;

    /*!
     * \brief Apply the expression
     * \param a The sub expression
     * \param c The expression where to store the results
     */
    template <typename A, typename C>
    static void apply(A&& a, C&& c) {
        static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "Fast-Fourrier Transform only supported for ETL expressions");

        Impl::apply(
            make_temporary(std::forward<A>(a)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "fft";
    }

    /*!
     * \brief Returns the DDth dimension of the expression
     * \tparam A The sub expression type
     * \tparam DD The dimension to get
     * \return the DDth dimension of the expression
     */
    template <typename A, std::size_t DD>
    static constexpr std::size_t dim() {
        return decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A>
    static std::size_t dim(const A& a, std::size_t d) {
        return etl_traits<A>::dim(a, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The sub expression
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
        return etl::decay_traits<A>::size();
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
        return D;
    }
};

/*!
 * \brief A configurable expression for FFT
 * \tparam T The value type
 * \tparam D The number of dimensions of the FFT
 * \tparam Impl The implementation to use
 */
template <typename T, typename A, std::size_t D, typename Impl>
struct basic_fft_expr_alpha :
        temporary_expr_un<basic_fft_expr_alpha<T, A, D, Impl>, T, A, detail::expr_result_t_alpha<basic_fft_expr_alpha<T, A, D, Impl>, A>>,
        impl_expr_alpha<basic_fft_expr_alpha<T, A, D, Impl>> {

    using this_type      = basic_fft_expr_alpha<T, A, D, Impl>; ///< The type of this expression
    using value_type     = T;                                   ///< The value type
    using base_type      = temporary_expr_un<this_type, T, A, detail::expr_result_t_alpha<this_type, A>>;
    using base_type_impl = impl_expr_alpha<this_type>;

    static constexpr const bool is_gpu = is_cufft_enabled; ///< Indicate if the expression is executed on GPU

    /*!
     * \brief The result type for a given sub expression type
     */
    using result_type = detail::expr_result_t_alpha<this_type, A>;

    basic_fft_expr_alpha(A a) : base_type(a) {}

    /*!
     * \brief Apply the expression
     * \param c The expression where to store the results
     */
    template <typename C>
    void apply(C&& c) const {
        static_assert(is_etl_expr<A>::value && is_etl_expr<C>::value, "Fast-Fourrier Transform only supported for ETL expressions");

        Impl::apply(make_temporary(this->a()), std::forward<C>(c));
    }

    auto allocate() const {
        return base_type_impl::allocate(this->a());
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static std::string desc() noexcept {
        return "fft";
    }
};

/*!
 * \brief Specialization for temporary_unary_expr.
 */
template <typename T, typename A, std::size_t D, typename Impl>
struct etl_traits<etl::basic_fft_expr_alpha<T, A, D, Impl>> {
    using expr_t = etl::basic_fft_expr_alpha<T, A, D, Impl>;
    using a_t    = std::decay_t<A>;

    using value_type = T;

    static constexpr const bool is_etl                  = true;                           ///< Indicates if the type is an ETL type
    static constexpr const bool is_transformer          = false;                          ///< Indicates if the type is a transformer
    static constexpr const bool is_view                 = false;                          ///< Indicates if the type is a view
    static constexpr const bool is_magic_view           = false;                          ///< Indicates if the type is a magic view
    static constexpr const bool is_fast                 = etl_traits<a_t>::is_fast;       ///< Indicates if the expression is fast
    static constexpr const bool is_linear               = true;                           ///< Indicates if the expression is linear
    static constexpr const bool is_value                = false;                          ///< Indicates if the expression is of value type
    static constexpr const bool is_generator            = false;                          ///< Indicates if the expression is a generated
    static constexpr const bool needs_temporary_visitor = true;                           ///< Indicates if the expression needs a temporary visitor
    static constexpr const bool needs_evaluator_visitor = true;                           ///< Indicaes if the expression needs an evaluator visitor
    static constexpr const order storage_order          = etl_traits<a_t>::storage_order; ///< The expression storage order

    static constexpr const bool is_gpu = is_cufft_enabled;

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    using vectorizable = std::true_type;

    /*!
     * \brief Returns the size of the given expression
     * \param v The expression to get the size for
     * \returns the size of the given expression
     */
    static std::size_t size(const expr_t& v) {
        return etl_traits<a_t>::size(v.a());
    }

    /*!
     * \brief Returns the dth dimension of the given expression
     * \param v The expression
     * \param d The dimension to get
     * \return The dth dimension of the given expression
     */
    static std::size_t dim(const expr_t& v, std::size_t d) {
        return etl_traits<a_t>::dim(v.a(), d);
    }

    /*!
     * \brief Returns the size of an expression of this fast type.
     * \returns the size of an expression of this fast type.
     */
    static constexpr std::size_t size() {
        return etl_traits<a_t>::size();
    }

    /*!
     * \brief Returns the Dth dimension of an expression of this type
     * \tparam DD The dimension to get
     * \return the Dth dimension of an expression of this type
     */
    template <std::size_t DD>
    static constexpr std::size_t dim() {
        return etl_traits<a_t>::template dim<DD>();
    }

    /*!
     * \brief Returns the number of expressions for this type
     * \return the number of dimensions of this type
     */
    static constexpr std::size_t dimensions() {
        return D;
    }
};

/*!
 * \brief Expression for 1D FFT
 */
template <typename T, typename A>
using fft1_expr_alpha = basic_fft_expr_alpha<T, A, 1, detail::fft1_impl>;

/*!
 * \brief Expression for 1D Inverse FFT
 */
template <typename T, typename A>
using ifft1_expr_alpha = basic_fft_expr_alpha<T, A, 1, detail::ifft1_impl>;

/*!
 * \brief Expression for 1D FFT
 */
template <typename T>
using fft1_expr = basic_fft_expr<T, 1, detail::fft1_impl>;

/*!
 * \brief Expression for 1D Inverse FFT
 */
template <typename T>
using ifft1_expr = basic_fft_expr<T, 1, detail::ifft1_impl>;

/*!
 * \brief Expression for 1D Inverse FFT in real
 */
template <typename T, typename A>
using ifft1_real_expr_alpha = basic_fft_expr_alpha<T, A, 1, detail::ifft1_real_impl>;

/*!
 * \brief Expression for 1D Inverse FFT in real
 */
template <typename T>
using ifft1_real_expr = basic_fft_expr<T, 1, detail::ifft1_real_impl>;

/*!
 * \brief Expression for 2D FFT
 */
template <typename T>
using fft2_expr = basic_fft_expr<T, 2, detail::fft2_impl>;

/*!
 * \brief Expression for 2D Inverse FFT
 */
template <typename T>
using ifft2_expr = basic_fft_expr<T, 2, detail::ifft2_impl>;

/*!
 * \brief Expression for 2D Inverse FFT in real
 */
template <typename T>
using ifft2_real_expr = basic_fft_expr<T, 2, detail::ifft2_real_impl>;

/*!
 * \brief Expression for many 1D FFT done at once
 */
template <typename T, typename A>
using fft1_many_expr_alpha = basic_fft_expr_alpha<T, A, 2, detail::fft1_many_impl>;

/*!
 * \brief Expression for many 1D Inverse FFT done at once
 */
template <typename T, typename A>
using ifft1_many_expr_alpha = basic_fft_expr_alpha<T, A, 2, detail::ifft1_many_impl>;

/*!
 * \brief Expression for many 1D FFT done at once
 */
template <typename T>
using fft1_many_expr = basic_fft_expr<T, 2, detail::fft1_many_impl>;

/*!
 * \brief Expression for many 1D Inverse FFT done at once
 */
template <typename T>
using ifft1_many_expr = basic_fft_expr<T, 2, detail::ifft1_many_impl>;

/*!
 * \brief Expression for many 2D FFT done at once
 */
template <typename T>
using fft2_many_expr = basic_fft_expr<T, 3, detail::fft2_many_impl>;

/*!
 * \brief Expression for many 2D Inverse FFT done at once
 */
template <typename T>
using ifft2_many_expr = basic_fft_expr<T, 3, detail::ifft2_many_impl>;

} //end of namespace etl
