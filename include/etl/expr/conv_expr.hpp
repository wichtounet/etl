//=======================================================================
// Copyright (c) 2014-2016 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

//Get the implementations
#include "etl/impl/conv.hpp"

namespace etl {

/*!
 * \brief A basic configurable convolution expr
 * \tparam T The value type
 * \tparam D The dimensions of convolution
 * \tparam TT The convolution type
 * \tparam Impl The implementation class
 */
template <typename T, std::size_t D, conv_type TT, typename Impl, std::size_t C4 = 0>
struct basic_conv_expr : impl_expr<basic_conv_expr<T, D, TT, Impl, C4>> {
    static_assert(D > 0, "0D convolution is not valid");

    using value_type = T;                                   ///< The type of value of the expression
    using this_type  = basic_conv_expr<T, D, TT, Impl, C4>; ///< The type of this expression

    static constexpr const bool is_gpu = is_cufft_enabled || is_cudnn_enabled; ///< Indicates if the expression runs on GPU

    /*!
     * \brief The result type for given sub types
     * \tparam A The left hand side epxpression type
     * \tparam B The right hand side epxpression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, A, B>;

    /*!
     * \brief Validate the convolutiond dimensions
     * \param a The input matrix
     * \þaram b The kernel matrix
     * \þaram c The output matrix
     */
    template <typename A, typename B, typename C, cpp_enable_if(all_fast<A,B,C>::value)>
    static void check(const A& a, const B& b, const C& c) {
        cpp_unused(a);
        cpp_unused(b);
        cpp_unused(c);

        Impl::template check<A,B,C>();
    }

    /*!
     * \brief Validate the convolutiond dimensions
     * \param a The input matrix
     * \þaram b The kernel matrix
     * \þaram c The output matrix
     */
    template <typename A, typename B, typename C, cpp_disable_if(all_fast<A,B,C>::value)>
    static void check(const A& a, const B& b, const C& c) {
        Impl::check(a, b, c);
    }

    /*!
     * \brief Apply the expression
     * \param a The left hand side
     * \param b The right hand side
     * \param c The expression where to store the results
     */
    template <typename A, typename B, typename C>
    static void apply(A&& a, B&& b, C&& c) {
        static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

        check(a, b, c);

        Impl::apply(
            make_temporary(std::forward<A>(a)),
            make_temporary(std::forward<B>(b)),
            std::forward<C>(c));
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static constexpr const char* desc() noexcept {
        return Impl::desc();
    }

    /*!
     * \brief Returns the DDth dimension of the expression
     * \return the DDth dimension of the expression
     */
    template <typename A, typename B, std::size_t DD>
    static constexpr std::size_t dim() {
        return Impl::template dim<DD, A, B>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The left hand side
     * \param b The right hand side
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A, typename B>
    static std::size_t dim(const A& a, const B& b, std::size_t d) {
        return Impl::dim(d, a, b);
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The left hand side
     * \param b The right hand side
     * \return the size of the expression
     */
    template <typename A, typename B>
    static std::size_t size(const A& a, const B& b) {
        std::size_t acc = 1;
        for (std::size_t i = 0; i < dimensions(); ++i) {
            acc *= this_type::dim(a, b, i);
        }
        return acc;
    }

    /*!
     * \brief Returns the multiplicative sum of the dimensions at the given indices
     * \return the multiplicative sum of the dimensions at the given indices
     */
    template <typename A, typename B, std::size_t... I>
    static constexpr std::size_t size_mul(const std::index_sequence<I...>& /*seq*/) {
        return mul_all<this_type::dim<A, B, I>()...>::value;
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A, typename B>
    static constexpr std::size_t size() {
        return size_mul<A, B>(std::make_index_sequence<dimensions()>());
    }

    /*!
     * \brief Returns the storage order of the expression.
     * \return the storage order of the expression
     */
    template <typename A, typename B>
    static constexpr etl::order order() {
        return etl::order::RowMajor;
    }

    /*!
     * \brief Returns the number of dimensions of the expression
     * \return the number of dimensions of the expression
     */
    static constexpr std::size_t dimensions() {
        return is_multi(TT) ? D + 1 : D;
    }
};

//1D convolution

/*!
 * \brief Expression for 1D valid convolution
 */
template <typename T>
using conv1_valid_expr = basic_conv_expr<T, 1, conv_type::VALID, detail::conv1_valid_impl>;

/*!
 * \brief Expression for 1D same convolution
 */
template <typename T>
using conv1_same_expr = basic_conv_expr<T, 1, conv_type::SAME, detail::conv1_same_impl>;

/*!
 * \brief Expression for 1D full convolution
 */
template <typename T>
using conv1_full_expr = basic_conv_expr<T, 1, conv_type::FULL, detail::conv1_full_impl>;

//2D convolutions

/*!
 * \brief Expression for 2D valid convolution
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv2_valid_expr = basic_conv_expr<T, 2, conv_type::VALID, detail::conv2_valid_impl<S1, S2, P1, P2>>;

/*!
 * \brief Expression for 2D valid convolution
 */
template <typename T>
using conv2_valid_flipped_expr = basic_conv_expr<T, 2, conv_type::VALID, detail::conv2_valid_flipped_impl>;

/*!
 * \brief Expression for 4D valid convolution
 */
template <typename T>
using conv4_valid_expr = basic_conv_expr<T, 4, conv_type::VALID, detail::conv4_valid_impl, 1>;

/*!
 * \brief Expression for 4D valid convolution
 */
template <typename T>
using conv4_valid_filter_expr = basic_conv_expr<T, 4, conv_type::VALID, detail::conv4_valid_filter_impl, 2>;

/*!
 * \brief Expression for 4D valid convolution
 */
template <typename T>
using conv4_valid_filter_flipped_expr = basic_conv_expr<T, 4, conv_type::VALID, detail::conv4_valid_filter_flipped_impl, 2>;

/*!
 * \brief Expression for 4D valid convolution
 */
template <typename T>
using conv4_full_expr = basic_conv_expr<T, 4, conv_type::FULL, detail::conv4_full_impl, 1>;

/*!
 * \brief Expression for 4D valid convolution
 */
template <typename T>
using conv4_valid_flipped_expr = basic_conv_expr<T, 4, conv_type::VALID, detail::conv4_valid_flipped_impl, 1>;

/*!
 * \brief Expression for 4D valid convolution
 */
template <typename T>
using conv4_full_flipped_expr = basic_conv_expr<T, 4, conv_type::FULL, detail::conv4_full_flipped_impl, 1>;

/*!
 * \brief Expression for 2D valid convolution, with multiple kernels
 */
template <typename T>
using conv2_valid_multi_expr = basic_conv_expr<T, 2, conv_type::VALID_MULTI, detail::conv2_valid_multi_impl>;

/*!
 * \brief Expression for 2D valid convolution, with multiple flipped kernels
 */
template <typename T>
using conv2_valid_multi_flipped_expr = basic_conv_expr<T, 2, conv_type::VALID_MULTI, detail::conv2_valid_multi_flipped_impl>;

/*!
 * \brief Expression for 2D same convolution, with multiple kernels
 */
template <typename T>
using conv2_same_multi_expr = basic_conv_expr<T, 2, conv_type::SAME_MULTI, detail::conv2_same_multi_impl>;

/*!
 * \brief Expression for 2D same convolution, with multiple flipped kernels
 */
template <typename T>
using conv2_same_multi_flipped_expr = basic_conv_expr<T, 2, conv_type::SAME_MULTI, detail::conv2_same_multi_flipped_impl>;

/*!
 * \brief Expression for 2D full convolution, with multiple kernels
 */
template <typename T>
using conv2_full_multi_expr = basic_conv_expr<T, 2, conv_type::FULL_MULTI, detail::conv2_full_multi_impl>;

/*!
 * \brief Expression for 2D full convolution, with multiple flipped kernels
 */
template <typename T>
using conv2_full_multi_flipped_expr = basic_conv_expr<T, 2, conv_type::FULL_MULTI, detail::conv2_full_multi_flipped_impl>;

/*!
 * \brief Expression for 2D same convolution
 */
template <typename T>
using conv2_same_expr = basic_conv_expr<T, 2, conv_type::SAME, detail::conv2_same_impl>;

/*!
 * \brief Expression for 2D same convolution
 */
template <typename T>
using conv2_same_flipped_expr = basic_conv_expr<T, 2, conv_type::SAME, detail::conv2_same_flipped_impl>;

/*!
 * \brief Expression for 2D full convolution
 */
template <typename T>
using conv2_full_expr = basic_conv_expr<T, 2, conv_type::FULL, detail::conv2_full_impl>;

/*!
 * \brief Expression for 2D full convolution
 */
template <typename T>
using conv2_full_flipped_expr = basic_conv_expr<T, 2, conv_type::FULL, detail::conv2_full_flipped_impl>;

//>2D convolutions

/*!
 * \brief Expression for >2D valid convolution
 */
template <typename T, std::size_t D>
using conv_deep_valid_expr = basic_conv_expr<T, D, conv_type::VALID, detail::conv_deep_valid_impl>;

/*!
 * \brief Expression for >2D same convolution
 */
template <typename T, std::size_t D>
using conv_deep_same_expr = basic_conv_expr<T, D, conv_type::SAME, detail::conv_deep_same_impl>;

/*!
 * \brief Expression for >2D full convolution
 */
template <typename T, std::size_t D>
using conv_deep_full_expr = basic_conv_expr<T, D, conv_type::FULL, detail::conv_deep_full_impl>;

} //end of namespace etl
