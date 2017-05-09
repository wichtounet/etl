//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
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
 * \tparam Impl The implementation class
 */
template <typename T, size_t D, typename Impl>
struct basic_conv_expr : impl_expr<basic_conv_expr<T, D, Impl>, T> {
    static_assert(D > 0, "0D convolution is not valid");

    using value_type = T;                           ///< The type of value of the expression
    using this_type  = basic_conv_expr<T, D, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = cufft_enabled || cudnn_enabled; ///< Indicates if the expression runs on GPU

    /*!
     * \brief The result type for given sub types
     * \tparam A The left hand side epxpression type
     * \tparam B The right hand side epxpression type
     */
    template <typename A, typename B>
    using result_type = detail::expr_result_t<this_type, T, A, B>;

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
            make_temporary(a),
            make_temporary(b),
            c);
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
    template <typename A, typename B, size_t DD>
    static constexpr size_t dim() {
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
    static size_t dim(const A& a, const B& b, size_t d) {
        return Impl::dim(d, a, b);
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The left hand side
     * \param b The right hand side
     * \return the size of the expression
     */
    template <typename A, typename B>
    static size_t size(const A& a, const B& b) {
        size_t acc = 1;
        for (size_t i = 0; i < dimensions(); ++i) {
            acc *= this_type::dim(a, b, i);
        }
        return acc;
    }

    /*!
     * \brief Returns the multiplicative sum of the dimensions at the given indices
     * \return the multiplicative sum of the dimensions at the given indices
     */
    template <typename A, typename B, size_t... I>
    static constexpr size_t size_mul(const std::index_sequence<I...>& /*seq*/) {
        return mul_all<this_type::dim<A, B, I>()...>::value;
    }

    /*!
     * \brief Returns the size of the expression
     * \return the size of the expression
     */
    template <typename A, typename B>
    static constexpr size_t size() {
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
    static constexpr size_t dimensions() {
        return D;
    }
};

/*!
 * \brief A basic configurable dynamic convolution expr
 * \tparam T The value type
 * \tparam D The dimensions of convolution
 * \tparam Impl The implementation class
 */
template <typename T, size_t D, typename Impl>
struct dyn_basic_conv_expr : dyn_impl_expr<dyn_basic_conv_expr<T, D, Impl>, T> {
    static_assert(D > 0, "0D convolution is not valid");

    using value_type = T;                               ///< The type of value of the expression
    using this_type  = dyn_basic_conv_expr<T, D, Impl>; ///< The type of this expression

    static constexpr bool is_gpu = cufft_enabled || cudnn_enabled; ///< Indicates if the expression runs on GPU

    Impl impl; ///< The implementation operator

    // Necessary to avoid ambiguity with the forwarding constructor
    dyn_basic_conv_expr(const dyn_basic_conv_expr& rhs) = default;
    dyn_basic_conv_expr(dyn_basic_conv_expr& rhs) = default;
    dyn_basic_conv_expr(dyn_basic_conv_expr&& rhs) noexcept = default;

    /*!
     * \brief Construct a new dyn_basic_conv_expr and forward all
     * the arguments to the implementation
     * \param args The arguments to forward
     */
    template<typename... Args>
    explicit dyn_basic_conv_expr(Args&&... args) : impl(std::forward<Args>(args)...){
        //Nothing else to init
    }

    /*!
     * \brief Apply the expression
     * \param a The left hand side
     * \param b The right hand side
     * \param c The expression where to store the results
     */
    template <typename A, typename B, typename C>
    void apply(const A& a, const B& b, C& c) const {
        static_assert(all_etl_expr<A, B, C>::value, "Convolution only supported for ETL expressions");

        impl.check(a, b, c);

        impl.apply(
            make_temporary(a),
            make_temporary(b),
            c);
    }

    /*!
     * \brief Returns a textual representation of the operation
     * \return a textual representation of the operation
     */
    static constexpr const char* desc() noexcept {
        return Impl::desc();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param a The left hand side
     * \param b The right hand side
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    template <typename A, typename B>
    size_t dim(const A& a, const B& b, size_t d) const noexcept {
        return impl.dim(d, a, b);
    }

    /*!
     * \brief Returns the size of the expression
     * \param a The left hand side
     * \param b The right hand side
     * \return the size of the expression
     */
    template <typename A, typename B>
    size_t size(const A& a, const B& b) const {
        size_t acc = 1;
        for (size_t i = 0; i < dimensions(); ++i) {
            acc *= this_type::dim(a, b, i);
        }
        return acc;
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
    static constexpr size_t dimensions() {
        return D;
    }
};

//2D convolutions

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv4_valid_expr = basic_conv_expr<T, 4, detail::conv4_valid_impl<S1, S2, P1, P2>>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv4_valid_flipped_expr = basic_conv_expr<T, 4, detail::conv4_valid_flipped_impl<S1, S2, P1, P2>>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T>
using dyn_conv4_valid_expr = dyn_basic_conv_expr<T, 4, detail::dyn_conv4_valid_impl>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T>
using dyn_conv4_valid_flipped_expr = dyn_basic_conv_expr<T, 4, detail::dyn_conv4_valid_flipped_impl>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv4_valid_back_expr = basic_conv_expr<T, 4, detail::conv4_valid_back_impl<S1, S2, P1, P2>>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv4_valid_back_flipped_expr = basic_conv_expr<T, 4, detail::conv4_valid_back_flipped_impl<S1, S2, P1, P2>>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T>
using dyn_conv4_valid_back_expr = dyn_basic_conv_expr<T, 4, detail::dyn_conv4_valid_back_impl>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T>
using dyn_conv4_valid_back_flipped_expr = dyn_basic_conv_expr<T, 4, detail::dyn_conv4_valid_back_flipped_impl>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv4_valid_filter_expr = basic_conv_expr<T, 4, detail::conv4_valid_filter_impl<S1, S2, P1, P2>>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv4_valid_filter_flipped_expr = basic_conv_expr<T, 4, detail::conv4_valid_filter_flipped_impl<S1, S2, P1, P2>>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T>
using dyn_conv4_valid_filter_expr = dyn_basic_conv_expr<T, 4, detail::dyn_conv4_valid_filter_impl>;

/*!
 * \brief Expression for 4D valid convolution
 */
template<typename T>
using dyn_conv4_valid_filter_flipped_expr = dyn_basic_conv_expr<T, 4, detail::dyn_conv4_valid_filter_flipped_impl>;

/*!
 * \brief Expression for 4D valid convolution
 */
template <typename T>
using conv4_full_expr = basic_conv_expr<T, 4, detail::conv4_full_impl>;

/*!
 * \brief Expression for 4D valid convolution
 */
template <typename T>
using conv4_full_flipped_expr = basic_conv_expr<T, 4, detail::conv4_full_flipped_impl>;

/*!
 * \brief Expression for 2D valid convolution, with multiple kernels
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv2_valid_multi_expr = basic_conv_expr<T, 3, detail::conv2_valid_multi_impl<S1,S2,P1,P2>>;

/*!
 * \brief Expression for 2D valid convolution, with multiple flipped kernels
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv2_valid_multi_flipped_expr = basic_conv_expr<T, 3, detail::conv2_valid_multi_flipped_impl<S1,S2,P1,P2>>;

/*!
 * \brief Expression for 2D valid convolution
 */
template<typename T>
using dyn_conv2_valid_multi_expr = dyn_basic_conv_expr<T, 3, detail::dyn_conv2_valid_multi_impl>;

/*!
 * \brief Expression for 2D valid convolution
 */
template<typename T>
using dyn_conv2_valid_multi_flipped_expr = dyn_basic_conv_expr<T, 3, detail::dyn_conv2_valid_multi_flipped_impl>;

/*!
 * \brief Expression for 2D valid convolution, with multiple kernels
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv2_valid_multi_multi_expr = basic_conv_expr<T, 4, detail::conv2_valid_multi_multi_impl<S1,S2,P1,P2>>;

/*!
 * \brief Expression for 2D valid convolution, with multiple flipped kernels
 */
template<typename T, size_t S1 = 0, size_t S2 = 0, size_t P1 = 0, size_t P2 = 0>
using conv2_valid_multi_multi_flipped_expr = basic_conv_expr<T, 4, detail::conv2_valid_multi_multi_flipped_impl<S1,S2,P1,P2>>;

/*!
 * \brief Expression for 2D valid convolution
 */
template<typename T>
using dyn_conv2_valid_multi_multi_expr = dyn_basic_conv_expr<T, 3, detail::dyn_conv2_valid_multi_multi_impl>;

/*!
 * \brief Expression for 2D valid convolution
 */
template<typename T>
using dyn_conv2_valid_multi_multi_flipped_expr = dyn_basic_conv_expr<T, 3, detail::dyn_conv2_valid_multi_multi_flipped_impl>;

/*!
 * \brief Expression for 2D same convolution, with multiple kernels
 */
template <typename T>
using conv2_same_multi_expr = basic_conv_expr<T, 3, detail::conv2_same_multi_impl>;

/*!
 * \brief Expression for 2D same convolution, with multiple flipped kernels
 */
template <typename T>
using conv2_same_multi_flipped_expr = basic_conv_expr<T, 3, detail::conv2_same_multi_flipped_impl>;

//>2D convolutions

/*!
 * \brief Expression for >2D valid convolution
 */
template <typename T, size_t D>
using conv_deep_valid_expr = basic_conv_expr<T, D, detail::conv_deep_valid_impl>;

/*!
 * \brief Expression for >2D same convolution
 */
template <typename T, size_t D>
using conv_deep_same_expr = basic_conv_expr<T, D, detail::conv_deep_same_impl>;

/*!
 * \brief Expression for >2D full convolution
 */
template <typename T, size_t D>
using conv_deep_full_expr = basic_conv_expr<T, D, detail::conv_deep_full_impl>;

} //end of namespace etl
