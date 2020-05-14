//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
template <typename A, typename T, typename Impl>
struct fft_expr : base_temporary_expr_un<fft_expr<A, T, Impl>, A> {
    using value_type = T;                                    ///< The type of value of the expression
    using this_type  = fft_expr<A, T, Impl>;                 ///< The type of this expression
    using base_type  = base_temporary_expr_un<this_type, A>; ///< The base type
    using sub_traits = decay_traits<A>;                      ///< The traits of the sub type

    static constexpr auto storage_order = sub_traits::storage_order; ///< The sub storage order

    /*!
     * \brief Indicates if the temporary expression can be directly evaluated
     * using only GPU.
     */
    static constexpr bool gpu_computable = Impl::template gpu_computable<A>;

    /*!
     * \brief Construct a new expression
     * \param a The sub expression
     */
    explicit fft_expr(A a) : base_type(a) {
        //Nothing else to init
    }

    // Assignment functions

    /*!
     * \brief Assign to a matrix of the same storage order
     * \param c The expression to which assign
     */
    template <typename C>
    void assign_to(C&& c) const {
        static_assert(all_etl_expr<A, C>, "max_pool_2d only supported for ETL expressions");
        static_assert(etl::dimensions<A>() == etl::dimensions<C>(), "max_pool_2d must be applied on matrices of same dimensionality");

        Impl::apply(this->a(), c);
    }

    /*!
     * \brief Add to the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_add_to(L&& lhs) const {
        std_add_evaluate(*this, lhs);
    }

    /*!
     * \brief Sub from the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_sub_to(L&& lhs) const {
        std_sub_evaluate(*this, lhs);
    }

    /*!
     * \brief Multiply the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mul_to(L&& lhs) const {
        std_mul_evaluate(*this, lhs);
    }

    /*!
     * \brief Divide the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_div_to(L&& lhs) const {
        std_div_evaluate(*this, lhs);
    }

    /*!
     * \brief Modulo the given left-hand-side expression
     * \param lhs The expression to which assign
     */
    template <typename L>
    void assign_mod_to(L&& lhs) const {
        std_mod_evaluate(*this, lhs);
    }

    /*!
     * \brief Print a representation of the expression on the given stream
     * \param os The output stream
     * \param expr The expression to print
     * \return the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const fft_expr& expr) {
        return os << "fft(" << expr._a << ")";
    }
};

/*!
 * \brief Traits for a transpose expression
 * \tparam A The transposed sub type
 */
template <typename A, typename T, typename Impl>
struct etl_traits<etl::fft_expr<A, T, Impl>> {
    using expr_t     = etl::fft_expr<A, T, Impl>; ///< The expression type
    using sub_expr_t = std::decay_t<A>;           ///< The sub expression type
    using sub_traits = etl_traits<sub_expr_t>;    ///< The sub traits
    using value_type = T;                         ///< The value type of the expression

    static constexpr bool is_etl         = true;                                 ///< Indicates if the type is an ETL expression
    static constexpr bool is_transformer = false;                                ///< Indicates if the type is a transformer
    static constexpr bool is_view        = false;                                ///< Indicates if the type is a view
    static constexpr bool is_magic_view  = false;                                ///< Indicates if the type is a magic view
    static constexpr bool is_fast        = sub_traits::is_fast;                  ///< Indicates if the expression is fast
    static constexpr bool is_linear      = false;                                ///< Indicates if the expression is linear
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
        return decay_traits<A>::template dim<DD>();
    }

    /*!
     * \brief Returns the dth dimension of the expression
     * \param e The sub expression
     * \param d The dimension to get
     * \return the dth dimension of the expression
     */
    static size_t dim(const expr_t& e, size_t d) {
        return etl::dim(e._a, d);
    }

    /*!
     * \brief Returns the size of the expression
     * \param e The sub expression
     * \return the size of the expression
     */
    static size_t size(const expr_t& e) {
        return sub_traits::size(e._a);
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
};

//Helpers to compute the type of the result

namespace detail {

/*!
 * \brief The output value type of an FFT based on the input
 */
template <typename A>
using fft_value_type = std::conditional_t<is_complex<A>, value_t<A>, std::complex<value_t<A>>>;

/*!
 * \brief The output value type of an Inverse FFT based on the input
 */
template <typename A>
using ifft_value_type = std::conditional_t<is_complex<A>, value_t<A>, std::complex<value_t<A>>>;

/*!
 * \brief The output value type of an Inverse FFT real based on the input
 */
template <typename A>
using ifft_real_value_type = std::conditional_t<is_complex<A>, typename value_t<A>::value_type, value_t<A>>;

} //end of namespace detail

/*!
 * \brief Creates an expression representing the 1D Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 1D FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::fft_value_type<A>, detail::fft1_impl> fft_1d(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");

    return fft_expr<detail::build_type<A>, detail::fft_value_type<A>, detail::fft1_impl>{a};
}

/*!
 * \brief Creates an expression representing the 1D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the 1D FFT of a
 */
template <typename A, typename C>
auto fft_1d(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = fft_1d(a);
    return c;
}

/*!
 * \brief Creates an expression representing the 1D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 1D inverse FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::ifft_value_type<A>, detail::ifft1_impl> ifft_1d(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");

    return fft_expr<detail::build_type<A>, detail::ifft_value_type<A>, detail::ifft1_impl>{a};
}

/*!
 * \brief Creates an expression representing the 1D inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the 1D inverse FFT of a
 */
template <typename A, typename C>
auto ifft_1d(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = ifft_1d(a);
    return c;
}

/*!
 * \brief Creates an expression representing the real part of the 1D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the real part of the 1D inverse FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::ifft_real_value_type<A>, detail::ifft1_real_impl> ifft_1d_real(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");

    return fft_expr<detail::build_type<A>, detail::ifft_real_value_type<A>, detail::ifft1_real_impl>{a};
}

/*!
 * \brief Creates an expression representing the real part of the 1D inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the real part of the 1D inverse FFT of a
 */
template <typename A, typename C>
auto ifft_1d_real(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = ifft_1d_real(a);
    return c;
}

/*!
 * \brief Creates an expression representing the 2D Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 2D FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::fft_value_type<A>, detail::fft2_impl> fft_2d(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");

    return fft_expr<detail::build_type<A>, detail::fft_value_type<A>, detail::fft2_impl>{a};
}

/*!
 * \brief Creates an expression representing the 2D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the 2D FFT of a
 */
template <typename A, typename C>
auto fft_2d(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = fft_2d(a);
    return c;
}

/*!
 * \brief Creates an expression representing the 2D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the 2D inverse FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::ifft_value_type<A>, detail::ifft2_impl> ifft_2d(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");

    return fft_expr<detail::build_type<A>, detail::ifft_value_type<A>, detail::ifft2_impl>{a};
}

/*!
 * \brief Creates an expression representing the 2D inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the 2D inverse FFT of a
 */
template <typename A, typename C>
auto ifft_2d(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = ifft_2d(a);
    return c;
}

/*!
 * \brief Creates an expression representing the real part of the 2D inverse Fast-Fourrier-Transform of the given expression
 * \param a The input expression
 * \return an expression representing the real part of the 2D inverse FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::ifft_real_value_type<A>, detail::ifft2_real_impl> ifft_2d_real(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");

    return fft_expr<detail::build_type<A>, detail::ifft_real_value_type<A>, detail::ifft2_real_impl>{a};
}

/*!
 * \brief Creates an expression representing the real part of the 2D inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 * \param a The input expression
 * \param c The result
 * \return an expression representing the real part of the 2D inverse FFT of a
 */
template <typename A, typename C>
auto ifft_2d_real(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    validate_assign(c, a);

    c = ifft_2d_real(a);
    return c;
}

/*!
 * \brief Creates an expression representing several 1D Fast-Fourrier-Transform of the given expression
 *
 * Only the last dimension is used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \return an expression representing several 1D FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::fft_value_type<A>, detail::fft1_many_impl> fft_1d_many(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2, "fft_many requires at least 2D matrices");

    return fft_expr<detail::build_type<A>, detail::fft_value_type<A>, detail::fft1_many_impl>{a};
}

/*!
 * \brief Creates an expression representing several 1D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 *
 * Only the last dimension is used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param c The result
 * \return an expression representing several 1D FFT of a
 */
template <typename A, typename C>
auto fft_1d_many(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2 && decay_traits<C>::dimensions() >= 2, "fft_many requires at least 2D matrices");
    validate_assign(c, a);

    c = fft_1d_many(a);
    return c;
}

/*!
 * \brief Creates an expression representing several 1D Inverse Fast-Fourrier-Transform of the given expression
 *
 * Only the last dimension is used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \return an expression representing several 1D FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::ifft_value_type<A>, detail::ifft1_many_impl> ifft_1d_many(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2, "ifft_many requires at least 2D matrices");

    return fft_expr<detail::build_type<A>, detail::ifft_value_type<A>, detail::ifft1_many_impl>{a};
}

/*!
 * \brief Creates an expression representing several 1D Inverse Fast-Fourrier-Transform of the given expression, the result will be stored in c
 *
 * Only the last dimension is used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param c The result
 * \return an expression representing several 1D FFT of a
 */
template <typename A, typename C>
auto ifft_1d_many(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 2 && decay_traits<C>::dimensions() >= 2, "ifft_many requires at least 2D matrices");
    validate_assign(c, a);

    c = ifft_1d_many(a);
    return c;
}

/*!
 * \brief Creates an expression representing several 2D Fast-Fourrier-Transform of the given expression
 *
 * Only the last two dimensions are used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \return an expression representing several 2D FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::fft_value_type<A>, detail::fft2_many_impl> fft_2d_many(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3, "fft_many requires at least 3D matrices");

    return fft_expr<detail::build_type<A>, detail::fft_value_type<A>, detail::fft2_many_impl>{a};
}

/*!
 * \brief Creates an expression representing several 2D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 *
 * Only the last two dimensions are used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param c The result
 * \return an expression representing several 2D FFT of a
 */
template <typename A, typename C>
auto fft_2d_many(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3 && decay_traits<C>::dimensions() >= 3, "fft_many requires at least 3D matrices");
    validate_assign(c, a);

    c = fft_2d_many(a);
    return c;
}

/*!
 * \brief Creates an expression representing several 2D Fast-Fourrier-Transform of the given expression
 *
 * Only the last two dimensions are used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \return an expression representing several 2D FFT of a
 */
template <typename A>
fft_expr<detail::build_type<A>, detail::ifft_value_type<A>, detail::ifft2_many_impl> ifft_2d_many(A&& a) {
    static_assert(is_etl_expr<A>, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3, "ifft_many requires at least 3D matrices");

    return fft_expr<detail::build_type<A>, detail::ifft_value_type<A>, detail::ifft2_many_impl>{a};
}

/*!
 * \brief Creates an expression representing several 2D Fast-Fourrier-Transform of the given expression, the result will be stored in c
 *
 * Only the last two dimensions are used for the FFT itself, the first dimensions are used as containers to perform multiple FFT.
 *
 * \param a The input expression
 * \param c The result
 * \return an expression representing several 2D FFT of a
 */
template <typename A, typename C>
auto ifft_2d_many(A&& a, C&& c) {
    static_assert(all_etl_expr<A, C>, "FFT only supported for ETL expressions");
    static_assert(decay_traits<A>::dimensions() >= 3 && decay_traits<C>::dimensions() >= 3, "ifft_many requires at least 3D matrices");
    validate_assign(c, a);

    c = ifft_2d_many(a);
    return c;
}

} //end of namespace etl
