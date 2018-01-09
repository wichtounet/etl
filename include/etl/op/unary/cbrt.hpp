//=======================================================================
// Copyright (c) 2014-2018 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

#include "etl/impl/egblas/cbrt.hpp"

namespace etl {

/*!
 * \brief Unary operation taking the cubic root value
 * \tparam T The type of value
 */
template <typename T>
struct cbrt_unary_op {
    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable =
               (is_single_precision_t<T> && impl::egblas::has_scbrt)
            || (is_double_precision_t<T> && impl::egblas::has_dcbrt)
            || (is_complex_single_t<T> && impl::egblas::has_ccbrt)
            || (is_complex_double_t<T> && impl::egblas::has_zcbrt);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return std::cbrt(x);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    static auto gpu_compute_hint(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1.0);
        impl::egblas::cbrt(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1);

        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        decltype(auto) t1 = select_smart_gpu_compute(x, y);

        T alpha(1.0);
        impl::egblas::cbrt(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "cbrt";
    }
};

/*!
 * \copydoc cbrt_unary_op
 */
template <typename TT>
struct cbrt_unary_op <std::complex<TT>>{
    using T = std::complex<TT>; ///< The real type

    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable =
               (is_single_precision_t<T> && impl::egblas::has_scbrt)
            || (is_double_precision_t<T> && impl::egblas::has_dcbrt)
            || (is_complex_single_t<T> && impl::egblas::has_ccbrt)
            || (is_complex_double_t<T> && impl::egblas::has_zcbrt);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& z) {
        auto z_abs = std::abs(z);
        auto z_arg = std::arg(z);

        auto new_abs = std::cbrt(z_abs);
        auto new_arg = z_arg / 3.0f;

        return {new_abs * std::cos(new_arg), new_abs * std::sin(new_arg)};
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    static auto gpu_compute_hint(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1.0);
        impl::egblas::cbrt(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1);

        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        decltype(auto) t1 = select_smart_gpu_compute(x, y);

        T alpha(1.0);
        impl::egblas::cbrt(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "cbrt";
    }
};

/*!
 * \copydoc cbrt_unary_op
 */
template <typename TT>
struct cbrt_unary_op <etl::complex<TT>>{
    using T = etl::complex<TT>; ///< The real type

    static constexpr bool linear      = true; ///< Indicates if the operator is linear
    static constexpr bool thread_safe = true; ///< Indicates if the operator is thread safe or not

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = false;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename E>
    static constexpr bool gpu_computable =
               (is_single_precision_t<T> && impl::egblas::has_scbrt)
            || (is_double_precision_t<T> && impl::egblas::has_dcbrt)
            || (is_complex_single_t<T> && impl::egblas::has_ccbrt)
            || (is_complex_double_t<T> && impl::egblas::has_zcbrt);

    /*!
     * \brief Apply the unary operator on x
     * \param x The value on which to apply the operator
     * \return The result of applying the unary operator on x
     */
    static constexpr T apply(const T& x) {
        return etl::cbrt(x);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     *
     * \return The result of applying the unary operator on x. The result must be a GPU computed expression.
     */
    template <typename X, typename Y>
    static auto gpu_compute_hint(const X& x, Y& y) noexcept {
        decltype(auto) t1 = smart_gpu_compute_hint(x, y);

        auto t2 = force_temporary_gpu_dim_only(t1);

        T alpha(1.0);
        impl::egblas::cbrt(etl::size(y), alpha, t1.gpu_memory(), 1, t2.gpu_memory(), 1);

        return t2;
    }
    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param x The expression of the unary operation
     * \param y The expression into which to store the reuslt
     */
    template <typename X, typename Y>
    static Y& gpu_compute(const X& x, Y& y) noexcept {
        decltype(auto) t1 = select_smart_gpu_compute(x, y);

        T alpha(1.0);
        impl::egblas::cbrt(etl::size(y), alpha, t1.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "cbrt";
    }
};

} //end of namespace etl
