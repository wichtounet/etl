//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once

namespace etl {

template <typename T>
struct mul_binary_op;

// detect x / (1.0 * y)

template <typename L, typename R>
struct is_axdy_right_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename L>
struct is_axdy_right_left_impl <L, binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, RightExpr>> {
    static constexpr bool value = true;
};

// detect x / (y * 1.0)

template <typename L, typename R>
struct is_axdy_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename L>
struct is_axdy_right_right_impl <L, binary_expr<T0, RightExpr, etl::mul_binary_op<T2>, etl::scalar<T1>>> {
    static constexpr bool value = true;
};

// detect (1.0 * x) / y

template <typename L, typename R>
struct is_axdy_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename R>
struct is_axdy_left_left_impl <binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, RightExpr>, R> {
    static constexpr bool value = true;
};

// detect (1.0 * x) / y

template <typename L, typename R>
struct is_axdy_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename R>
struct is_axdy_left_right_impl <binary_expr<T0, RightExpr, etl::mul_binary_op<T2>, etl::scalar<T1>>, R> {
    static constexpr bool value = true;
};

// detect x / (1.0 + y)

template <typename L, typename R>
struct is_axdbpy_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename L>
struct is_axdbpy_left_impl <L, binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, RightExpr>> {
    static constexpr bool value = true;
};

// detect x / (y + 1.0)

template <typename L, typename R>
struct is_axdbpy_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename L>
struct is_axdbpy_right_impl <L, binary_expr<T0, RightExpr, etl::plus_binary_op<T2>, etl::scalar<T1>>> {
    static constexpr bool value = true;
};

// detect (1.0 * x) / (1.0 + y)

template <typename L, typename R>
struct is_axdbpy_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_axdbpy_left_left_impl <binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, R1>, binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (1.0 * x) / (y + 1.0)

template <typename L, typename R>
struct is_axdbpy_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_axdbpy_left_right_impl <binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, R1>, binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (x * 1.0) / (1.0 + y)

template <typename L, typename R>
struct is_axdbpy_right_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_axdbpy_right_left_impl <binary_expr<T0, R1, etl::mul_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (x * 1.0) / (y + 1.0)

template <typename L, typename R>
struct is_axdbpy_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_axdbpy_right_right_impl <binary_expr<T0, R1, etl::mul_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / (1.0 + y)

template <typename L, typename R>
struct is_apxdbpy_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdbpy_left_left_impl <binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, R1>, binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / (y + 1.0)

template <typename L, typename R>
struct is_apxdbpy_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdbpy_left_right_impl <binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, R1>, binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / (1.0 + y)

template <typename L, typename R>
struct is_apxdbpy_right_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdbpy_right_left_impl <binary_expr<T0, R1, etl::plus_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / (y + 1.0)

template <typename L, typename R>
struct is_apxdbpy_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdbpy_right_right_impl <binary_expr<T0, R1, etl::plus_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / y

template <typename L, typename R>
struct is_apxdby_left_impl {
    static constexpr bool value = false;
};

template <typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_left_impl <binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>, R1> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / y

template <typename L, typename R>
struct is_apxdby_right_impl {
    static constexpr bool value = false;
};

template <typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_right_impl <binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>, R1> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / (1.0 * y)

template <typename L, typename R>
struct is_apxdby_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_left_left_impl <binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, R1>, binary_expr<T3, etl::scalar<T4>, etl::mul_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / (y * 1.0)

template <typename L, typename R>
struct is_apxdby_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_left_right_impl <binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, R1>, binary_expr<T3, R2, etl::mul_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / (1.0 * y)

template <typename L, typename R>
struct is_apxdby_right_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_right_left_impl <binary_expr<T0, R1, etl::plus_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, etl::scalar<T4>, etl::mul_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / (y * 1.0)

template <typename L, typename R>
struct is_apxdby_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_right_right_impl <binary_expr<T0, R1, etl::plus_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, R2, etl::mul_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// Variable templates helper

template <typename L, typename R>
static constexpr bool is_apxdbpy_left_left = is_apxdbpy_left_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_apxdbpy_left_right = is_apxdbpy_left_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_apxdbpy_right_left = is_apxdbpy_right_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_apxdbpy_right_right = is_apxdbpy_right_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_apxdbpy = is_apxdbpy_left_left<L, R> || is_apxdbpy_left_right<L, R> || is_apxdbpy_right_left<L, R> || is_apxdbpy_right_right<L, R>;

template <typename L, typename R>
static constexpr bool is_apxdby_left_left = is_apxdby_left_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_apxdby_left_right = is_apxdby_left_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_apxdby_right_left = is_apxdby_right_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_apxdby_right_right = is_apxdby_right_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_apxdby_left = is_apxdby_left_impl<L, R>::value && !is_apxdbpy<L, R> && !is_apxdby_left_left<L, R> && !is_apxdby_left_right<L, R> && !is_apxdby_right_left<L, R> && !is_apxdby_right_right<L, R>;

template <typename L, typename R>
static constexpr bool is_apxdby_right = is_apxdby_right_impl<L, R>::value && !is_apxdbpy<L, R> && !is_apxdby_left_left<L, R> && !is_apxdby_left_right<L, R> && !is_apxdby_right_left<L, R> && !is_apxdby_right_right<L, R>;

template <typename L, typename R>
static constexpr bool is_apxdby = is_apxdby_left<L, R> || is_apxdby_right<L, R> || is_apxdby_left_left<L, R> || is_apxdby_left_right<L, R> || is_apxdby_right_left<L, R> || is_apxdby_right_right<L, R>;

template <typename L, typename R>
static constexpr bool is_axdbpy_left_left = is_axdbpy_left_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axdbpy_left_right = is_axdbpy_left_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axdbpy_right_left = is_axdbpy_right_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axdbpy_right_right = is_axdbpy_right_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axdbpy_left = is_axdbpy_left_impl<L, R>::value && !is_axdbpy_left_left<L, R> && !is_axdbpy_right_left<L, R> && !is_apxdbpy<L, R>;

template <typename L, typename R>
static constexpr bool is_axdbpy_right = is_axdbpy_right_impl<L, R>::value && !is_axdbpy_left_right<L, R> && !is_axdbpy_right_right<L, R> && !is_apxdbpy<L, R>;

template <typename L, typename R>
static constexpr bool is_axdbpy = is_axdbpy_left<L, R> || is_axdbpy_right<L, R> || is_axdbpy_left_left<L, R> || is_axdbpy_left_right<L, R> || is_axdbpy_right_left<L, R> || is_axdbpy_right_right<L, R>;

template <typename L, typename R>
static constexpr bool is_axdy_right_left = is_axdy_right_left_impl<L, R>::value && !is_axdbpy<L, R> && !is_apxdby<L, R>;

template <typename L, typename R>
static constexpr bool is_axdy_right_right = is_axdy_right_right_impl<L, R>::value && !is_axdbpy<L, R> && !is_apxdby<L, R>;

template <typename L, typename R>
static constexpr bool is_axdy_left_left = is_axdy_left_left_impl<L, R>::value && !is_axdbpy<L, R> && !is_apxdby<L, R>;

template <typename L, typename R>
static constexpr bool is_axdy_left_right = is_axdy_left_right_impl<L, R>::value && !is_axdbpy<L, R> && !is_apxdby<L, R>;

template <typename L, typename R>
static constexpr bool is_axdy = is_axdy_right_left<L, R> || is_axdy_right_right<L, R> || is_axdy_left_left<L, R> || is_axdy_left_right<L, R>;

template <typename L, typename R>
static constexpr bool is_special = is_axdy<L, R> || is_axdbpy<L, R> || is_apxdbpy<L, R> || is_apxdby<L, R>;

/*!
 * \brief Binary operator for scalar division
 */
template <typename T>
struct div_binary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the given vector mode
     * \tparam V The vector mode
     *
     * Note: Integer division is not yet supported
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = is_floating_t<T> || (is_complex_t<T> && V != vector_mode_t::AVX512);

    /*!
     * \brief Indicates if the operator can be computd on GPU
     */
    template<typename L, typename R>
    static constexpr bool gpu_computable =
            (
                    (!is_scalar<L> && !is_scalar<R>)
                &&  egblas_enabled
                &&  (
                        (is_single_precision_t<T> && impl::egblas::has_saxdy)
                    ||  (is_double_precision_t<T> && impl::egblas::has_daxdy)
                    ||  (is_complex_single_t<T> && impl::egblas::has_caxdy)
                    ||  (is_complex_double_t<T> && impl::egblas::has_zaxdy))
            )
        ||  (
                    (!is_scalar<L> && is_scalar<R>)
                &&  cublas_enabled
            )
        ||  (
                    (is_scalar<L> && !is_scalar<R>)
                && (
                        (is_single_precision_t<T> && impl::egblas::has_scalar_sdiv)
                    ||  (is_double_precision_t<T> && impl::egblas::has_scalar_ddiv)
                    ||  (is_complex_single_t<T> && impl::egblas::has_scalar_cdiv)
                    ||  (is_complex_double_t<T> && impl::egblas::has_scalar_zdiv))
            )
            ;

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type       = typename V::template vec_type<T>;

    /*!
     * \brief Apply the unary operator on lhs and rhs
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     * \return The result of applying the binary operator on lhs and rhs
     */
    static constexpr T apply(const T& lhs, const T& rhs) noexcept {
        return lhs / rhs;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static vec_type<V> load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        return V::div(lhs, rhs);
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y>
    static auto gpu_compute_hint(const L& lhs, const R& rhs, Y& y) noexcept {
        cpp_unused(y);

        auto t3 = force_temporary_gpu_dim_only(y);
        gpu_compute(lhs, rhs, t3);
        return t3;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdy_right_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::axdy_3(etl::size(y), rhs_lhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdy_right_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::axdy_3(etl::size(y), rhs_rhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdy_left_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::axdy_3(etl::size(y), T(1) / lhs_lhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdy_left_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::axdy_3(etl::size(y), T(1) / lhs_rhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdbpy_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::axdbpy_3(etl::size(y), T(1), x.gpu_memory(), 1, rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdbpy_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::axdbpy_3(etl::size(y), T(1), x.gpu_memory(), 1, rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdbpy_left_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::axdbpy_3(etl::size(y), lhs_lhs.value, x.gpu_memory(), 1, rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdbpy_left_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::axdbpy_3(etl::size(y), lhs_lhs.value, x.gpu_memory(), 1, rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdbpy_right_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::axdbpy_3(etl::size(y), lhs_rhs.value, x.gpu_memory(), 1, rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axdbpy_right_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::axdbpy_3(etl::size(y), lhs_rhs.value, x.gpu_memory(), 1, rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdbpy_left_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::apxdbpy_3(etl::size(y), lhs_lhs.value, x.gpu_memory(), 1, rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdbpy_left_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::apxdbpy_3(etl::size(y), lhs_lhs.value, x.gpu_memory(), 1, rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdbpy_right_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::apxdbpy_3(etl::size(y), lhs_rhs.value, x.gpu_memory(), 1, rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdbpy_right_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::apxdbpy_3(etl::size(y), lhs_rhs.value, x.gpu_memory(), 1, rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdby_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::apxdby_3(etl::size(y), lhs_lhs.value, x.gpu_memory(), 1, T(1), y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdby_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::apxdby_3(etl::size(y), lhs_rhs.value, x.gpu_memory(), 1, T(1), y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdby_left_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::apxdby_3(etl::size(y), lhs_lhs.value, x.gpu_memory(), 1, rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdby_left_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::apxdby_3(etl::size(y), lhs_lhs.value, x.gpu_memory(), 1, rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdby_right_left<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::apxdby_3(etl::size(y), lhs_rhs.value, x.gpu_memory(), 1, rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_apxdby_right_right<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::apxdby_3(etl::size(y), lhs_rhs.value, x.gpu_memory(), 1, rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(!is_scalar<L> && !is_scalar<R> && !is_special<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& y) noexcept {
        smart_gpu_compute(lhs, y);

        decltype(auto) t2 = smart_gpu_compute_hint(rhs, y);

        value_t<L> alpha(1);

        impl::egblas::axdy(etl::size(y), alpha, t2.gpu_memory(), 1, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(!is_scalar<L> && is_scalar<R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& y) noexcept {
        auto s = T(1) / rhs.value;

        smart_gpu_compute(lhs, y);

        impl::egblas::scalar_mul(etl::size(y), s, y.gpu_memory(), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Compute the result of the operation using the GPU
     *
     * \param lhs The left hand side value on which to apply the operator
     * \param rhs The right hand side value on which to apply the operator
     *
     * \return The result of applying the binary operator on lhs and rhs. The result must be a GPU computed expression.
     */
    template <typename L, typename R, typename Y, cpp_enable_iff(is_scalar<L> && !is_scalar<R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& y) noexcept {
        auto s = lhs.value;

        smart_gpu_compute(rhs, y);

        impl::egblas::scalar_div(s, y.gpu_memory(), etl::size(y), 1);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "/";
    }
};

} //end of namespace etl
