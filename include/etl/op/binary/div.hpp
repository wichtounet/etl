//=======================================================================
// Copyright (c) 2014-2020 Baptiste Wicht
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
struct is_axdy_right_left_impl<L, binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, RightExpr>> {
    static constexpr bool value = true;
};

// detect x / (y * 1.0)

template <typename L, typename R>
struct is_axdy_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename L>
struct is_axdy_right_right_impl<L, binary_expr<T0, RightExpr, etl::mul_binary_op<T2>, etl::scalar<T1>>> {
    static constexpr bool value = true;
};

// detect (1.0 * x) / y

template <typename L, typename R>
struct is_axdy_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename R>
struct is_axdy_left_left_impl<binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, RightExpr>, R> {
    static constexpr bool value = true;
};

// detect (1.0 * x) / y

template <typename L, typename R>
struct is_axdy_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename R>
struct is_axdy_left_right_impl<binary_expr<T0, RightExpr, etl::mul_binary_op<T2>, etl::scalar<T1>>, R> {
    static constexpr bool value = true;
};

// detect x / (1.0 + y)

template <typename L, typename R>
struct is_axdbpy_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename L>
struct is_axdbpy_left_impl<L, binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, RightExpr>> {
    static constexpr bool value = true;
};

// detect x / (y + 1.0)

template <typename L, typename R>
struct is_axdbpy_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename L>
struct is_axdbpy_right_impl<L, binary_expr<T0, RightExpr, etl::plus_binary_op<T2>, etl::scalar<T1>>> {
    static constexpr bool value = true;
};

// detect (1.0 * x) / (1.0 + y)

template <typename L, typename R>
struct is_axdbpy_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_axdbpy_left_left_impl<binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, R1>, binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (1.0 * x) / (y + 1.0)

template <typename L, typename R>
struct is_axdbpy_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_axdbpy_left_right_impl<binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, R1>, binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (x * 1.0) / (1.0 + y)

template <typename L, typename R>
struct is_axdbpy_right_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_axdbpy_right_left_impl<binary_expr<T0, R1, etl::mul_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (x * 1.0) / (y + 1.0)

template <typename L, typename R>
struct is_axdbpy_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_axdbpy_right_right_impl<binary_expr<T0, R1, etl::mul_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / (1.0 + y)

template <typename L, typename R>
struct is_apxdbpy_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdbpy_left_left_impl<binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, R1>, binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / (y + 1.0)

template <typename L, typename R>
struct is_apxdbpy_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdbpy_left_right_impl<binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, R1>,
                                  binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / (1.0 + y)

template <typename L, typename R>
struct is_apxdbpy_right_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdbpy_right_left_impl<binary_expr<T0, R1, etl::plus_binary_op<T2>, etl::scalar<T1>>,
                                  binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / (y + 1.0)

template <typename L, typename R>
struct is_apxdbpy_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdbpy_right_right_impl<binary_expr<T0, R1, etl::plus_binary_op<T2>, etl::scalar<T1>>,
                                   binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / y

template <typename L, typename R>
struct is_apxdby_left_impl {
    static constexpr bool value = false;
};

template <typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_left_impl<binary_expr<T3, etl::scalar<T4>, etl::plus_binary_op<T5>, R2>, R1> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / y

template <typename L, typename R>
struct is_apxdby_right_impl {
    static constexpr bool value = false;
};

template <typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_right_impl<binary_expr<T3, R2, etl::plus_binary_op<T5>, etl::scalar<T4>>, R1> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / (1.0 * y)

template <typename L, typename R>
struct is_apxdby_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_left_left_impl<binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, R1>, binary_expr<T3, etl::scalar<T4>, etl::mul_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (1.0 + x) / (y * 1.0)

template <typename L, typename R>
struct is_apxdby_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_left_right_impl<binary_expr<T0, etl::scalar<T1>, etl::plus_binary_op<T2>, R1>, binary_expr<T3, R2, etl::mul_binary_op<T5>, etl::scalar<T4>>> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / (1.0 * y)

template <typename L, typename R>
struct is_apxdby_right_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_right_left_impl<binary_expr<T0, R1, etl::plus_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, etl::scalar<T4>, etl::mul_binary_op<T5>, R2>> {
    static constexpr bool value = true;
};

// detect (x + 1.0) / (y * 1.0)

template <typename L, typename R>
struct is_apxdby_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename R1, typename R2>
struct is_apxdby_right_right_impl<binary_expr<T0, R1, etl::plus_binary_op<T2>, etl::scalar<T1>>, binary_expr<T3, R2, etl::mul_binary_op<T5>, etl::scalar<T4>>> {
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
static constexpr bool is_apxdby_left =
    is_apxdby_left_impl<L, R>::value
    && !is_apxdbpy<L, R> && !is_apxdby_left_left<L, R> && !is_apxdby_left_right<L, R> && !is_apxdby_right_left<L, R> && !is_apxdby_right_right<L, R>;

template <typename L, typename R>
static constexpr bool is_apxdby_right =
    is_apxdby_right_impl<L, R>::value
    && !is_apxdbpy<L, R> && !is_apxdby_left_left<L, R> && !is_apxdby_left_right<L, R> && !is_apxdby_right_left<L, R> && !is_apxdby_right_right<L, R>;

template <typename L, typename R>
static constexpr bool is_apxdby =
    is_apxdby_left<
        L,
        R> || is_apxdby_right<L, R> || is_apxdby_left_left<L, R> || is_apxdby_left_right<L, R> || is_apxdby_right_left<L, R> || is_apxdby_right_right<L, R>;

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
static constexpr bool is_axdbpy =
    is_axdbpy_left<
        L,
        R> || is_axdbpy_right<L, R> || is_axdbpy_left_left<L, R> || is_axdbpy_left_right<L, R> || is_axdbpy_right_left<L, R> || is_axdbpy_right_right<L, R>;

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
static constexpr bool is_special_div = is_axdy<L, R> || is_axdbpy<L, R> || is_apxdbpy<L, R> || is_apxdby<L, R>;

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
    template <typename L, typename R>
    static constexpr bool gpu_computable =
        ((!is_scalar<L> && !is_scalar<R>)&&((is_single_precision_t<T> && impl::egblas::has_saxdy_3 && impl::egblas::has_saxdbpy_3
                                             && impl::egblas::has_sapxdbpy_3 && impl::egblas::has_sapxdby_3)
                                            || (is_double_precision_t<T> && impl::egblas::has_daxdy_3 && impl::egblas::has_daxdbpy_3
                                                && impl::egblas::has_dapxdbpy_3 && impl::egblas::has_dapxdby_3)
                                            || (is_complex_single_t<T> && impl::egblas::has_caxdy_3 && impl::egblas::has_caxdbpy_3
                                                && impl::egblas::has_capxdbpy_3 && impl::egblas::has_capxdby_3)
                                            || (is_complex_double_t<T> && impl::egblas::has_zaxdy_3 && impl::egblas::has_zaxdbpy_3
                                                && impl::egblas::has_zapxdbpy_3 && impl::egblas::has_zapxdby_3)))
        || ((is_scalar<L> != is_scalar<R>)&&((is_single_precision_t<T> && impl::egblas::has_scalar_smul && impl::egblas::has_scalar_sdiv)
                                             || (is_double_precision_t<T> && impl::egblas::has_scalar_dmul && impl::egblas::has_scalar_ddiv)
                                             || (is_complex_single_t<T> && impl::egblas::has_scalar_cmul && impl::egblas::has_scalar_cdiv)
                                             || (is_complex_double_t<T> && impl::egblas::has_scalar_zmul && impl::egblas::has_scalar_zdiv)));

    /*!
     * \brief Estimate the complexity of operator
     * \return An estimation of the complexity of the operator
     */
    static constexpr int complexity() {
        return 1;
    }

    /*!
     * The vectorization type for V
     */
    template <typename V = default_vec>
    using vec_type = typename V::template vec_type<T>;

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
    template <typename L, typename R, typename Y>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        if constexpr (is_axdy_right_left<L, R>) {
            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::axdy_3(etl::size(yy), rhs_lhs.value, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdy_right_right<L, R>) {
            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::axdy_3(etl::size(yy), rhs_rhs.value, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdy_left_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs)>;

            impl::egblas::axdy_3(etl::size(yy), T(1) / lhs_lhs.value, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdy_left_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs)>;

            impl::egblas::axdy_3(etl::size(yy), T(1) / lhs_rhs.value, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdbpy_left<L, R>) {
            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::axdbpy_3(etl::size(yy), T(1), x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdbpy_right<L, R>) {
            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::axdbpy_3(etl::size(yy), T(1), x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdbpy_left_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::axdbpy_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdbpy_left_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::axdbpy_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdbpy_right_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::axdbpy_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axdbpy_right_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::axdbpy_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdbpy_left_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::apxdbpy_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdbpy_left_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::apxdbpy_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdbpy_right_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::apxdbpy_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdbpy_right_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::apxdbpy_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdby_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs)>;

            impl::egblas::apxdby_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, T(1), y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdby_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs)>;

            impl::egblas::apxdby_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, T(1), y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdby_left_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::apxdby_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdby_left_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::apxdby_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdby_right_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::apxdby_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_apxdby_right_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::apxdby_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (!is_scalar<L> && !is_scalar<R> && !is_special_div<L, R>) {
            decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs)>;

            value_t<L> alpha(1);
            impl::egblas::axdy_3(etl::size(yy), alpha, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (!is_scalar<L> && is_scalar<R> && !is_special_div<L, R>) {
            auto s = T(1) / rhs.value;

            smart_gpu_compute(lhs, yy);

            impl::egblas::scalar_mul(yy.gpu_memory(), etl::size(yy), 1, s);
        } else if constexpr (is_scalar<L> && !is_scalar<R> && !is_special_div<L, R>) {
            auto s = lhs.value;

            smart_gpu_compute(rhs, yy);

            impl::egblas::scalar_div(s, yy.gpu_memory(), etl::size(yy), 1);
        }

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
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
