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

// detect 1.0 * x + y

template <typename L, typename R>
struct is_axpy_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename R>
struct is_axpy_left_left_impl<binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, RightExpr>, R> {
    static constexpr bool value = !is_scalar<R>;
};

// detect x * 1.0 + y

template <typename L, typename R>
struct is_axpy_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename LeftExpr, typename R>
struct is_axpy_left_right_impl<binary_expr<T0, LeftExpr, etl::mul_binary_op<T2>, etl::scalar<T1>>, R> {
    static constexpr bool value = !is_scalar<R>;
};

// detect x + 1.0 * y

template <typename L, typename R>
struct is_axpy_right_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename L>
struct is_axpy_right_left_impl<L, binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, RightExpr>> {
    static constexpr bool value = !is_scalar<L> && !is_scalar<RightExpr>;
};

// detect x + y * 1.0

template <typename L, typename R>
struct is_axpy_right_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename LeftExpr, typename L>
struct is_axpy_right_right_impl<L, binary_expr<T0, LeftExpr, etl::mul_binary_op<T2>, etl::scalar<T1>>> {
    static constexpr bool value = !is_scalar<LeftExpr>;
};

// detect 1.0 * x + 1.0 * y

template <typename L, typename R>
struct is_axpby_left_left_impl {
    static constexpr bool value = false;
};

template <typename LT1, typename LT2, typename LT3, typename LRightExpr, typename RT1, typename RT2, typename RT3, typename RRightExpr>
struct is_axpby_left_left_impl<binary_expr<LT1, etl::scalar<LT2>, etl::mul_binary_op<LT3>, LRightExpr>,
                               binary_expr<RT1, etl::scalar<RT2>, etl::mul_binary_op<RT3>, RRightExpr>> {
    static constexpr bool value = true;
};

// detect 1.0 * x + y * 1.0

template <typename L, typename R>
struct is_axpby_left_right_impl {
    static constexpr bool value = false;
};

template <typename LT1, typename LT2, typename LT3, typename LRightExpr, typename RT1, typename RT2, typename RT3, typename RLeftExpr>
struct is_axpby_left_right_impl<binary_expr<LT1, etl::scalar<LT2>, etl::mul_binary_op<LT3>, LRightExpr>,
                                binary_expr<RT1, RLeftExpr, etl::mul_binary_op<RT3>, etl::scalar<RT2>>> {
    static constexpr bool value = true;
};

// detect x * 1.0 + 1.0 * y

template <typename L, typename R>
struct is_axpby_right_left_impl {
    static constexpr bool value = false;
};

template <typename LT1, typename LT2, typename LT3, typename LLeftExpr, typename RT1, typename RT2, typename RT3, typename RRightExpr>
struct is_axpby_right_left_impl<binary_expr<LT1, LLeftExpr, etl::mul_binary_op<LT3>, etl::scalar<LT2>>,
                                binary_expr<RT1, etl::scalar<RT2>, etl::mul_binary_op<RT3>, RRightExpr>> {
    static constexpr bool value = true;
};

// detect x * 1.0 + y * 1.0

template <typename L, typename R>
struct is_axpby_right_right_impl {
    static constexpr bool value = false;
};

template <typename LT1, typename LT2, typename LT3, typename LLeftExpr, typename RT1, typename RT2, typename RT3, typename RLeftExpr>
struct is_axpby_right_right_impl<binary_expr<LT1, LLeftExpr, etl::mul_binary_op<LT3>, etl::scalar<LT2>>,
                                 binary_expr<RT1, RLeftExpr, etl::mul_binary_op<RT3>, etl::scalar<RT2>>> {
    static constexpr bool value = true;
};

// Variable templates helper

template <typename L, typename R>
static constexpr bool is_axpby_left_left = is_axpby_left_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axpby_left_right = is_axpby_left_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axpby_right_left = is_axpby_right_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axpby_right_right = is_axpby_right_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axpby = is_axpby_left_left<L, R> || is_axpby_right_right<L, R> || is_axpby_left_right<L, R> || is_axpby_right_left<L, R>;

template <typename L, typename R>
static constexpr bool is_axpy_left_left = is_axpy_left_left_impl<L, R>::value && !is_axpby<L, R>;

template <typename L, typename R>
static constexpr bool is_axpy_left_right = is_axpy_left_right_impl<L, R>::value && !is_axpby<L, R>;

template <typename L, typename R>
static constexpr bool is_axpy_right_left = is_axpy_right_left_impl<L, R>::value && !is_axpby<L, R>;

template <typename L, typename R>
static constexpr bool is_axpy_right_right = is_axpy_right_right_impl<L, R>::value && !is_axpby<L, R>;

template <typename L, typename R>
static constexpr bool is_axpy = is_axpy_left_left<L, R> || is_axpy_left_right<L, R> || is_axpy_right_left<L, R> || is_axpy_right_right<L, R>;

template <typename L, typename R>
static constexpr bool is_special_plus = is_axpy<L, R> || is_axpby<L, R>;

/*!
 * \brief Binary operator for scalar addition
 */
template <typename T>
struct plus_binary_op {
    static constexpr bool linear      = true;  ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe = true;  ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func   = false; ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = true;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template <typename L, typename R>
    static constexpr bool gpu_computable =
        ((!is_scalar<L> && !is_scalar<R>)&&((is_single_precision_t<T> && impl::egblas::has_saxpy_3 && impl::egblas::has_saxpby_3)
                                            || (is_double_precision_t<T> && impl::egblas::has_daxpy_3 && impl::egblas::has_daxpby_3)
                                            || (is_complex_single_t<T> && impl::egblas::has_caxpy_3 && impl::egblas::has_caxpby_3)
                                            || (is_complex_double_t<T> && impl::egblas::has_zaxpy_3 && impl::egblas::has_zaxpby_3)))
        || ((is_scalar<L> != is_scalar<R>)&&((is_single_precision_t<T> && impl::egblas::has_scalar_sadd)
                                             || (is_double_precision_t<T> && impl::egblas::has_scalar_dadd)
                                             || (is_complex_single_t<T> && impl::egblas::has_scalar_cadd)
                                             || (is_complex_double_t<T> && impl::egblas::has_scalar_zadd)));

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
        return lhs + rhs;
    }

    /*!
     * \brief Compute several applications of the operator at a time
     * \param lhs The left hand side vector
     * \param rhs The right hand side vector
     * \tparam V The vectorization mode
     * \return a vector containing several results of the operator
     */
    template <typename V = default_vec>
    static ETL_STRONG_INLINE(vec_type<V>) load(const vec_type<V>& lhs, const vec_type<V>& rhs) noexcept {
        return V::add(lhs, rhs);
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
        if constexpr (!is_scalar<L> && !is_scalar<R> && is_axpy_left_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs)>;

            impl::egblas::axpy_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (!is_scalar<L> && !is_scalar<R> && is_axpy_left_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs)>;

            impl::egblas::axpy_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axpy_right_left<L, R>) {
            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(rhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(lhs, yy);

            constexpr auto incx = gpu_inc<decltype(rhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(lhs)>;

            impl::egblas::axpy_3(etl::size(yy), rhs_lhs.value, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axpy_right_right<L, R>) {
            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(rhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(lhs, yy);

            constexpr auto incx = gpu_inc<decltype(rhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(lhs)>;

            impl::egblas::axpy_3(etl::size(yy), rhs_rhs.value, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axpby_left_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::axpby_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axpby_left_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_rhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::axpby_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axpby_right_left<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_rhs)>;

            impl::egblas::axpby_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, rhs_lhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_axpby_right_right<L, R>) {
            auto& lhs_lhs = lhs.get_lhs();
            auto& lhs_rhs = lhs.get_rhs();

            auto& rhs_lhs = rhs.get_lhs();
            auto& rhs_rhs = rhs.get_rhs();

            decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs_lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs_lhs)>;

            impl::egblas::axpby_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), incx, rhs_rhs.value, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (!is_scalar<L> && !is_scalar<R> && !is_special_plus<L, R>) {
            decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
            decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

            constexpr auto incx = gpu_inc<decltype(lhs)>;
            constexpr auto incy = gpu_inc<decltype(rhs)>;

            value_t<L> alpha(1);
            impl::egblas::axpy_3(etl::size(yy), alpha, x.gpu_memory(), incx, y.gpu_memory(), incy, yy.gpu_memory(), 1);
        } else if constexpr (is_scalar<L> && !is_scalar<R>) {
            auto s = lhs.value;

            smart_gpu_compute(rhs, yy);

            impl::egblas::scalar_add(yy.gpu_memory(), etl::size(yy), 1, s);
        } else if constexpr (!is_scalar<L> && is_scalar<R>) {
            auto s = rhs.value;

            smart_gpu_compute(lhs, yy);

            impl::egblas::scalar_add(yy.gpu_memory(), etl::size(yy), 1, s);
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
        return "+";
    }
};

} //end of namespace etl
