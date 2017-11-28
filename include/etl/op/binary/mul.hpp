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

// detect 1.0 * (x * y)

template <typename L, typename R>
struct is_axmy_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename LeftExpr, typename RightExpr>
struct is_axmy_left_impl <etl::scalar<T0>, binary_expr<T1, LeftExpr, etl::mul_binary_op<T2>, RightExpr>> {
    static constexpr bool value = true;
};

// detect (x * y) * 1.0

template <typename L, typename R>
struct is_axmy_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename LeftExpr, typename RightExpr>
struct is_axmy_right_impl <binary_expr<T1, LeftExpr, etl::mul_binary_op<T2>, RightExpr>, etl::scalar<T0>> {
    static constexpr bool value = true;
};

// detect (1.0 * x) * y

template <typename L, typename R>
struct is_axmy_left_left_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename RightExpr, typename R>
struct is_axmy_left_left_impl <binary_expr<T1, etl::scalar<T0>, etl::mul_binary_op<T2>, RightExpr>, R> {
    static constexpr bool value = true;
};

// detect (x * 1.0) * y

template <typename L, typename R>
struct is_axmy_left_right_impl {
    static constexpr bool value = false;
};

template <typename T0, typename T1, typename T2, typename LeftExpr, typename R>
struct is_axmy_left_right_impl <binary_expr<T1, LeftExpr, etl::mul_binary_op<T2>, etl::scalar<T0>>, R> {
    static constexpr bool value = !is_scalar<R>;
};

// detect x * (1.0 * y)

template <typename L, typename R>
struct is_axmy_right_left_impl {
    static constexpr bool value = false;
};

template <typename L, typename T0, typename T1, typename T2, typename RightExpr>
struct is_axmy_right_left_impl <L, binary_expr<T0, etl::scalar<T1>, etl::mul_binary_op<T2>, RightExpr>> {
    static constexpr bool value = !is_scalar<L>;
};

// detect x * (1.0 * y)

template <typename L, typename R>
struct is_axmy_right_right_impl {
    static constexpr bool value = false;
};

template <typename L, typename T0, typename T1, typename T2, typename RightExpr>
struct is_axmy_right_right_impl <L, binary_expr<T0, RightExpr, etl::mul_binary_op<T2>, etl::scalar<T1>>> {
    static constexpr bool value = true;
};

// Variable templates helper

template <typename L, typename R>
static constexpr bool is_axmy_left = is_axmy_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axmy_right = is_axmy_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axmy_left_left = is_axmy_left_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axmy_left_right = is_axmy_left_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axmy_right_left = is_axmy_right_left_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axmy_right_right = is_axmy_right_right_impl<L, R>::value;

template <typename L, typename R>
static constexpr bool is_axmy = is_axmy_left<L, R> || is_axmy_right<L, R> || is_axmy_left_left<L, R> || is_axmy_left_right<L, R> || is_axmy_right_left<L, R> || is_axmy_right_right<L, R>;

/*!
 * \brief Binary operator for scalar multiplication
 */
template <typename T>
struct mul_binary_op {
    static constexpr bool linear         = true;                               ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe    = true;                               ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func      = false;                              ///< Indicates if the description must be printed as function

    /*!
     * \brief Indicates if the expression is vectorizable using the
     * given vector mode
     * \tparam V The vector mode
     */
    template <vector_mode_t V>
    static constexpr bool vectorizable = V == vector_mode_t::AVX512 ? !is_complex_t<T> : true;

    /*!
     * \brief Indicates if the operator can be computed on GPU
     */
    template<typename L, typename R>
    static constexpr bool gpu_computable =
                (is_single_precision_t<T> && impl::egblas::has_saxmy)
            ||  (is_double_precision_t<T> && impl::egblas::has_daxmy)
            ||  (is_complex_single_t<T> && impl::egblas::has_caxmy)
            ||  (is_complex_double_t<T> && impl::egblas::has_zaxmy)
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
        return lhs * rhs;
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
        return V::mul(lhs, rhs);
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
    template <typename L, typename R, typename YY, cpp_enable_iff(is_axmy_left<L, R>)>
    static YY& gpu_compute(const L& lhs, const R& rhs, YY& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(rhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::axmy_3(etl::size(yy), lhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename YY, cpp_enable_iff(is_axmy_right<L, R>)>
    static YY& gpu_compute(const L& lhs, const R& rhs, YY& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(lhs_rhs, yy);

        impl::egblas::axmy_3(etl::size(yy), rhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename YY, cpp_enable_iff(is_axmy_left_left<L, R>)>
    static YY& gpu_compute(const L& lhs, const R& rhs, YY& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::axmy_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename YY, cpp_enable_iff(is_axmy_left_right<L, R>)>
    static YY& gpu_compute(const L& lhs, const R& rhs, YY& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::axmy_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename YY, cpp_enable_iff(is_axmy_right_left<L, R>)>
    static YY& gpu_compute(const L& lhs, const R& rhs, YY& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::axmy_3(etl::size(yy), rhs_lhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename YY, cpp_enable_iff(is_axmy_right_right<L, R>)>
    static YY& gpu_compute(const L& lhs, const R& rhs, YY& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::axmy_3(etl::size(yy), rhs_rhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(!is_scalar<L> && !is_scalar<R> && !is_axmy<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::axmy_3(etl::size(yy), 1, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_scalar<L> && !is_scalar<R> && !is_axmy<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        smart_gpu_compute(rhs, yy);

        impl::egblas::scalar_mul(yy.gpu_memory(), etl::size(yy), 1, lhs.value);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(!is_scalar<L> && is_scalar<R> && !is_axmy<L, R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        smart_gpu_compute(lhs, yy);

        impl::egblas::scalar_mul(yy.gpu_memory(), etl::size(yy), 1, rhs.value);

        yy.validate_gpu();
        yy.invalidate_cpu();

        return yy;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "*";
    }
};

} //end of namespace etl
