//=======================================================================
// Copyright (c) 2014-2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#pragma once


namespace etl {

/*!
 * \brief Binary operator for scalar subtraction
 */
template <typename T>
struct minus_binary_op {
    static constexpr bool linear         = true;           ///< Indicates if the operator is linear or not
    static constexpr bool thread_safe    = true;           ///< Indicates if the operator is thread safe or not
    static constexpr bool desc_func      = false;          ///< Indicates if the description must be printed as function

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
    template<typename L, typename R>
    static constexpr bool gpu_computable =
            ((!is_scalar<L> && !is_scalar<R>) && cublas_enabled)
        ||  (
                    (is_scalar<L> != is_scalar<R>)
                &&  (
                            (is_single_precision_t<T> && impl::egblas::has_scalar_sadd && cublas_enabled)
                        ||  (is_double_precision_t<T> && impl::egblas::has_scalar_dadd && cublas_enabled)
                    )
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
        return lhs - rhs;
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
        return V::sub(lhs, rhs);
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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axpy_right_left<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(rhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(lhs, yy);

        impl::egblas::axpy_3(etl::size(yy), -rhs_lhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axpy_right_right<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(rhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(lhs, yy);

        impl::egblas::axpy_3(etl::size(yy), -rhs_rhs.value, x.gpu_memory(), 1, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axpy_left_left<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::axpby_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), 1, T(-1), y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axpy_left_right<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::axpby_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), 1, T(-1), y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axpby_left_left<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::axpby_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), 1, T(-1) * rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axpby_left_right<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_rhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::axpby_3(etl::size(yy), lhs_lhs.value, x.gpu_memory(), 1, T(-1) * rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axpby_right_left<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_rhs, yy);

        impl::egblas::axpby_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), 1, T(-1) * rhs_lhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(is_axpby_right_right<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        auto& lhs_lhs = lhs.get_lhs();
        auto& lhs_rhs = lhs.get_rhs();

        auto& rhs_lhs = rhs.get_lhs();
        auto& rhs_rhs = rhs.get_rhs();

        decltype(auto) x = smart_gpu_compute_hint(lhs_lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs_lhs, yy);

        impl::egblas::axpby_3(etl::size(yy), lhs_rhs.value, x.gpu_memory(), 1, T(-1) * rhs_rhs.value, y.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(!is_scalar<L> && !is_scalar<R> && !is_special<L,R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& yy) noexcept {
        decltype(auto) x = smart_gpu_compute_hint(lhs, yy);
        decltype(auto) y = smart_gpu_compute_hint(rhs, yy);

        impl::egblas::axpy_3(etl::size(y), value_t<L>(-1), y.gpu_memory(), 1, x.gpu_memory(), 1, yy.gpu_memory(), 1);

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
    template <typename L, typename R, typename Y, cpp_enable_iff(!is_scalar<L> && is_scalar<R>)>
    static Y& gpu_compute(const L& lhs, const R& rhs, Y& y) noexcept {
        auto s = -rhs.value;

        smart_gpu_compute(lhs, y);

        impl::egblas::scalar_add(y.gpu_memory(), etl::size(y), 1, s);

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

        impl::egblas::scalar_mul(y.gpu_memory(), etl::size(y), 1, value_t<L>(-1));
        impl::egblas::scalar_add(y.gpu_memory(), etl::size(y), 1, s);

        y.validate_gpu();
        y.invalidate_cpu();

        return y;
    }

    /*!
     * \brief Returns a textual representation of the operator
     * \return a string representing the operator
     */
    static std::string desc() noexcept {
        return "-";
    }
};

} //end of namespace etl
